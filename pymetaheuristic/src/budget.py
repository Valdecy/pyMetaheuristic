"""Strict function-evaluation budget utilities.

The budget wrapper is the authoritative counter for objective-function calls.
It raises *before* a call that would exceed the configured limit, so every
execution mode can guarantee::

    actual_objective_calls <= max_evaluations

Budget exhaustion is an internal control-flow event.  The default engine policy
is ``return_best``: retain the best candidate that was genuinely evaluated and
return a normal result instead of exposing the exception to the user.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before an objective evaluation would exceed a strict FE budget."""


@dataclass
class SharedEvaluationBudget:
    """Callable objective wrapper with a strict evaluation counter.

    Parameters
    ----------
    target_function:
        User objective function.
    max_evaluations:
        Maximum number of actual calls. ``None`` disables enforcement.
    objective_name:
        Human-readable name used in error messages.
    objective:
        Optimization direction, ``"min"`` or ``"max"``.  This is used only
        for a conservative raw-objective fallback when an engine is interrupted
        before it can return an initialized state.  ProblemSpec's cached,
        constraint-aware fitness remains the preferred source of truth.

    Notes
    -----
    ``active_label`` is optional. Collaborative runners use it to attribute
    actual evaluations to islands; ordinary runs use the algorithm ID.
    """

    target_function: Callable[[Any], float]
    max_evaluations: int | None
    objective_name: str = "objective"
    objective: str = "min"
    used: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_label_category: dict[str, dict[str, int]] = field(default_factory=dict)
    successful_by_label: dict[str, int] = field(default_factory=dict)
    best_raw_by_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_label: str | None = None
    active_category: str | None = None

    def __post_init__(self) -> None:
        if self.max_evaluations is not None:
            self.max_evaluations = int(self.max_evaluations)
            if self.max_evaluations < 1:
                raise ValueError("max_evaluations must be >= 1 when a strict FE budget is used.")
        self.objective = str(self.objective or "min").lower()
        if self.objective not in {"min", "max"}:
            raise ValueError("objective must be 'min' or 'max'.")

    @property
    def enabled(self) -> bool:
        return self.max_evaluations is not None

    @property
    def remaining(self) -> int | None:
        if self.max_evaluations is None:
            return None
        return max(0, int(self.max_evaluations) - int(self.used))

    @property
    def exhausted(self) -> bool:
        return self.max_evaluations is not None and self.used >= self.max_evaluations

    def used_for(self, label: str | None) -> int:
        """Return actual objective calls attributed to ``label``."""
        if label is None:
            return int(self.used)
        return int(self.by_label.get(str(label), 0))

    def successful_for(self, label: str | None) -> int:
        """Return successful objective calls attributed to ``label``."""
        if label is None:
            return int(sum(self.successful_by_label.values()))
        return int(self.successful_by_label.get(str(label), 0))

    def used_for_category(self, category: str | None) -> int:
        """Return actual objective calls attributed to ``category``."""
        if category is None:
            return int(self.used)
        return int(self.by_category.get(str(category), 0))

    def used_for_label_category(self, label: str | None, category: str | None) -> int:
        """Return calls attributed jointly to an island label and category."""
        label_key = "unlabelled" if label is None else str(label)
        category_key = "other" if category is None else str(category)
        return int(self.by_label_category.get(label_key, {}).get(category_key, 0))

    def best_raw_for(self, label: str | None) -> dict[str, Any] | None:
        """Return a defensive copy of the best successful raw evaluation."""
        key = "unlabelled" if label is None else str(label)
        record = self.best_raw_by_label.get(key)
        return None if record is None else deepcopy(record)

    @contextmanager
    def use_label(self, label: str | None) -> Iterator["SharedEvaluationBudget"]:
        previous = self.active_label
        self.active_label = None if label is None else str(label)
        try:
            yield self
        finally:
            self.active_label = previous

    @contextmanager
    def use_category(self, category: str | None) -> Iterator["SharedEvaluationBudget"]:
        """Temporarily attribute evaluations to a high-level activity category."""
        previous = self.active_category
        self.active_category = None if category is None else str(category)
        try:
            yield self
        finally:
            self.active_category = previous

    def _record_success(self, label: str, position: Any, value: Any) -> None:
        try:
            fitness = float(value)
        except Exception:
            return
        self.successful_by_label[label] = int(self.successful_by_label.get(label, 0)) + 1
        candidate = {
            "position": deepcopy(position),
            "raw_fitness": fitness,
            "fitness": fitness,
            "violation": 0.0,
            "is_feasible": True,
            "handler": "none",
        }
        incumbent = self.best_raw_by_label.get(label)
        if incumbent is None:
            self.best_raw_by_label[label] = candidate
            return
        incumbent_fitness = float(incumbent["fitness"])
        better = fitness < incumbent_fitness if self.objective == "min" else fitness > incumbent_fitness
        if better:
            self.best_raw_by_label[label] = candidate

    def __call__(self, position):
        if self.max_evaluations is not None and self.used >= self.max_evaluations:
            raise EvaluationBudgetExceeded(
                f"max_evaluations={self.max_evaluations} exhausted before evaluating "
                f"{self.objective_name}."
            )

        label = self.active_label or "unlabelled"
        category = self.active_category or "other"
        # Count the actual attempted user-objective call immediately before
        # invocation. Even objectives that raise consume computational budget.
        self.used += 1
        self.by_label[label] = int(self.by_label.get(label, 0)) + 1
        self.by_category[category] = int(self.by_category.get(category, 0)) + 1
        label_categories = self.by_label_category.setdefault(label, {})
        label_categories[category] = int(label_categories.get(category, 0)) + 1
        value = self.target_function(position)
        self._record_success(label, position, value)
        return value


@dataclass
class FairEvaluationScheduler:
    """Central least-normalized-FE scheduler for island systems.

    The scheduler does not inspect or modify engine internals. It chooses the
    active island with the smallest accumulated evaluation count divided by its
    optional positive weight. Ties are broken by a deterministic rotating
    cursor, avoiding a permanent advantage for the first island in the list.

    Because engines execute native, indivisible ``step()`` calls, exact equality
    is not guaranteed. The scheduler instead minimizes avoidable imbalance and
    lets the shared hard budget remain the final authority.
    """

    labels: Sequence[str]
    weights: Mapping[str, float] | None = None
    cursor: int = 0
    selection_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = [str(label) for label in self.labels]
        if not self.labels:
            raise ValueError("FairEvaluationScheduler requires at least one label.")
        supplied = dict(self.weights or {})
        normalized: dict[str, float] = {}
        for label in self.labels:
            weight = float(supplied.get(label, 1.0))
            if weight <= 0.0:
                raise ValueError(f"Scheduler weight for {label!r} must be > 0.")
            normalized[label] = weight
            self.selection_counts.setdefault(label, 0)
        self.weights = normalized
        self.cursor %= len(self.labels)

    @property
    def policy_name(self) -> str:
        return "least_normalized_evaluations_first"

    @property
    def explanation(self) -> str:
        return (
            "The runner selects the active island with the smallest accumulated "
            "objective-evaluation count divided by its scheduler weight. Equal "
            "weights therefore approximate equal FE access. Ties rotate across "
            "islands. Each selected island still executes one native engine step, "
            "so residual imbalance can remain when step sizes differ."
        )

    def score(self, label: str, evaluations_by_label: Mapping[str, int]) -> float:
        label = str(label)
        return float(evaluations_by_label.get(label, 0)) / float(self.weights.get(label, 1.0))

    def select(self, active_labels: Sequence[str], evaluations_by_label: Mapping[str, int]) -> str:
        active = {str(label) for label in active_labels}
        if not active:
            raise ValueError("Cannot select from an empty active-island set.")

        ordered_active = [label for label in self.labels if label in active]
        if not ordered_active:
            # Defensive support for labels added after construction.
            ordered_active = sorted(active)

        scores = {label: self.score(label, evaluations_by_label) for label in ordered_active}
        minimum = min(scores.values())
        tolerance = 1e-12 * max(1.0, abs(minimum))
        tied = {label for label, value in scores.items() if abs(value - minimum) <= tolerance}

        selected = None
        n_labels = len(self.labels)
        for offset in range(n_labels):
            candidate = self.labels[(self.cursor + offset) % n_labels]
            if candidate in tied:
                selected = candidate
                break
        if selected is None:
            selected = min(tied)

        if selected in self.labels:
            self.cursor = (self.labels.index(selected) + 1) % n_labels
        self.selection_counts[selected] = int(self.selection_counts.get(selected, 0)) + 1
        return selected


def format_budget_report(
    budget: SharedEvaluationBudget | None,
    *,
    labels: Sequence[str] | None = None,
    scheduler: FairEvaluationScheduler | None = None,
    requested_budget: int | None = None,
    prefix: str = "budget",
) -> str:
    """Build a concise, user-facing explanation of FE consumption."""
    if budget is None:
        return f"[{prefix}] No strict function-evaluation budget was configured."

    requested = budget.max_evaluations if requested_budget is None else requested_budget
    requested = None if requested is None else int(requested)
    used = int(budget.used)
    remaining = None if requested is None else max(0, requested - used)
    utilization = None if requested in (None, 0) else 100.0 * used / requested

    policy = scheduler.policy_name if scheduler is not None else "shared_hard_cap"
    lines = [f"[{prefix}] policy={policy}"]
    if scheduler is not None:
        lines.append(f"[{prefix}] {scheduler.explanation}")
    if requested is None:
        lines.append(f"[{prefix}] consumed={used} FEs")
    else:
        lines.append(
            f"[{prefix}] consumed={used}/{requested} FEs "
            f"({utilization:.2f}% utilization), remaining={remaining}"
        )

    category_parts = [
        f"{name}={count}"
        for name, count in sorted(budget.by_category.items())
        if int(count) > 0
    ]
    if category_parts:
        lines.append(f"[{prefix}] categories: " + ", ".join(category_parts))

    ordered_labels = [str(label) for label in (labels or [])]
    extras = sorted(set(budget.by_label) - set(ordered_labels))
    ordered_labels.extend(extras)
    if ordered_labels:
        lines.append(f"[{prefix}] by island:")
        for label in ordered_labels:
            count = int(budget.by_label.get(label, 0))
            share = 0.0 if used <= 0 else 100.0 * count / used
            category_map = budget.by_label_category.get(label, {})
            category_text = ", ".join(
                f"{name}={value}"
                for name, value in sorted(category_map.items())
                if int(value) > 0
            )
            selections = None if scheduler is None else scheduler.selection_counts.get(label, 0)
            suffix_parts = []
            if selections is not None:
                suffix_parts.append(f"scheduled_steps={selections}")
            if category_text:
                suffix_parts.append(category_text)
            suffix = f"; {'; '.join(suffix_parts)}" if suffix_parts else ""
            lines.append(f"  - {label}: {count} FEs ({share:.2f}%){suffix}")

    named_counts = [int(budget.by_label.get(label, 0)) for label in (labels or [])]
    if named_counts:
        gap = max(named_counts) - min(named_counts)
        lines.append(
            f"[{prefix}] final island FE gap={gap}. This gap can remain because "
            "initialization, migration/orchestration, and native engine steps consume "
            "indivisible numbers of evaluations; the scheduler compensates on later turns "
            "but does not rewrite engine behavior."
        )
    return "\n".join(lines)


def make_shared_budget(
    target_function,
    max_evaluations: int | None,
    *,
    objective_name: str = "objective",
    objective: str = "min",
) -> SharedEvaluationBudget | None:
    """Return a strict wrapper, or ``None`` when no FE limit is requested."""
    if max_evaluations is None:
        return None
    return SharedEvaluationBudget(
        target_function=target_function,
        max_evaluations=max_evaluations,
        objective_name=objective_name,
        objective=objective,
    )


__all__ = [
    "EvaluationBudgetExceeded",
    "FairEvaluationScheduler",
    "SharedEvaluationBudget",
    "format_budget_report",
    "make_shared_budget",
]
