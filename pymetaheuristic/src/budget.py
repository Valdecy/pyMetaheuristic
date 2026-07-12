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
from typing import Any, Callable, Iterator


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
    successful_by_label: dict[str, int] = field(default_factory=dict)
    best_raw_by_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_label: str | None = None

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
        # Count the actual attempted user-objective call immediately before
        # invocation. Even objectives that raise consume computational budget.
        self.used += 1
        self.by_label[label] = int(self.by_label.get(label, 0)) + 1
        value = self.target_function(position)
        self._record_success(label, position, value)
        return value


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
    "SharedEvaluationBudget",
    "make_shared_budget",
]
