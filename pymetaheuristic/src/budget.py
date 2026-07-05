"""Shared function-evaluation budget utilities for collaborative runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before an objective evaluation would exceed a shared FE budget."""


@dataclass
class SharedEvaluationBudget:
    """Callable objective wrapper with a strict global evaluation counter.

    The wrapper is intentionally small: it counts *actual* calls to the user
    objective and raises before the first call that would exceed ``max_evaluations``.
    Runners set ``active_label`` while initializing or stepping an island so the
    final report can expose per-island FE usage without relying on heterogeneous
    engine-local counters.
    """

    target_function: Callable[[Any], float]
    max_evaluations: int | None
    objective_name: str = "objective"
    used: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    active_label: str | None = None

    def __post_init__(self) -> None:
        if self.max_evaluations is not None:
            self.max_evaluations = int(self.max_evaluations)
            if self.max_evaluations < 1:
                raise ValueError("max_evaluations must be >= 1 when a shared FE budget is used.")

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

    @contextmanager
    def use_label(self, label: str | None) -> Iterator["SharedEvaluationBudget"]:
        previous = self.active_label
        self.active_label = label
        try:
            yield self
        finally:
            self.active_label = previous

    def __call__(self, position):
        if self.max_evaluations is not None and self.used >= self.max_evaluations:
            raise EvaluationBudgetExceeded(
                f"Shared max_evaluations={self.max_evaluations} exhausted before evaluating {self.objective_name}."
            )
        label = self.active_label or "unlabelled"
        self.used += 1
        self.by_label[label] = int(self.by_label.get(label, 0)) + 1
        return self.target_function(position)


def make_shared_budget(target_function, max_evaluations: int | None) -> SharedEvaluationBudget | None:
    if max_evaluations is None:
        return None
    return SharedEvaluationBudget(target_function=target_function, max_evaluations=max_evaluations)
