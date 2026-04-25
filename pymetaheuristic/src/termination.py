"""
pyMetaheuristic src — Composable Termination Criteria
======================================================

Supports four independent stopping conditions that can be combined freely.
The first condition that triggers wins.

Conditions
----------
* max_steps       : Maximum number of algorithm macro-steps / iterations.
* max_evaluations : Maximum number of objective-function evaluations.
* max_time        : Wall-clock time bound in seconds.
* max_early_stop  : Early stopping — halt if global best has not improved
                    by more than *epsilon* for this many consecutive steps.

Usage
-----
    from pymetaheuristic.src.termination import Termination

    term = Termination(max_steps=500, max_evaluations=50_000, max_time=30.0)
    result = pymetaheuristic.optimize(
        algorithm="pso", target_function=fn,
        min_values=lb, max_values=ub,
        termination=term,
    )

    # or pass as a dict — same effect:
    result = pymetaheuristic.optimize(
        ...
        termination={"max_steps": 500, "max_evaluations": 50_000},
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Termination:
    """
    Composable stopping condition object.

    Parameters
    ----------
    max_steps : int | None
        Maximum number of algorithm macro-steps (iterations / epochs).
        Mirrors ``EngineConfig.max_steps``.
    max_evaluations : int | None
        Maximum number of objective-function evaluations.
        Mirrors ``EngineConfig.max_evaluations``.
    max_time : float | None
        Wall-clock time limit in seconds.
        Mirrors ``EngineConfig.timeout_seconds``.
    max_early_stop : int | None
        Stop if the global best has not improved by more than *epsilon*
        for this many consecutive steps.
    epsilon : float
        Minimum improvement threshold for the early-stopping criterion
        (default 1e-10).
    target_fitness : float | None
        Stop immediately once this fitness value is reached or exceeded
        (direction-aware: min → reached when best ≤ target;
        max → reached when best ≥ target).
        Mirrors ``EngineConfig.target_fitness``.

    Notes
    -----
    At least one condition must be set; otherwise the constructor raises
    ``ValueError``.

    A ``Termination`` object can be passed directly to ``optimize()``,
    ``cooperative_optimize()``, and ``orchestrated_optimize()`` via the
    *termination* keyword argument.  It can also be converted to / from a
    plain dict with ``to_dict()`` / ``Termination.from_dict()``.
    """

    max_steps:       int   | None = None
    max_evaluations: int   | None = None
    max_time:        float | None = None
    max_early_stop:  int   | None = None
    epsilon:         float        = 1e-10
    target_fitness:  float | None = None

    # ------------------------------------------------------------------ #
    # Internal tracking — reset on every call to start()
    # ------------------------------------------------------------------ #
    _start_time:        float | None = field(default=None, init=False, repr=False)
    _start_step:        int          = field(default=0,    init=False, repr=False)
    _start_evaluations: int          = field(default=0,    init=False, repr=False)
    _stagnation_count:  int          = field(default=0,    init=False, repr=False)
    _last_best:         float | None = field(default=None, init=False, repr=False)
    _triggered_reason:  str  | None  = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if all(v is None for v in (
            self.max_steps, self.max_evaluations,
            self.max_time, self.max_early_stop, self.target_fitness,
        )):
            raise ValueError(
                "Termination: at least one stopping condition must be set. "
                "Provide max_steps, max_evaluations, max_time, "
                "max_early_stop, or target_fitness."
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError("max_time must be > 0")
        if self.max_early_stop is not None and self.max_early_stop < 1:
            raise ValueError("max_early_stop must be >= 1")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self, step: int = 0, evaluations: int = 0) -> None:
        """
        Record the start reference point.  Must be called before the run loop.

        Parameters
        ----------
        step : int
            Current step counter at start (usually 0).
        evaluations : int
            Current evaluation counter at start (usually 0).
        """
        self._start_time        = time.perf_counter()
        self._start_step        = step
        self._start_evaluations = evaluations
        self._stagnation_count  = 0
        self._last_best         = None
        self._triggered_reason  = None

    def update(self, best_fitness: float | None) -> None:
        """
        Update early-stopping stagnation counter.

        Call once per step, passing the current global best fitness.
        """
        if self.max_early_stop is None or best_fitness is None:
            return
        if self._last_best is None:
            self._last_best = best_fitness
            self._stagnation_count = 0
            return
        improvement = abs(best_fitness - self._last_best)
        if improvement <= self.epsilon:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
            self._last_best = best_fitness

    def should_stop(
        self,
        step: int,
        evaluations: int,
        best_fitness: float | None = None,
        objective: str = "min",
    ) -> bool:
        """
        Return True if any active stopping condition has been triggered.

        Parameters
        ----------
        step        : current step counter
        evaluations : current evaluation counter
        best_fitness: current global best (required for ES / target checks)
        objective   : "min" or "max" (used for target_fitness direction)
        """
        # MG — maximum generations / steps
        if self.max_steps is not None:
            if (step - self._start_step) >= self.max_steps:
                self._triggered_reason = "max_steps"
                return True

        # FE — maximum function evaluations
        if self.max_evaluations is not None:
            if (evaluations - self._start_evaluations) >= self.max_evaluations:
                self._triggered_reason = "max_evaluations"
                return True

        # TB — time bound
        if self.max_time is not None and self._start_time is not None:
            if (time.perf_counter() - self._start_time) >= self.max_time:
                self._triggered_reason = "max_time"
                return True

        # ES — early stopping (stagnation)
        if self.max_early_stop is not None:
            if self._stagnation_count >= self.max_early_stop:
                self._triggered_reason = "max_early_stop"
                return True

        # TF — target fitness
        if self.target_fitness is not None and best_fitness is not None:
            if objective == "min" and best_fitness <= self.target_fitness:
                self._triggered_reason = "target_fitness"
                return True
            if objective == "max" and best_fitness >= self.target_fitness:
                self._triggered_reason = "target_fitness"
                return True

        return False

    @property
    def triggered_reason(self) -> str | None:
        """Return the reason that triggered termination, or None if not yet triggered."""
        return self._triggered_reason

    @property
    def elapsed_time(self) -> float | None:
        """Wall-clock seconds since start() was called, or None if not started."""
        if self._start_time is None:
            return None
        return time.perf_counter() - self._start_time

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return {
            "max_steps":       self.max_steps,
            "max_evaluations": self.max_evaluations,
            "max_time":        self.max_time,
            "max_early_stop":  self.max_early_stop,
            "epsilon":         self.epsilon,
            "target_fitness":  self.target_fitness,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Termination":
        """Reconstruct from a plain dictionary."""
        return cls(
            max_steps       = d.get("max_steps"),
            max_evaluations = d.get("max_evaluations"),
            max_time        = d.get("max_time"),
            max_early_stop  = d.get("max_early_stop"),
            epsilon         = d.get("epsilon", 1e-10),
            target_fitness  = d.get("target_fitness"),
        )

    @classmethod
    def _from_any(cls, obj: "Termination | dict | None") -> "Termination | None":
        """Internal: normalise a user-supplied termination spec."""
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        raise TypeError(
            f"termination must be a Termination instance or a dict, got {type(obj)}"
        )

    def __repr__(self) -> str:
        parts = []
        if self.max_steps       is not None: parts.append(f"max_steps={self.max_steps}")
        if self.max_evaluations is not None: parts.append(f"max_evaluations={self.max_evaluations}")
        if self.max_time        is not None: parts.append(f"max_time={self.max_time}s")
        if self.max_early_stop  is not None: parts.append(f"max_early_stop={self.max_early_stop}(ε={self.epsilon})")
        if self.target_fitness  is not None: parts.append(f"target_fitness={self.target_fitness}")
        return f"Termination({', '.join(parts)})"
