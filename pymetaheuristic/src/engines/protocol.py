"""
pyMetaheuristic src — Core Engine Protocol
==========================================
Author: Valdecy Pereira  (protocol design + reference implementation)

Design principles
-----------------
1. Preserve native instinct of each algorithm.
2. Mandatory universal unit: best-so-far.
3. Mandatory progression unit: one native macro-step.
4. Track both steps and evaluations; benchmark on max_evaluations.
5. Capability flags — never force.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence
from types import MethodType
import numpy as np

from ..budget import EvaluationBudgetExceeded, SharedEvaluationBudget


# ---------------------------------------------------------------------------
# 1. ProblemSpec
# ---------------------------------------------------------------------------

@dataclass
class ProblemSpec:
    """Describes the optimisation problem, independently of any algorithm."""

    target_function: Callable[[list[float]], float]
    min_values:      list[float]
    max_values:      list[float]
    objective:       str                          = "min"   # "min" | "max"
    constraints:     list[Callable] | None        = None
    constraint_handler: str | None               = None
    variable_types:  list[str] | None            = None    # "float","int","binary"
    metadata:        dict[str, Any]              = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.min_values)

    def is_better(self, a: float, b: float) -> bool:
        """Return True if fitness *a* is strictly better than *b*."""
        return a < b if self.objective == "min" else a > b

    def worst_fitness(self) -> float:
        return float("inf") if self.objective == "min" else float("-inf")

    # ------------------------- constraint helpers -------------------------

    @property
    def has_constraints(self) -> bool:
        return bool(self.constraints)

    def _handler(self) -> str:
        return (self.constraint_handler or "none").lower()

    def _penalty_coefficient(self) -> float:
        return float(self.metadata.get("penalty_coefficient", 1e6))

    def _equality_tolerance(self) -> float:
        return float(self.metadata.get("equality_tolerance", 1e-6))

    def _resample_attempts(self) -> int:
        return int(self.metadata.get("resample_attempts", 25))

    def _normalize_variable_types(self) -> list[Any]:
        """Return one variable-type descriptor per dimension.

        Supported descriptors are intentionally permissive so old scripts keep
        working and mixed/discrete benchmark wrappers can be explicit:

        - None, "float", "real", "continuous": continuous variable.
        - "int", "integer", "discrete": nearest integer.
        - "binary", "bool", "boolean": 0/1 variable.
        - "step:<h>" or "step:<h>:round|ceil|floor": multiple of h.
        - dict(type="step", step=h, mode="round|ceil|floor").
        """
        vt = self.variable_types
        if vt is None:
            return [None] * self.dimension
        if isinstance(vt, str):
            vt = [vt] * self.dimension
        else:
            vt = list(vt)
        if len(vt) == 1 and self.dimension != 1:
            vt = vt * self.dimension
        if len(vt) != self.dimension:
            raise ValueError(
                f"variable_types must have length {self.dimension}; received {len(vt)}"
            )
        return vt

    @staticmethod
    def _parse_step_descriptor(descriptor: Any) -> tuple[float | None, str]:
        if isinstance(descriptor, dict):
            step = descriptor.get("step", descriptor.get("quantum", descriptor.get("multiple")))
            mode = str(descriptor.get("mode", descriptor.get("rounding", "round"))).lower()
            return (None if step is None else float(step), mode)
        text = str(descriptor).strip().lower()
        # Accept: step:0.0625, step:0.0625:ceil, multiple:0.0625:floor
        if text.startswith(("step:", "multiple:", "quantum:")):
            parts = text.split(":")
            step = float(parts[1]) if len(parts) >= 2 and parts[1] else None
            mode = parts[2] if len(parts) >= 3 and parts[2] else "round"
            return step, mode
        return None, "round"

    def apply_variable_types(self, position) -> np.ndarray:
        """Clip a candidate and project it onto declared variable domains.

        This is deliberately implemented in ProblemSpec rather than in each
        individual engine so solo optimizers, island systems, candidate
        injection, callbacks, and benchmark wrappers all use the same decoding
        rule.  The returned array is always numeric and has the same dimension
        as the problem.
        """
        pos = np.asarray(position, dtype=float).copy()
        lo = np.asarray(self.min_values, dtype=float)
        hi = np.asarray(self.max_values, dtype=float)
        pos = np.clip(pos, lo, hi)
        for i, descriptor in enumerate(self._normalize_variable_types()):
            if descriptor is None:
                continue
            if isinstance(descriptor, dict):
                kind = str(descriptor.get("type", descriptor.get("kind", "float"))).lower()
            else:
                kind = str(descriptor).strip().lower()
            if kind in {"", "float", "real", "continuous", "double"}:
                continue
            if kind in {"int", "integer", "discrete", "ordinal"}:
                pos[i] = np.round(pos[i])
            elif kind in {"binary", "bool", "boolean"}:
                threshold = 0.5 if lo[i] <= 0.0 <= hi[i] and lo[i] <= 1.0 <= hi[i] else (lo[i] + hi[i]) / 2.0
                pos[i] = 1.0 if pos[i] >= threshold else 0.0
            elif kind.startswith(("step:", "multiple:", "quantum:")) or (isinstance(descriptor, dict) and kind in {"step", "multiple", "quantum"}):
                step, mode = self._parse_step_descriptor(descriptor)
                if step is None or step <= 0:
                    raise ValueError(f"Invalid step variable descriptor at index {i}: {descriptor!r}")
                scaled = pos[i] / step
                if mode in {"ceil", "ceiling", "up"}:
                    pos[i] = np.ceil(scaled) * step
                elif mode in {"floor", "down"}:
                    pos[i] = np.floor(scaled) * step
                else:
                    pos[i] = np.round(scaled) * step
            else:
                raise ValueError(f"Unsupported variable type at index {i}: {descriptor!r}")
        return np.clip(pos, lo, hi)

    def clip_position(self, position) -> np.ndarray:
        return self.apply_variable_types(position)

    def _assign_back(self, original, new_value: np.ndarray) -> None:
        try:
            if isinstance(original, np.ndarray):
                original[...] = new_value
            elif isinstance(original, list):
                original[:] = new_value.tolist()
        except Exception:
            pass

    def _repair_position(self, position) -> np.ndarray:
        raw = np.asarray(position, dtype=float).copy()
        repaired = self.apply_variable_types(raw)
        repair_fn = self.metadata.get("repair_function")
        use_raw = bool(self.metadata.get("use_raw_repair_input", False))
        if callable(repair_fn):
            seed_input = raw if use_raw else repaired
            try:
                candidate = repair_fn(seed_input.copy(), self.min_values, self.max_values)
            except TypeError:
                candidate = repair_fn(seed_input.copy())
            if candidate is not None:
                repaired = self.apply_variable_types(candidate)
        return repaired

    def _resample_position(self) -> np.ndarray:
        lo = np.asarray(self.min_values, dtype=float)
        hi = np.asarray(self.max_values, dtype=float)
        return self.apply_variable_types(np.random.uniform(lo, hi, self.dimension))

    def _normalize_constraint_value(self, value) -> float:
        tol = self._equality_tolerance()
        if isinstance(value, dict):
            if "type" in value and "value" in value:
                kind = str(value["type"]).lower()
                val = value["value"]
                if kind in {"eq", "equality"}:
                    return max(0.0, abs(float(val)) - tol)
                return max(0.0, float(val))
            total = 0.0
            for key in ("ineq", "ineqs", "g"):
                if key in value:
                    arr = np.atleast_1d(value[key]).astype(float)
                    total += float(np.maximum(arr, 0.0).sum())
            for key in ("eq", "eqs", "h"):
                if key in value:
                    arr = np.atleast_1d(value[key]).astype(float)
                    total += float(np.maximum(np.abs(arr) - tol, 0.0).sum())
            if total > 0.0:
                return total
            if "feasible" in value:
                return 0.0 if bool(value["feasible"]) else 1.0
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            kind, val = value
            kind = kind.lower()
            if kind in {"eq", "equality"}:
                return max(0.0, abs(float(val)) - tol)
            if kind in {"ineq", "inequality", "g"}:
                return max(0.0, float(val))
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.dtype == np.bool_ or all(isinstance(v, (bool, np.bool_)) for v in arr.tolist()):
                return 0.0 if bool(np.all(arr)) else 1.0
            arr = arr.astype(float)
            return float(np.maximum(arr, 0.0).sum())
        if isinstance(value, (bool, np.bool_)):
            return 0.0 if bool(value) else 1.0
        return max(0.0, float(value))

    def violation(self, position) -> float:
        if not self.constraints:
            return 0.0
        pos = np.asarray(position, dtype=float)
        total = 0.0
        for constraint in self.constraints:
            value = constraint(pos.tolist())
            total += self._normalize_constraint_value(value)
        return float(total)

    def is_feasible_position(self, position) -> bool:
        return self.violation(position) <= self._equality_tolerance()

    def score_from_raw(self, raw_fitness: float, violation: float) -> float:
        if not self.has_constraints or violation <= self._equality_tolerance():
            return float(raw_fitness)
        handler = self._handler()
        if handler in {"none", "repair", "reject_resample"}:
            handler = "penalty"
        if handler in {"deb", "feasibility", "feasibility_first", "feasible_first"}:
            big = float(self.metadata.get("feasibility_big_m", 1e12))
            if self.objective == "min":
                return big + violation
            return -big - violation
        coeff = self._penalty_coefficient()
        if self.objective == "min":
            return float(raw_fitness) + coeff * violation
        return float(raw_fitness) - coeff * violation

    def _evaluation_cache_key(self, position) -> tuple[tuple[int, ...], bytes]:
        pos = self.apply_variable_types(position)
        arr = np.ascontiguousarray(pos, dtype=np.float64)
        return tuple(arr.shape), arr.tobytes()

    def _cache_evaluation_details(self, details: dict[str, Any]) -> None:
        """Cache already-computed details without evaluating the objective again.

        BaseEngine uses this cache for history and final metadata. This removes
        the former post-step/post-run objective reevaluations that were invisible
        to engine-local counters and could exceed ``max_evaluations``.
        """
        try:
            cache = self.metadata.get("_evaluation_details_cache")
            if not isinstance(cache, OrderedDict):
                cache = OrderedDict()
                self.metadata["_evaluation_details_cache"] = cache
            key = self._evaluation_cache_key(details["position"])
            cached = {
                "position": np.asarray(details["position"], dtype=float).copy(),
                "raw_fitness": float(details["raw_fitness"]),
                "fitness": float(details["fitness"]),
                "violation": float(details["violation"]),
                "is_feasible": bool(details["is_feasible"]),
                "handler": details.get("handler"),
            }
            cache[key] = cached
            cache.move_to_end(key)

            # Keep a constant-memory ledger of successful evaluations.  This is
            # the authoritative source for the default ``return_best`` policy
            # when a strict budget interrupts initialization or a macro-step
            # before the engine can return a complete native state.
            self.metadata["_successful_evaluation_count"] = int(
                self.metadata.get("_successful_evaluation_count", 0)
            ) + 1
            self.metadata["_last_successful_evaluation_details"] = dict(cached)
            incumbent = self.metadata.get("_best_successful_evaluation_details")
            if not isinstance(incumbent, dict) or self.is_better(
                float(cached["fitness"]), float(incumbent["fitness"])
            ):
                self.metadata["_best_successful_evaluation_details"] = dict(cached)

            limit = max(16, int(self.metadata.get("evaluation_cache_size", 8192)))
            while len(cache) > limit:
                cache.popitem(last=False)
        except Exception:
            # Caching is observational and must never affect optimization.
            pass

    def cached_evaluation_details(self, position) -> dict[str, Any] | None:
        """Return cached details for an already-evaluated position, if present."""
        try:
            cache = self.metadata.get("_evaluation_details_cache")
            if not isinstance(cache, OrderedDict):
                return None
            key = self._evaluation_cache_key(position)
            details = cache.get(key)
            if details is None:
                return None
            cache.move_to_end(key)
            return {
                "position": np.asarray(details["position"], dtype=float).copy(),
                "raw_fitness": float(details["raw_fitness"]),
                "fitness": float(details["fitness"]),
                "violation": float(details["violation"]),
                "is_feasible": bool(details["is_feasible"]),
                "handler": details.get("handler"),
            }
        except Exception:
            return None

    def best_successful_evaluation_details(self) -> dict[str, Any] | None:
        """Return the best genuinely evaluated, constraint-aware candidate."""
        details = self.metadata.get("_best_successful_evaluation_details")
        if not isinstance(details, dict):
            return None
        try:
            return {
                "position": np.asarray(details["position"], dtype=float).copy(),
                "raw_fitness": float(details["raw_fitness"]),
                "fitness": float(details["fitness"]),
                "violation": float(details["violation"]),
                "is_feasible": bool(details["is_feasible"]),
                "handler": details.get("handler"),
            }
        except Exception:
            return None

    def successful_evaluation_count(self) -> int:
        """Return successful evaluations observed through ``ProblemSpec``."""
        return int(self.metadata.get("_successful_evaluation_count", 0))

    def evaluate_details(self, position, apply_handler: bool = True) -> dict[str, Any]:
        # Variable-domain projection is always active, not only for constrained
        # problems.  This ensures integer, binary, and stepped variables are
        # respected before the objective and constraints see the candidate.
        pos = self.apply_variable_types(position)
        handler = self._handler()
        if self.has_constraints and apply_handler:
            if handler == "repair":
                pos = self._repair_position(pos)
            elif handler in {"reject_resample", "resample", "reject"}:
                if not self.is_feasible_position(pos):
                    attempts = max(1, self._resample_attempts())
                    found = False
                    for _ in range(attempts):
                        trial = self._resample_position()
                        if self.is_feasible_position(trial):
                            pos = trial
                            found = True
                            break
                    if not found:
                        pos = self._repair_position(pos)
        violation = self.violation(pos)
        raw_fitness = float(self.target_function(pos.tolist()))
        effective = self.score_from_raw(raw_fitness, violation)
        feasible = violation <= self._equality_tolerance()
        details = {
            "position": pos,
            "raw_fitness": raw_fitness,
            "fitness": effective,
            "violation": float(violation),
            "is_feasible": bool(feasible),
            "handler": handler,
        }
        self._cache_evaluation_details(details)
        return details

    def evaluate(self, position) -> float:
        details = self.evaluate_details(position, apply_handler=True)
        self._assign_back(position, details["position"])
        # Passive EvoMapX probe.  This records already-computed objective
        # evaluations and never changes the returned value, repaired position,
        # random state, or evaluation budget.
        try:
            probe = (self.metadata or {}).get("_evomapx_probe")
            if probe is not None and getattr(probe, "enabled", False):
                probe.record_evaluation(
                    details.get("position"),
                    float(details["fitness"]),
                    raw_fitness=float(details.get("raw_fitness", details["fitness"])),
                    details=details,
                )
        except Exception:
            # Observability must never break optimization.
            pass
        return float(details["fitness"])


# ---------------------------------------------------------------------------
# 2. EngineConfig
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """
    Universal stopping / logging / seeding controls.
    Algorithm-specific knobs go inside *params*.
    """

    max_steps:               int   | None = None
    max_evaluations:         int   | None = None
    target_fitness:          float | None = None
    seed:                    int   | None = None
    verbose:                 bool        = False
    store_history:           bool        = True
    store_population_snapshots: bool     = False
    snapshot_interval:       int         = 1
    timeout_seconds:         float | None = None
    params:                  dict[str, Any] = field(default_factory=dict)
    callbacks:               Any = None
    init_function:           Callable | None = None
    budget_exhaustion_policy: str = "return_best"


# ---------------------------------------------------------------------------
# 3. CapabilityProfile
# ---------------------------------------------------------------------------

@dataclass
class CapabilityProfile:
    """
    Declares what an engine supports.

    Notes
    -----
    * supports_native_constraints
        Algorithm-specific constraint machinery implemented directly by the
        engine.
    * supports_framework_constraints
        Shared constraint handling available through ProblemSpec when
        constraint functions are supplied.
    * supports_constraints
        Effective constraint support for the current engine instance.
        This value may be activated at runtime when a constrained problem is
        created.
    """

    has_population:                 bool = False
    has_archive:                    bool = False
    supports_candidate_injection:   bool | None = None
    supports_restart:               bool | None = None
    supports_checkpoint:            bool = True
    supports_native_constraints:    bool = False
    supports_framework_constraints: bool = True
    supports_constraints:           bool = False
    supports_discrete:              bool = False
    supports_integer:               bool = False
    supports_mixed:                 bool = False
    supports_async_messages:        bool = False
    supports_diversity_metrics:     bool = False
    supports_snapshot_fit:          bool | None = None


# ---------------------------------------------------------------------------
# 4. CandidateRecord  — universal interoperability unit
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """
    The smallest unit exchanged between engines (hybrids, archipelagos, etc.).

    *role* tags the semantic purpose of this candidate so receiving engines
    can apply the right injection policy:
        "best"    – best found so far
        "elite"   – member of elite set
        "current" – current search position
        "diverse" – selected for diversity
        "restart" – seed for a fresh restart
    """

    position:         list[float]
    fitness:          float
    source_algorithm: str | None        = None
    source_step:      int | None        = None
    role:             str               = "best"
    metadata:         dict[str, Any]   = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 5. EngineState  — serialisable native state
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    """
    Carries ALL state needed to pause, resume, or checkpoint a run.

    Algorithm-specific objects (population matrix, velocities, temperature,
    harmony memory, …) live in *payload*. The base fields cover the universal
    contract.
    """

    step:               int             = 0
    evaluations:        int             = 0
    best_position:      list[float] | None = None
    best_fitness:       float | None    = None
    initialized:        bool            = False
    terminated:         bool            = False
    termination_reason: str | None      = None
    start_time:         float | None    = None
    elapsed_time:       float           = 0.0
    rng_state:          Any             = None
    payload:            dict[str, Any]  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 6. OptimizationResult
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Standardised result returned by *finalize()*."""

    algorithm_id:        str
    best_position:       list[float]
    best_fitness:        float
    steps:               int
    evaluations:         int
    termination_reason:  str | None
    history:             list[dict[str, Any]] = field(default_factory=list)
    population_snapshots: list[dict[str, Any]] = field(default_factory=list)
    capabilities:        CapabilityProfile | None = None
    metadata:            dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id":       self.algorithm_id,
            "best_position":      list(self.best_position),
            "best_fitness":       float(self.best_fitness),
            "steps":              self.steps,
            "evaluations":        self.evaluations,
            "termination_reason": self.termination_reason,
            "history":            self.history,
            "population_snapshots": self.population_snapshots,
            "capabilities":       self.capabilities,
            "metadata":           self.metadata,
        }

    def evomapx_analysis(self, **kwargs):
        """Build an EvoMapX report for this single-algorithm result.

        This mirrors the package-level ``pymetaheuristic.evomapx_analysis``
        function, but keeps the public README/Colab style working:
        ``result.evomapx_analysis(level="operator")``.
        """
        from ..evomapx import evomapx_analysis
        return evomapx_analysis(self, **kwargs)

    def explain_evomapx(self, **kwargs) -> str:
        """Return a textual EvoMapX explanation for this result."""
        from ..evomapx import evomapx_analysis, explain_evomapx
        return explain_evomapx(evomapx_analysis(self, **kwargs))

    def plot_evomapx_attribution(self, **kwargs):
        """Plot the EvoMapX attribution/OAM heatmap for this result."""
        from ..evomapx import plot_attribution_heatmap
        return plot_attribution_heatmap(self, **kwargs)

    def plot_evomapx_cds(self, **kwargs):
        """Plot the EvoMapX Convergence Driver Score bars for this result."""
        from ..evomapx import plot_cds_bar
        return plot_cds_bar(self, **kwargs)

    def plot_evomapx_cds_time_series(self, **kwargs):
        """Plot the EvoMapX CDS time series for this result."""
        from ..evomapx import plot_cds_time_series
        return plot_cds_time_series(self, **kwargs)

    def plot_evomapx_peg(self, **kwargs):
        """Plot the EvoMapX population evolution graph for this result."""
        from ..evomapx import plot_population_evolution_graph
        return plot_population_evolution_graph(self, **kwargs)

    def export_evomapx_json(self, filepath, **kwargs) -> str:
        """Export this result's EvoMapX report as JSON."""
        from ..evomapx import export_evomapx_json
        return export_evomapx_json(self, filepath, **kwargs)

    def export_evomapx_csv(self, filepath, **kwargs) -> str:
        """Export this result's EvoMapX attribution table as CSV."""
        from ..evomapx import export_evomapx_csv
        return export_evomapx_csv(self, filepath, **kwargs)


# ---------------------------------------------------------------------------
# 7. BaseEngine  — the mandatory protocol every algorithm must satisfy
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    """
    Abstract base for every algorithm engine in pyMetaheuristic src.

    Subclasses must implement:
        initialize()         → EngineState
        step(state)          → EngineState   (one native macro-step)
        observe(state)       → dict           (telemetry snapshot)
        get_best_candidate() → CandidateRecord
        finalize(state)      → OptimizationResult

    They may override the default (no-op) implementations of:
        export_candidates()
        inject_candidates()
        get_population()
        export_state() / import_state()
    """

    # Subclasses must set these as class attributes
    algorithm_id:   str
    algorithm_name: str
    family:         str
    capabilities:   CapabilityProfile

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        self.problem = problem
        self.config  = config
        self.capabilities = replace(self.capabilities)
        self.capabilities.supports_constraints = bool(
            self.capabilities.supports_native_constraints
            or (
                problem.has_constraints
                and self.capabilities.supports_framework_constraints
            )
        )
        from ..callbacks import CallbackList
        self._callbacks = CallbackList.from_any(getattr(config, "callbacks", None))
        self._callbacks.set_engine(self)
        self._stop_requested_reason: str | None = None
        from ..evomapx_probe import EvoMapXProbe
        self._evomapx_probe = EvoMapXProbe.from_engine(self)
        try:
            if getattr(self._evomapx_probe, "enabled", False):
                self.problem.metadata["_evomapx_probe"] = self._evomapx_probe
            elif self.problem.metadata.get("_evomapx_probe") is self._evomapx_probe:
                self.problem.metadata.pop("_evomapx_probe", None)
        except Exception:
            pass
        self._wrap_initialize_for_custom_init()

    @classmethod
    def info(cls) -> str:
        defaults = getattr(cls, "_DEFAULTS", {}) or {}
        defaults_text = ", ".join(f"{k}={v}" for k, v in defaults.items()) if defaults else "none"
        doc = (cls.__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else cls.algorithm_name
        reference = getattr(cls, "_REFERENCE", {}) or {}
        lines = [
            f"Algorithm: {cls.algorithm_name} [{cls.algorithm_id}]",
            f"Family: {cls.family}",
            f"Summary: {summary}",
            f"Defaults: {defaults_text}",
        ]
        doi = reference.get("doi")
        if doi:
            lines.append(f"DOI: {doi}")
        else:
            lines.append("DOI: not available in the curated source table for this release.")
        return "\n".join(lines)

    def request_stop(self, reason: str = "callback_stop") -> None:
        self._stop_requested_reason = str(reason)

    def clear_stop_request(self) -> None:
        self._stop_requested_reason = None

    def _strict_evaluation_budget(self) -> SharedEvaluationBudget | None:
        target = getattr(self.problem, "target_function", None)
        if isinstance(target, SharedEvaluationBudget):
            return target
        budget = (self.problem.metadata or {}).get("_evaluation_budget")
        return budget if isinstance(budget, SharedEvaluationBudget) else None

    def _strict_evaluation_budget_label(self) -> str | None:
        label = (self.problem.metadata or {}).get("_evaluation_budget_label")
        return None if label is None else str(label)

    def _sync_evaluation_count(self, state: EngineState, label: str | None = None) -> int:
        """Synchronize an engine state with the authoritative objective counter.

        For ordinary runs, ``_evaluation_budget_label`` is the algorithm ID.
        Collaborative runners pass the island label explicitly so local state
        telemetry remains per-island while the wrapper also maintains a global
        total.
        """
        budget = self._strict_evaluation_budget()
        if budget is None:
            return int(state.evaluations)
        resolved_label = label if label is not None else self._strict_evaluation_budget_label()
        if resolved_label is None:
            # A shared collaborative budget has no engine-local label outside
            # the runner's context. Do not replace a local counter by the global
            # total accidentally.
            return int(state.evaluations)
        state.evaluations = budget.used_for(resolved_label)
        return int(state.evaluations)

    def _remember_best_evaluation_details(self, state: EngineState) -> dict[str, Any] | None:
        if state.best_position is None or state.best_fitness is None:
            return None
        details = self.problem.cached_evaluation_details(state.best_position)
        if details is None and not self.problem.has_constraints:
            details = {
                "position": np.asarray(state.best_position, dtype=float).copy(),
                "raw_fitness": float(state.best_fitness),
                "fitness": float(state.best_fitness),
                "violation": 0.0,
                "is_feasible": True,
                "handler": self.problem.constraint_handler or "none",
            }
        if details is not None:
            state.payload["_best_evaluation_details"] = details
        return details

    def _best_evaluation_details(self, state: EngineState) -> dict[str, Any] | None:
        details = state.payload.get("_best_evaluation_details")
        if isinstance(details, dict):
            return details
        return self._remember_best_evaluation_details(state)

    def _budget_exhaustion_policy(self) -> str:
        policy = str(
            getattr(self.config, "budget_exhaustion_policy", "return_best")
            or "return_best"
        ).strip().lower()
        aliases = {
            "return": "return_best",
            "best": "return_best",
            "graceful": "return_best",
            "error": "raise",
        }
        policy = aliases.get(policy, policy)
        if policy not in {"return_best", "raise"}:
            raise ValueError(
                "budget_exhaustion_policy must be 'return_best' or 'raise'."
            )
        return policy

    def _best_available_evaluation_details(
        self,
        state: EngineState | None = None,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the best candidate that was genuinely evaluated.

        Preference order is constraint-aware ProblemSpec telemetry, the last
        complete engine state, then the strict wrapper's raw-objective fallback.
        No objective function is called here.
        """
        candidates: list[dict[str, Any]] = []

        details = self.problem.best_successful_evaluation_details()
        if isinstance(details, dict):
            candidates.append(details)

        if state is not None and state.best_position is not None and state.best_fitness is not None:
            state_details = self.problem.cached_evaluation_details(state.best_position)
            if state_details is None:
                state_details = {
                    "position": np.asarray(state.best_position, dtype=float).copy(),
                    "raw_fitness": float(state.best_fitness),
                    "fitness": float(state.best_fitness),
                    "violation": 0.0,
                    "is_feasible": True,
                    "handler": self.problem.constraint_handler or "none",
                }
            candidates.append(state_details)

        budget = self._strict_evaluation_budget()
        if budget is not None and (not self.problem.has_constraints or not candidates):
            raw = budget.best_raw_for(label if label is not None else self._strict_evaluation_budget_label())
            if isinstance(raw, dict):
                try:
                    raw["position"] = self.problem.apply_variable_types(raw["position"])
                    candidates.append(raw)
                except Exception:
                    pass

        if not candidates:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if self.problem.is_better(float(candidate["fitness"]), float(best["fitness"])):
                best = candidate
        return {
            "position": np.asarray(best["position"], dtype=float).copy(),
            "raw_fitness": float(best.get("raw_fitness", best["fitness"])),
            "fitness": float(best["fitness"]),
            "violation": float(best.get("violation", 0.0)),
            "is_feasible": bool(best.get("is_feasible", True)),
            "handler": best.get("handler", self.problem.constraint_handler or "none"),
        }

    def _prepare_budget_exhausted_state(
        self,
        state: EngineState | None,
        *,
        label: str | None,
        completed_steps: int,
        phase: str,
        phase_evaluations: int | None = None,
    ) -> EngineState:
        """Create a safe terminal state from successful evaluations only."""
        if state is None:
            state = EngineState()
        budget = self._strict_evaluation_budget()
        if budget is not None:
            state.evaluations = int(budget.used_for(label))
        state.step = max(0, int(completed_steps))
        state.terminated = True
        state.termination_reason = "max_evaluations"
        state.initialized = phase != "initialization"
        details = self._best_available_evaluation_details(state, label)
        if details is None:
            raise RuntimeError(
                "The evaluation budget was exhausted before any successful "
                "objective value was available for return_best finalization."
            )
        state.best_position = np.asarray(details["position"], dtype=float).tolist()
        state.best_fitness = float(details["fitness"])
        state.payload["_best_evaluation_details"] = details
        state.payload["_budget_exhausted_phase"] = str(phase)
        state.payload["_budget_exhausted_partial_state"] = True
        if phase_evaluations is not None:
            state.payload["_budget_exhausted_phase_evaluations"] = max(0, int(phase_evaluations))
        return state

    def _make_budget_exhausted_result(
        self,
        state: EngineState,
        *,
        label: str | None,
        phase: str,
        history: list[dict[str, Any]] | None = None,
        snapshots: list[dict[str, Any]] | None = None,
        improvement_history: list[dict[str, Any]] | None = None,
    ) -> OptimizationResult:
        """Build a normal result without touching incomplete native payloads."""
        details = self._best_available_evaluation_details(state, label)
        if details is None:
            raise RuntimeError("No successful evaluation is available to return.")

        position = self.problem.apply_variable_types(details["position"]).astype(float).tolist()
        budget = self._strict_evaluation_budget()
        evaluations = int(state.evaluations)
        successful = self.problem.successful_evaluation_count()
        if budget is not None:
            evaluations = int(budget.used_for(label))
            successful = max(successful, int(budget.successful_for(label)))

        metadata: dict[str, Any] = {
            "algorithm_name": self.algorithm_name,
            "family": self.family,
            "constraint_handler": self.problem.constraint_handler or "none",
            "best_raw_fitness": float(details["raw_fitness"]),
            "best_violation": float(details["violation"]),
            "best_is_feasible": bool(details["is_feasible"]),
            "actual_evaluations": evaluations,
            "successful_evaluations": int(successful),
            "evaluation_budget_enforcement": "hard",
            "budget_exhaustion_policy": "return_best",
            "budget_exhaustion_phase": str(phase),
            "partial_initialization": phase == "initialization",
            "partial_step": phase == "step",
            "partial_native_state_committed": False,
            "best_partial_candidate_retained": True,
            "improvement_history": list(improvement_history or []),
            "n_improvements": len(improvement_history or []),
        }
        if budget is not None and budget.max_evaluations is not None:
            metadata["max_evaluations"] = int(budget.max_evaluations)
        phase_evaluations = state.payload.get("_budget_exhausted_phase_evaluations")
        if phase_evaluations is not None:
            metadata["partial_phase_evaluations"] = int(phase_evaluations)

        if phase == "initialization" and self.capabilities.has_population:
            requested = int(self._infer_population_size())
            metadata["requested_population_size"] = requested
            metadata["evaluated_population_size"] = int(successful)
            if 0 <= successful <= requested:
                metadata["unevaluated_candidates_discarded"] = int(requested - successful)

        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=position,
            best_fitness=float(details["fitness"]),
            steps=int(state.step),
            evaluations=evaluations,
            termination_reason="max_evaluations",
            history=list(history or []),
            population_snapshots=list(snapshots or []),
            capabilities=replace(self.capabilities),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> EngineState:
        """
        Build all native structures, evaluate initial candidate(s),
        set best_position / best_fitness, and return a fresh EngineState
        with initialized=True and step=0.
        Must NOT run the main loop.
        """

    @abstractmethod
    def step(self, state: EngineState) -> EngineState:
        """
        Perform one native macro-iteration.
        Must increment state.step and state.evaluations correctly.
        All algorithm-specific data lives in state.payload.
        """

    @abstractmethod
    def observe(self, state: EngineState) -> dict:
        """
        Return a telemetry snapshot.  Minimum required keys:
            step, evaluations, best_fitness
        Add algorithm-native keys (temperature, diversity, …) freely.
        """

    @abstractmethod
    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        """Always returns the current best candidate."""

    @abstractmethod
    def finalize(self, state: EngineState) -> OptimizationResult:
        """Assemble and return the standardised OptimizationResult."""

    def _infer_population_size(self) -> int:
        for attr in ("_n", "_swarm_size", "_food", "_solutions", "_size", "_pop_size"):
            value = getattr(self, attr, None)
            if value is not None:
                try:
                    return max(1, int(value))
                except Exception:
                    pass
        for key in ("population_size", "swarm_size", "solutions", "size"):
            if key in getattr(self.config, "params", {}):
                try:
                    return max(1, int(self.config.params[key]))
                except Exception:
                    pass
        return 1

    def _wrap_initialize_for_custom_init(self) -> None:
        init_function = getattr(self.config, "init_function", None)
        if not callable(init_function):
            return
        original_initialize = self.initialize
        engine = self

        def wrapped_initialize():
            pop_size = engine._infer_population_size()
            dim = int(engine.problem.dimension)
            cache: dict[str, Any] = {"built": False, "used": False, "positions": None}
            original_uniform = np.random.uniform

            def _build_positions() -> np.ndarray:
                if cache["built"]:
                    return cache["positions"]
                rng = np.random.default_rng(engine.config.seed)
                try:
                    data = init_function(problem=engine.problem, pop_size=pop_size, rng=rng, engine=engine)
                except TypeError:
                    try:
                        data = init_function(engine.problem, pop_size, rng)
                    except TypeError:
                        data = init_function(engine.problem, pop_size)
                if isinstance(data, tuple):
                    data = data[0]
                arr = np.asarray(data, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape != (pop_size, dim):
                    raise ValueError(
                        f"init_function must return shape {(pop_size, dim)}, received {arr.shape}"
                    )
                lo = np.asarray(engine.problem.min_values, dtype=float)
                hi = np.asarray(engine.problem.max_values, dtype=float)
                arr = np.vstack([engine.problem.apply_variable_types(row) for row in np.clip(arr, lo, hi)])
                cache["positions"] = arr
                cache["built"] = True
                return arr

            def hooked_uniform(low=0.0, high=1.0, size=None):
                if cache["used"]:
                    return original_uniform(low, high, size)
                try:
                    if size is None:
                        shape = np.broadcast(np.asarray(low), np.asarray(high)).shape
                    else:
                        shape = tuple(int(s) for s in np.atleast_1d(size).tolist())
                except Exception:
                    shape = None
                target_shapes = {(pop_size, dim)}
                if pop_size == 1:
                    target_shapes.update({(dim,), (1, dim)})
                if shape in target_shapes:
                    positions = _build_positions()
                    cache["used"] = True
                    if shape == (dim,):
                        return positions[0].copy()
                    if shape == (1, dim):
                        return positions[:1].copy()
                    return positions.copy()
                return original_uniform(low, high, size)

            np.random.uniform = hooked_uniform
            try:
                return original_initialize()
            finally:
                np.random.uniform = original_uniform

        self.initialize = wrapped_initialize

    def _callback_payload(self, state: EngineState) -> tuple[Any, Any]:
        pop_key = self._population_payload_key(state)
        population = None
        fitness = None
        if pop_key is not None:
            pop = state.payload.get(pop_key)
            if hasattr(pop, "shape") and len(getattr(pop, "shape", [])) == 2 and pop.shape[1] >= self.problem.dimension + 1:
                population = np.asarray(pop[:, :-1], dtype=float).copy()
                fitness = np.asarray(pop[:, -1], dtype=float).copy()
        return population, fitness

    def _run_callbacks(self, hook: str, state: EngineState, observation: dict[str, Any] | None = None) -> None:
        if self._callbacks is None:
            return
        population, fitness = self._callback_payload(state)
        fn = getattr(self._callbacks, hook, None)
        if callable(fn):
            fn(
                population,
                fitness,
                None if state.best_position is None else list(state.best_position),
                state.best_fitness,
                state=state,
                observation=dict(observation or {}),
                engine=self,
            ) if hook in {"before_iteration", "after_iteration"} else fn(state=state, observation=dict(observation or {}), engine=self)

    # ------------------------------------------------------------------
    # Stop condition (shared, centralised)
    # ------------------------------------------------------------------

    def should_stop(self, state: EngineState) -> bool:
        self._sync_evaluation_count(state)
        if self._stop_requested_reason is not None:
            state.termination_reason = self._stop_requested_reason
            state.terminated = True
            return True
        if state.terminated:
            return True
        cfg = self.config
        if cfg.max_steps is not None and state.step >= cfg.max_steps:
            state.termination_reason = "max_steps"
            return True
        if cfg.max_evaluations is not None and state.evaluations >= cfg.max_evaluations:
            state.termination_reason = "max_evaluations"
            return True
        if cfg.target_fitness is not None and state.best_fitness is not None:
            if self.problem.is_better(state.best_fitness, cfg.target_fitness) or \
               state.best_fitness == cfg.target_fitness:
                state.termination_reason = "target_fitness"
                return True
        if cfg.timeout_seconds is not None and state.start_time is not None:
            if time.perf_counter() - state.start_time >= cfg.timeout_seconds:
                state.termination_reason = "timeout"
                return True
        return False

    # ------------------------------------------------------------------
    # Convenience run loop (engines may override for finer control)
    # ------------------------------------------------------------------

    def run(self) -> OptimizationResult:
        _termination = getattr(self.config, "_termination_obj", None)
        if _termination is not None:
            _termination.start(step=0, evaluations=0)

        self.clear_stop_request()
        self._callbacks.before_run(engine=self)
        self._evomapx_probe.begin_run()

        budget = self._strict_evaluation_budget()
        budget_label = self._strict_evaluation_budget_label() or self.algorithm_id
        budget_policy = self._budget_exhaustion_policy()
        run_started = time.perf_counter()
        try:
            if budget is not None:
                with budget.use_label(budget_label):
                    state = self.initialize()
            else:
                state = self.initialize()
        except EvaluationBudgetExceeded as exc:
            if budget_policy == "raise":
                raise ValueError(
                    f"max_evaluations={self.config.max_evaluations} is too small to "
                    f"complete initialization of '{self.algorithm_id}' under a strict "
                    "function-evaluation budget."
                ) from exc

            used = 0 if budget is None else int(budget.used_for(budget_label))
            state = self._prepare_budget_exhausted_state(
                None,
                label=budget_label,
                completed_steps=0,
                phase="initialization",
                phase_evaluations=used,
            )
            state.start_time = run_started
            state.elapsed_time = time.perf_counter() - run_started
            result = self._make_budget_exhausted_result(
                state,
                label=budget_label,
                phase="initialization",
                history=[],
                snapshots=[],
                improvement_history=[{
                    "step": 0,
                    "evaluations": int(state.evaluations),
                    "best_fitness": float(state.best_fitness),
                }],
            )
            result.metadata["elapsed_time"] = float(state.elapsed_time)
            self._evomapx_probe.finalize_result(result)
            try:
                self._callbacks.after_run(state=state, engine=self, result=result)
            finally:
                self._sync_evaluation_count(state, budget_label if budget is not None else None)
                result.evaluations = int(state.evaluations)
                result.metadata["actual_evaluations"] = int(state.evaluations)
            return result

        self._sync_evaluation_count(state, budget_label if budget is not None else None)
        self._remember_best_evaluation_details(state)
        self._evomapx_probe.after_initialize(state)
        state.start_time = time.perf_counter()
        history: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        improvement_history: list[dict[str, Any]] = []

        if state.best_fitness is not None:
            improvement_history.append({
                "step": int(state.step),
                "evaluations": int(state.evaluations),
                "best_fitness": float(state.best_fitness),
            })

        _prev_best: float | None = state.best_fitness
        budget_exhausted_phase: str | None = None

        while not self.should_stop(state):
            if _termination is not None:
                _termination.update(state.best_fitness)
                if _termination.should_stop(
                    step=state.step,
                    evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    objective=self.problem.objective,
                ):
                    state.termination_reason = _termination.triggered_reason
                    break

            self._run_callbacks("before_iteration", state)
            if self.should_stop(state):
                break

            self._evomapx_probe.before_step(state)
            completed_steps_before = int(state.step)
            evaluations_before = int(state.evaluations)
            try:
                if budget is not None:
                    with budget.use_label(budget_label):
                        state = self.step(state)
                else:
                    state = self.step(state)
            except EvaluationBudgetExceeded:
                if budget_policy == "raise":
                    raise
                # The native step is incomplete.  Keep every genuinely
                # evaluated candidate in the best-evaluation ledger, but do not
                # commit or finalize the partially mutated native payload.
                self._sync_evaluation_count(state, budget_label)
                phase_evaluations = max(0, int(state.evaluations) - evaluations_before)
                state = self._prepare_budget_exhausted_state(
                    state,
                    label=budget_label,
                    completed_steps=completed_steps_before,
                    phase="step",
                    phase_evaluations=phase_evaluations,
                )
                state.elapsed_time = time.perf_counter() - state.start_time
                budget_exhausted_phase = "step"
                try:
                    self._evomapx_probe.cancel_step()
                except Exception:
                    pass
                break

            self._sync_evaluation_count(state, budget_label if budget is not None else None)
            self._remember_best_evaluation_details(state)
            state.elapsed_time = time.perf_counter() - state.start_time
            obs = dict(self.observe(state))
            # Engine-local observations may contain a manually maintained count;
            # overwrite it with the authoritative actual-call counter.
            obs["evaluations"] = int(state.evaluations)
            obs = self._evomapx_probe.after_step(state, obs)

            if "diversity" not in obs and self.capabilities.has_population:
                obs["diversity"] = self._compute_diversity(state)
            obs.setdefault("global_best_fitness", state.best_fitness)

            improved = False
            if _prev_best is not None and state.best_fitness is not None:
                improved = self.problem.is_better(state.best_fitness, _prev_best)
                obs["exploitation"] = 1.0 if improved else 0.0
                obs["exploration"] = 0.0 if improved else 1.0
            elif state.best_fitness is not None:
                improved = True
                obs["exploitation"] = 1.0
                obs["exploration"] = 0.0
            obs["improved"] = bool(improved)

            if improved and state.best_fitness is not None:
                improvement_history.append({
                    "step": int(state.step),
                    "evaluations": int(state.evaluations),
                    "best_fitness": float(state.best_fitness),
                })
            if state.best_fitness is not None:
                _prev_best = state.best_fitness

            if self.problem.has_constraints and state.best_position is not None:
                best_details = self._best_evaluation_details(state)
                if best_details is not None:
                    obs["best_raw_fitness"] = best_details["raw_fitness"]
                    obs["best_violation"] = best_details["violation"]
                    obs["best_is_feasible"] = best_details["is_feasible"]

            obs["elapsed_time"] = state.elapsed_time
            if self.config.store_history:
                history.append(obs)

            # Store the post-step snapshot before checking terminal conditions so
            # the final population is not silently lost when max_steps is hit.
            if self.config.store_population_snapshots and self.capabilities.has_population and state.step % self.config.snapshot_interval == 0:
                try:
                    pop = self.get_population(state)
                    _snapshot_population = [{"position": c.position, "fitness": c.fitness} for c in pop]
                    _snapshot_population = self._evomapx_probe.enrich_population_snapshot(state, _snapshot_population)
                    snapshots.append({
                        "step": state.step,
                        "population": _snapshot_population,
                    })
                except NotImplementedError:
                    pass

            self._run_callbacks("after_iteration", state, observation=obs)
            if self.should_stop(state):
                break

            if self.config.verbose:
                print(
                    f"[{self.algorithm_id}] step={obs['step']:5d}  evals={obs['evaluations']:7d}  best={obs['best_fitness']:.6g}"
                )

        self._sync_evaluation_count(state, budget_label if budget is not None else None)
        self._remember_best_evaluation_details(state)
        if budget_exhausted_phase is not None:
            result = self._make_budget_exhausted_result(
                state,
                label=budget_label,
                phase=budget_exhausted_phase,
                history=history,
                snapshots=snapshots,
                improvement_history=improvement_history,
            )
        elif budget is not None:
            with budget.use_label(budget_label):
                result = self.finalize(state)
        else:
            result = self.finalize(state)

        # Ensure reported best positions obey declared variable domains even
        # for engines that internally use continuous latent vectors. The stored
        # best_fitness is unchanged because it came from an evaluated candidate.
        if result.best_position is not None:
            decoded_best = self.problem.apply_variable_types(result.best_position).astype(float).tolist()
            result.best_position = decoded_best
            state.best_position = decoded_best
        result.history = history
        result.population_snapshots = snapshots
        result.evaluations = int(state.evaluations)
        result.termination_reason = state.termination_reason or result.termination_reason

        best_details = self._best_evaluation_details(state)
        if best_details is not None:
            result.metadata.setdefault("constraint_handler", self.problem.constraint_handler or "none")
            result.metadata.setdefault("best_raw_fitness", best_details["raw_fitness"])
            result.metadata.setdefault("best_violation", best_details["violation"])
            result.metadata.setdefault("best_is_feasible", best_details["is_feasible"])

        if history:
            divs = [h["diversity"] for h in history if "diversity" in h and h["diversity"] is not None]
            result.metadata["mean_diversity"] = float(np.mean(divs)) if divs else None
            result.metadata["final_diversity"] = divs[-1] if divs else None
            exploits = [h["exploitation"] for h in history if "exploitation" in h]
            if exploits:
                result.metadata["exploitation_ratio"] = float(np.mean(exploits))
                result.metadata["exploration_ratio"] = 1.0 - result.metadata["exploitation_ratio"]
        result.metadata["improvement_history"] = improvement_history
        result.metadata["n_improvements"] = len(improvement_history)
        if budget is not None:
            result.metadata["max_evaluations"] = int(budget.max_evaluations)
            result.metadata["actual_evaluations"] = int(budget.used_for(budget_label))
            result.metadata["evaluation_budget_enforcement"] = "hard"

        self._evomapx_probe.finalize_result(result)
        try:
            self._callbacks.after_run(state=state, engine=self, result=result)
        finally:
            # A callback is allowed to inspect the engine, but if it calls the
            # objective the strict wrapper remains authoritative and the result
            # is synchronized to the actual number of calls.
            self._sync_evaluation_count(state, budget_label if budget is not None else None)
            result.evaluations = int(state.evaluations)
            if budget is not None:
                result.metadata["actual_evaluations"] = int(state.evaluations)
        return result

    # ------------------------------------------------------------------
    # Feature 2: Generic population diversity computation
    # ------------------------------------------------------------------

    def _compute_diversity(self, state: "EngineState") -> float | None:
        """
        Compute population diversity as the mean normalised distance from
        the centroid.  Falls back to None if no population payload exists.
        """
        pop_key = self._population_payload_key(state)
        if pop_key is None:
            return None
        pop = state.payload.get(pop_key)
        if pop is None or not hasattr(pop, "shape") or len(pop.shape) != 2:
            return None
        pos  = pop[:, :-1]
        lo   = np.array(self.problem.min_values, dtype=float)
        hi   = np.array(self.problem.max_values, dtype=float)
        denom = float(np.linalg.norm(hi - lo)) or 1.0
        centroid = pos.mean(axis=0)
        try:
            return float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Optional: candidate exchange (default: export best, reject inject)
    # ------------------------------------------------------------------

    def export_candidates(
        self,
        state: EngineState,
        k: int = 1,
        mode: str = "best",
    ) -> list[CandidateRecord]:
        """
        Default behaviour for population-based engines:
            - "best" / "elite": export the best *k* population members
            - "diverse": export a greedy farthest-point subset
        Engines without an exposed population fall back to the single best.
        """
        if self.capabilities.has_population:
            try:
                pop = self.get_population(state)
            except NotImplementedError:
                pop = None
            if pop:
                ranked = sorted(pop, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
                if mode == "diverse":
                    idx = self._diverse_candidate_indices(ranked, k)
                    chosen = [ranked[i] for i in idx]
                    role = "diverse"
                else:
                    chosen = ranked[:max(1, k)]
                    role = "elite" if mode in {"elite", "best"} else mode
                out = []
                for cand in chosen:
                    out.append(CandidateRecord(
                        position=list(cand.position),
                        fitness=float(cand.fitness),
                        source_algorithm=self.algorithm_id,
                        source_step=state.step,
                        role=role,
                        metadata=dict(getattr(cand, "metadata", {}) or {}),
                    ))
                return out
        return [self.get_best_candidate(state)]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        """
        Generic native injection for engines whose state is candidate-centric.
        It replaces the worst individuals in a population-like payload, reevaluates
        the incoming candidates, updates the best-so-far, and then calls the
        optional post-repair hook so algorithms can rebuild auxiliary state.
        """
        if not self.capabilities.supports_candidate_injection or not candidates:
            return state
        pop_key = self._population_payload_key(state)
        if pop_key is None:
            return state
        pop = state.payload[pop_key]
        if pop is None or len(pop.shape) != 2 or pop.shape[1] < self.problem.dimension + 1:
            return state
        fitness = pop[:, -1]
        descending = (self.problem.objective == "max")
        worst_idx = list(self._fitness_order(fitness, descending=descending))
        replaced = []
        for wi, cand in zip(worst_idx, candidates):
            pos = self._clip_position(cand.position)
            fit = self.problem.evaluate(pos)
            pop[wi, :-1] = pos
            pop[wi, -1] = fit
            state.evaluations += 1
            replaced.append(int(wi))
        state.payload[pop_key] = pop
        self._refresh_best_from_population(state, pop)
        self._post_injection_repair(state, replaced, candidates)
        return state

    # ------------------------------------------------------------------
    # Optional: population access
    # ------------------------------------------------------------------

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        """Raise NotImplementedError if has_population is False."""
        raise NotImplementedError(
            f"{self.algorithm_id} does not expose a population "
            "(capabilities.has_population is False)."
        )

    # ------------------------------------------------------------------
    # Optional: checkpoint / resume
    # ------------------------------------------------------------------

    def export_state(self, state: EngineState) -> dict:
        # NOTE: State export/import should support faithful cloning when needed.
        # cloning. The default implementation returns a reference to the live
        # state object, which is NOT a faithful clone on its own — consumers
        # that need a clone must perform
        # a deep copy themselves. Engine subclasses whose state contains
        # unpickleable handles should override both ``export_state`` and
        # ``import_state`` so the round-trip produces an independent instance.
        return {"algorithm_id": self.algorithm_id, "state": state}

    def import_state(self, payload: dict) -> EngineState:
        return payload["state"]

    def reconfigure(self, state: EngineState, params: dict[str, Any]) -> EngineState:
        """Optional runtime parameter update. Default behaviour is a safe no-op."""
        if not params:
            return state
        state.payload.setdefault("runtime_params", {}).update(dict(params))
        return state

    def restart(
        self,
        state: EngineState,
        seeds: list[CandidateRecord] | None = None,
        preserve_best: bool = True,
    ) -> EngineState:
        """Optional restart hook. Default behaviour reinjects seeds when possible, else no-op."""
        if seeds and self.capabilities.supports_candidate_injection:
            state = self.inject_candidates(state, seeds, policy="native")
        state.payload.setdefault("restart_metadata", {})["preserve_best"] = bool(preserve_best)
        return state

    # ------------------------------------------------------------------
    # Generic helpers for export / injection
    # ------------------------------------------------------------------

    def _evaluate_population(self, positions) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        fit = np.empty(pos.shape[0], dtype=float)
        for i in range(pos.shape[0]):
            fit[i] = self.problem.evaluate(pos[i, :])
        return fit

    def _population_payload_key(self, state: EngineState) -> str | None:
        for key in ("population", "sources", "positions", "memories"):
            if key in state.payload:
                return key
        return None

    def _clip_position(self, position) -> list[float]:
        return self.problem.apply_variable_types(position).astype(float).tolist()

    def _fitness_order(self, fitness_values, descending: bool = False):
        import numpy as _np
        idx = _np.argsort(fitness_values)
        return idx[::-1] if descending else idx

    def _refresh_best_from_population(self, state: EngineState, pop) -> None:
        idx = int(self._fitness_order(pop[:, -1], descending=(self.problem.objective == "max"))[0])
        best_row = pop[idx, :].copy()
        state.best_position = best_row[:-1].tolist()
        state.best_fitness = float(best_row[-1])

    def _diverse_candidate_indices(self, candidates: list[CandidateRecord], k: int) -> list[int]:
        import numpy as _np
        n = len(candidates)
        if n == 0:
            return []
        if k <= 1:
            best_idx = min(range(n), key=lambda i: candidates[i].fitness) if self.problem.objective == "min" else max(range(n), key=lambda i: candidates[i].fitness)
            return [best_idx]
        pos = _np.array([c.position for c in candidates], dtype=float)
        best_idx = min(range(n), key=lambda i: candidates[i].fitness) if self.problem.objective == "min" else max(range(n), key=lambda i: candidates[i].fitness)
        selected = [best_idx]
        while len(selected) < min(k, n):
            dists = _np.min(_np.linalg.norm(pos[None, :, :] - pos[selected, :][:, None, :], axis=2), axis=0)
            selected.append(int(_np.argmax(dists)))
        return selected

    def _post_injection_repair(self, state: EngineState, replaced_indices: list[int], candidates: list[CandidateRecord]) -> None:
        pop_key = self._population_payload_key(state)
        if pop_key is None:
            return
        pop = state.payload[pop_key]
        idx = int(self._fitness_order(pop[:, -1], descending=(self.problem.objective == "max"))[0])
        best_row = pop[idx, :].copy()
        for key in ("elite", "leader", "rabbit", "best"):
            if key in state.payload:
                value = state.payload[key]
                if hasattr(value, "shape") and len(value.shape) == 1 and value.shape[0] == pop.shape[1]:
                    state.payload[key] = best_row.copy()


# ---------------------------------------------------------------------------
# 8. Migration checklist helper (data only, no logic)
# ---------------------------------------------------------------------------

MIGRATION_CHECKLIST: list[str] = [
    "1. What is the native macro-step?",
    "2. What lives in payload?",
    "3. How are evaluations counted?",
    "4. How is best_position / best_fitness updated?",
    "5. Does it have a population?",
    "6. Can it export more than one candidate?",
    "7. Can it receive migrants / restarts?",
    "8. What telemetry fields are meaningful?",
    "9. What constitutes a faithful checkpoint?",
    "10. What stop conditions beyond the shared ones?",
]
