"""Shared utilities for native NumPy ports of additional engines."""
from __future__ import annotations

import math
import warnings
import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def safe_norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)) + 1.0e-12)


def levy_flight(dim: int, beta: float = 1.5, scale: float = 1.0) -> np.ndarray:
    beta = float(beta)
    sigma_u = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, int(dim))
    v = np.random.normal(0.0, 1.0, int(dim))
    step = u / (np.abs(v) ** (1.0 / beta) + 1.0e-12)
    return float(scale) * step


def cosine_distance_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    yn = np.linalg.norm(y, axis=1, keepdims=True).T
    sim = (x @ y.T) / (xn * yn + 1.0e-12)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def weighted_mean_rows(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    denom = float(np.sum(w)) + 1.0e-12
    return np.sum(rows * w, axis=0) / denom


def top_indices(fit: np.ndarray, objective: str, k: int) -> np.ndarray:
    idx = np.argsort(np.asarray(fit, dtype=float))
    if str(objective).lower() == "max":
        idx = idx[::-1]
    return idx[:max(1, int(k))]


class PortedPopulationEngine(BaseEngine):
    """Convenience base for population-shaped engines using the pymh protocol."""

    _DEFAULTS = {"population_size": 30}

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        _size = self._params.get("population_size", None)
        if _size is None:
            _size = self._params.get("swarm_size", 30)
        if _size is None:
            _size = 30
        self._n = max(2, int(_size))
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def _lo(self) -> np.ndarray:
        return _as_array(self.problem.min_values)

    @property
    def _hi(self) -> np.ndarray:
        return _as_array(self.problem.max_values)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, stacklevel=2)
            self._warned.add(key)

    def _new_positions(self, n: int | None = None) -> np.ndarray:
        return np.random.uniform(self._lo, self._hi, (self._n if n is None else int(n), self.problem.dimension))

    def _pop_from_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.clip(np.asarray(positions, dtype=float), self._lo, self._hi)
        fit = self._evaluate_population(positions)
        return np.hstack((positions, fit[:, None]))

    def _best_index(self, fit: np.ndarray) -> int:
        return int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))

    def _worst_index(self, fit: np.ndarray) -> int:
        return int(np.argmax(fit) if self.problem.objective == "min" else np.argmin(fit))

    def _order(self, fit: np.ndarray) -> np.ndarray:
        idx = np.argsort(fit)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _better_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a < b if self.problem.objective == "min" else a > b

    def _quality(self, fit: np.ndarray) -> np.ndarray:
        fit = np.asarray(fit, dtype=float)
        if fit.size == 0:
            return fit
        if self.problem.objective == "min":
            q = fit.max() - fit
        else:
            q = fit - fit.min()
        denom = float(np.ptp(q))
        return np.ones_like(q) if denom <= 1e-30 else (q - q.min()) / denom

    def _rand_indices(self, n: int, exclude: int, k: int) -> np.ndarray:
        pool = np.array([i for i in range(n) if i != exclude], dtype=int)
        if pool.size == 0:
            return np.array([exclude] * k, dtype=int)
        replace = pool.size < k
        return np.random.choice(pool, size=k, replace=replace)

    def _maybe_update_best(self, state: EngineState, pop: np.ndarray) -> None:
        idx = self._best_index(pop[:, -1])
        row = pop[idx]
        if state.best_fitness is None or self._is_better(row[-1], state.best_fitness):
            state.best_position = row[:-1].tolist()
            state.best_fitness = float(row[-1])

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {}

    def initialize(self) -> EngineState:
        # The source library exposes task-local history and repair hooks. pymh maps
        # those responsibilities to ProblemSpec.evaluate()/clip_position().
        self._warn_once(
            "task_mapping",
            f"[{self.algorithm_id}] Source task-level repair/history is mapped to ProblemSpec evaluation; source task history is not tracked.",
        )
        pop = self._pop_from_positions(self._new_positions())
        idx = self._best_index(pop[:, -1])
        payload = {"population": pop}
        payload.update(self._initialize_payload(pop))
        return EngineState(
            step=0,
            evaluations=pop.shape[0],
            best_position=pop[idx, :-1].tolist(),
            best_fitness=float(pop[idx, -1]),
            initialized=True,
            payload=payload,
        )

    def _step_impl(self, state: EngineState, pop: np.ndarray) -> tuple[np.ndarray, int, dict]:
        # Conservative fallback: Gaussian stochastic hill-climbing on each member.
        sigma = float(self._params.get("sigma", 0.1)) * self._span
        trial = np.clip(pop[:, :-1] + np.random.normal(0.0, sigma, pop[:, :-1].shape), self._lo, self._hi)
        fit = self._evaluate_population(trial)
        mask = self._better_mask(fit, pop[:, -1])
        pop = pop.copy()
        pop[mask, :-1] = trial[mask]
        pop[mask, -1] = fit[mask]
        return pop, trial.shape[0], {}

    def step(self, state: EngineState) -> EngineState:
        pop = np.asarray(state.payload["population"], dtype=float)
        pop, evals, updates = self._step_impl(state, pop.copy())
        pop[:, :-1] = np.clip(pop[:, :-1], self._lo, self._hi)
        state.payload["population"] = pop
        state.payload.update(updates or {})
        state.step += 1
        state.evaluations += int(evals)
        self._maybe_update_best(state, pop)
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(pop[:, -1])),
            "std_fitness": float(np.std(pop[:, -1])),
            "diversity": self._compute_diversity(state),
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "population_size": int(state.payload["population"].shape[0]),
                "elapsed_time": state.elapsed_time,
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = state.payload["population"]
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current",
            )
            for i in range(pop.shape[0])
        ]


def de_trial(engine: PortedPopulationEngine, pop: np.ndarray, i: int, F: float, CR: float, best: np.ndarray | None = None) -> np.ndarray:
    n, dim = pop.shape[0], engine.problem.dimension
    ids = engine._rand_indices(n, i, 3)
    r1, r2, r3 = pop[ids[0], :-1], pop[ids[1], :-1], pop[ids[2], :-1]
    base = best if best is not None else r1
    mutant = base + F * (r2 - r3)
    cross = np.random.rand(dim) < CR
    cross[np.random.randint(dim)] = True
    trial = np.where(cross, mutant, pop[i, :-1])
    return np.clip(trial, engine._lo, engine._hi)


class PortedTrajectoryEngine(BaseEngine):
    """Convenience base for single-trajectory engines."""

    _DEFAULTS = {"delta": 0.5}

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def _lo(self) -> np.ndarray:
        return _as_array(self.problem.min_values)

    @property
    def _hi(self) -> np.ndarray:
        return _as_array(self.problem.max_values)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, stacklevel=2)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once(
            "task_mapping",
            f"[{self.algorithm_id}] Source task-level repair/history is mapped to ProblemSpec evaluation; source task history is not tracked.",
        )
        pos = np.random.uniform(self._lo, self._hi, self.problem.dimension)
        fit = self.problem.evaluate(pos)
        return EngineState(
            step=0,
            evaluations=1,
            best_position=pos.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload={"current": pos, "current_fit": float(fit), "delta": float(self._params.get("delta", 0.5))},
        )

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def step(self, state: EngineState) -> EngineState:
        current = np.asarray(state.payload["current"], dtype=float)
        current_fit = float(state.payload["current_fit"])
        delta = float(state.payload.get("delta", self._params.get("delta", 0.5)))
        trials = []
        for _ in range(int(self._params.get("neighborhood_size", 2 * self.problem.dimension))):
            trials.append(np.clip(current + np.random.uniform(-delta, delta, self.problem.dimension) * self._span, self._lo, self._hi))
        trials = np.asarray(trials)
        fit = np.array([self.problem.evaluate(x) for x in trials])
        idx = int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))
        if self._is_better(fit[idx], current_fit):
            current, current_fit = trials[idx], float(fit[idx])
            delta *= float(self._params.get("expand", 1.05))
        else:
            delta *= float(self._params.get("contract", 0.95))
        state.payload.update({"current": current, "current_fit": current_fit, "delta": max(delta, 1e-12)})
        state.step += 1
        state.evaluations += len(trials)
        if self._is_better(current_fit, state.best_fitness):
            state.best_position = current.tolist()
            state.best_fitness = float(current_fit)
        return state

    def observe(self, state: EngineState) -> dict:
        return {"step": state.step, "evaluations": state.evaluations, "best_fitness": state.best_fitness, "delta": state.payload.get("delta")}

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(list(state.best_position), float(state.best_fitness), self.algorithm_id, state.step, "best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={"algorithm_name": self.algorithm_name, "elapsed_time": state.elapsed_time},
        )
