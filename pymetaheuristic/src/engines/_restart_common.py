"""Shared restart-capable engines used by local-search and CMA-ES variants."""
from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


class RestartLocalSearchEngine(BaseEngine):
    """Small restart-capable continuous local-search base class.

    The specialized engines in this package (ILS, GRASP, VNS, basin hopping,
    MSLS) supply the higher-level move logic and reuse the local-improvement and
    restart helpers below.
    """

    _DEFAULTS = {
        "step_size": 0.12,
        "min_step_size": 1.0e-8,
        "local_search_iters": 20,
        "neighborhood_size": None,
        "expand": 1.05,
        "contract": 0.55,
        "restart_stagnation_steps": 20,
    }

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _clip(self, position) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=float), self._lo, self._hi)

    def _is_better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return self.problem.is_better(float(a), float(b))

    def _energy(self, fitness: float) -> float:
        value = float(fitness)
        return value if self.problem.objective == "min" else -value

    def _random_position(self) -> np.ndarray:
        return np.random.uniform(self._lo, self._hi, self.problem.dimension)

    def initialize(self) -> EngineState:
        pos, fit, evals, delta = self._new_local_optimum()
        return EngineState(
            step=0,
            evaluations=int(evals),
            best_position=pos.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload={
                "current": pos,
                "current_fit": float(fit),
                "delta": float(delta),
                "stagnation": 0,
                "restarts": 0,
                "last_accepted": True,
            },
        )

    def _new_local_optimum(self) -> tuple[np.ndarray, float, int, float]:
        start = self._random_position()
        return self._local_search(start)

    def _local_search(self, start, start_fit: float | None = None) -> tuple[np.ndarray, float, int, float]:
        current = self._clip(start)
        if start_fit is None:
            current_fit = float(self.problem.evaluate(current))
            evals = 1
        else:
            current_fit = float(start_fit)
            evals = 0
        delta = float(self._params.get("step_size", 0.12))
        min_delta = float(self._params.get("min_step_size", 1.0e-8))
        max_iters = max(1, int(self._params.get("local_search_iters", 20)))
        neighborhood = self._params.get("neighborhood_size")
        if neighborhood is None:
            neighborhood = max(2, 2 * self.problem.dimension)
        neighborhood = max(1, int(neighborhood))
        for _ in range(max_iters):
            noise = np.random.uniform(-delta, delta, (neighborhood, self.problem.dimension)) * self._span
            trials = np.clip(current + noise, self._lo, self._hi)
            fit = np.asarray([float(self.problem.evaluate(row)) for row in trials], dtype=float)
            evals += trials.shape[0]
            idx = int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))
            if self._is_better(float(fit[idx]), current_fit):
                current = trials[idx].copy()
                current_fit = float(fit[idx])
                delta *= float(self._params.get("expand", 1.05))
            else:
                delta *= float(self._params.get("contract", 0.55))
            if delta <= min_delta:
                break
        return current, float(current_fit), int(evals), float(delta)

    def _restart_current(self, state: EngineState) -> tuple[np.ndarray, float, int, float]:
        pos, fit, evals, delta = self._new_local_optimum()
        state.payload["restarts"] = int(state.payload.get("restarts", 0)) + 1
        return pos, float(fit), int(evals), float(delta)

    def restart(self, state: EngineState, seeds: list[CandidateRecord] | None = None, preserve_best: bool = True) -> EngineState:
        if seeds:
            seed = seeds[0]
            pos, fit, evals, delta = self._local_search(seed.position)
        else:
            pos, fit, evals, delta = self._new_local_optimum()
        if (not preserve_best) or self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)
        state.payload.update(
            current=pos,
            current_fit=float(fit),
            delta=float(delta),
            stagnation=0,
            restarts=int(state.payload.get("restarts", 0)) + 1,
            last_accepted=True,
        )
        state.evaluations += int(evals)
        return state

    def observe(self, state: EngineState) -> dict:
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "current_fitness": state.payload.get("current_fit"),
            "delta": state.payload.get("delta"),
            "stagnation_steps": state.payload.get("stagnation", 0),
            "restarts": state.payload.get("restarts", 0),
            "last_accepted": state.payload.get("last_accepted"),
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
                "family": self.family,
                "restarts": int(state.payload.get("restarts", 0)),
                "elapsed_time": state.elapsed_time,
            },
        )


class RestartCMAESBase(BaseEngine):
    """Lightweight restart CMA-ES base for IPOP/BIPOP style engines."""

    _DEFAULTS = {
        "population_size": 20,
        "sigma": 0.30,
        "restart_stagnation_steps": 30,
        "population_multiplier": 2.0,
    }

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        self._base_lambda = max(4, int(self._params.get("population_size", 20)))
        self._population_multiplier = float(self._params.get("population_multiplier", 2.0))
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _is_better(self, a: float, b: float | None) -> bool:
        if b is None:
            return True
        return self.problem.is_better(float(a), float(b))

    def _next_lambda(self, restart_index: int, payload: dict | None = None) -> int:
        if restart_index <= 0:
            return self._base_lambda
        return int(round(self._base_lambda * (self._population_multiplier ** restart_index)))

    def _initial_payload(self, restart_index: int = 0) -> dict:
        lam = max(4, int(self._next_lambda(restart_index)))
        mean = np.random.uniform(self._lo, self._hi, self.problem.dimension)
        sigma = float(self._params.get("sigma", 0.30)) * self._span
        return {
            "mean": mean,
            "sigma": sigma,
            "lambda": lam,
            "restart_index": int(restart_index),
            "stagnation": 0,
            "population": None,
        }

    def initialize(self) -> EngineState:
        payload = self._initial_payload(0)
        fit = float(self.problem.evaluate(payload["mean"]))
        payload["population"] = np.hstack((payload["mean"].reshape(1, -1), np.asarray([[fit]], dtype=float)))
        return EngineState(
            step=0,
            evaluations=1,
            best_position=payload["mean"].tolist(),
            best_fitness=fit,
            initialized=True,
            payload=payload,
        )

    def _sample_population(self, payload: dict) -> tuple[np.ndarray, np.ndarray]:
        lam = max(4, int(payload.get("lambda", self._base_lambda)))
        mean = np.asarray(payload["mean"], dtype=float)
        sigma = np.asarray(payload["sigma"], dtype=float)
        positions = np.random.normal(loc=mean, scale=sigma, size=(lam, self.problem.dimension))
        positions = np.clip(positions, self._lo, self._hi)
        fitness = np.asarray([float(self.problem.evaluate(row)) for row in positions], dtype=float)
        return positions, fitness

    def step(self, state: EngineState) -> EngineState:
        payload = state.payload
        positions, fitness = self._sample_population(payload)
        order = np.argsort(fitness) if self.problem.objective == "min" else np.argsort(fitness)[::-1]
        mu = max(1, len(order) // 2)
        elite = positions[order[:mu]]
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        new_mean = np.sum(elite * weights.reshape(-1, 1), axis=0)
        best_idx = int(order[0])
        best_fit = float(fitness[best_idx])
        if self._is_better(best_fit, state.best_fitness):
            state.best_position = positions[best_idx].tolist()
            state.best_fitness = best_fit
            payload["stagnation"] = 0
            payload["sigma"] = np.maximum(np.asarray(payload["sigma"], dtype=float) * 0.98, 1.0e-12)
        else:
            payload["stagnation"] = int(payload.get("stagnation", 0)) + 1
            payload["sigma"] = np.minimum(np.asarray(payload["sigma"], dtype=float) * 1.02, self._span)
        payload["mean"] = np.clip(new_mean, self._lo, self._hi)
        payload["population"] = np.hstack((positions, fitness[:, None]))
        if int(payload.get("stagnation", 0)) >= int(self._params.get("restart_stagnation_steps", 30)):
            restart_index = int(payload.get("restart_index", 0)) + 1
            best_position = state.best_position
            best_fitness = state.best_fitness
            payload.clear()
            payload.update(self._initial_payload(restart_index))
            state.best_position = best_position
            state.best_fitness = best_fitness
        state.step += 1
        state.evaluations += int(len(fitness))
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload.get("population")
        mean_fitness = None if pop is None else float(np.mean(pop[:, -1]))
        std_fitness = None if pop is None else float(np.std(pop[:, -1]))
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": mean_fitness,
            "std_fitness": std_fitness,
            "diversity": self._compute_diversity(state),
            "sigma_mean": float(np.mean(np.asarray(state.payload.get("sigma", 0.0), dtype=float))),
            "restart_index": int(state.payload.get("restart_index", 0)),
            "population_size": int(state.payload.get("lambda", self._base_lambda)),
            "stagnation_steps": int(state.payload.get("stagnation", 0)),
        }

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = state.payload.get("population")
        if pop is None:
            return []
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

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def restart(self, state: EngineState, seeds: list[CandidateRecord] | None = None, preserve_best: bool = True) -> EngineState:
        restart_index = int(state.payload.get("restart_index", 0)) + 1
        best_position = list(state.best_position) if preserve_best and state.best_position is not None else None
        best_fitness = state.best_fitness if preserve_best else None
        state.payload.clear()
        state.payload.update(self._initial_payload(restart_index))
        if seeds:
            seed_pos = np.clip(np.asarray(seeds[0].position, dtype=float), self._lo, self._hi)
            state.payload["mean"] = seed_pos
            fit = float(self.problem.evaluate(seed_pos))
            state.evaluations += 1
            if best_fitness is None or self._is_better(fit, best_fitness):
                best_position = seed_pos.tolist()
                best_fitness = fit
        if best_position is not None:
            state.best_position = best_position
            state.best_fitness = float(best_fitness)
        return state

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
                "family": self.family,
                "restart_index": int(state.payload.get("restart_index", 0)),
                "population_size": int(state.payload.get("lambda", self._base_lambda)),
                "elapsed_time": state.elapsed_time,
            },
        )
