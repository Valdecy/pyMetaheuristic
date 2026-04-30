"""pyMetaheuristic src — Improved Adaptive Grey Wolf Optimization (IAGWO) Engine."""
from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineState,
    OptimizationResult,
)


class IAGWOEngine(BaseEngine):
    """IAGWO — PSO-style search, IMF inertia weighting, and adaptive scaling."""

    algorithm_id = "iagwo"
    algorithm_name = "Improved Adaptive Grey Wolf Optimization"
    family = "swarm"
    _REFERENCE = {"doi": "10.1007/s10462-024-10821-3"}
    capabilities = CapabilityProfile(has_population=True)
    _DEFAULTS = {
        "population_size": 30,
        "theta": 0.5,
        "imf_a": 0.6,
        "imf_b": 0.02,
        "imf_c": 0.05,
        "imf_d": 0.3,
        "velocity_clamp": None,
    }

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])
        self._theta = float(p["theta"])
        self._imf_a = float(p["imf_a"])
        self._imf_b = float(p["imf_b"])
        self._imf_c = float(p["imf_c"])
        self._imf_d = float(p["imf_d"])
        self._velocity_clamp = p["velocity_clamp"]
        if self._n < 4:
            raise ValueError("population_size must be at least 4 for IAGWO.")
        if self._theta <= 0.0:
            raise ValueError("theta must be > 0.")
        if config.seed is not None:
            np.random.seed(config.seed)

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _objective_scores(self, fitness: np.ndarray) -> np.ndarray:
        raw = np.asarray(fitness, dtype=float)
        work = raw if self.problem.objective == "min" else -raw
        return work - work.min() + 1e-12

    def _init_pop(self, n=None):
        if n is None:
            n = self._n
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        return np.hstack((pos, fit[:, np.newaxis]))

    def initialize(self):
        pop = self._init_pop()
        order = self._order(pop[:, -1])
        pop = pop[order]
        elite = pop[0, :].copy()
        velocity = np.zeros((self._n, self.problem.dimension), dtype=float)
        selfbest_pos = pop[:, :-1].copy()
        selfbest_fit = pop[:, -1].copy()
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload={
                "population": pop,
                "elite": elite,
                "velocity": velocity,
                "selfbest_position": selfbest_pos,
                "selfbest_fitness": selfbest_fit,
            },
        )

    def _imf_weight(self, t: int) -> float:
        return float(self._imf_a * np.exp(-self._imf_b * np.exp(-self._imf_c * t)) + self._imf_d)

    def step(self, state):
        pop = np.asarray(state.payload["population"], dtype=float).copy()
        pop = pop[self._order(pop[:, -1])]
        velocity = np.asarray(state.payload["velocity"], dtype=float).copy()
        selfbest_position = np.asarray(state.payload["selfbest_position"], dtype=float).copy()
        selfbest_fitness = np.asarray(state.payload["selfbest_fitness"], dtype=float).copy()

        pos = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        alpha, beta, delta = pos[0].copy(), pos[1].copy(), pos[2].copy()
        best_pos = alpha.copy()
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        span = hi - lo

        phi = np.random.random((self._n, self.problem.dimension))
        velocity += phi * (best_pos - pos) + phi * (selfbest_position - pos)
        if self._velocity_clamp is not None:
            vmax = float(self._velocity_clamp)
            velocity = np.clip(velocity, -vmax, vmax)
        pso_pos = np.clip(pos + velocity, lo, hi)

        T = max(1, self.config.max_steps or 500)
        t = state.step + 1
        a_linear = 2.0 - 2.0 * (t / T)
        omega = self._imf_weight(t)

        updated = np.empty_like(pso_pos)
        for i in range(self._n):
            x = pso_pos[i]
            r1 = np.random.random(self.problem.dimension)
            r2 = np.random.random(self.problem.dimension)
            A1 = 2.0 * a_linear * r1 - a_linear
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha - x)
            X1 = alpha - omega * A1 * D_alpha

            r1 = np.random.random(self.problem.dimension)
            r2 = np.random.random(self.problem.dimension)
            A2 = 2.0 * a_linear * r1 - a_linear
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta - x)
            X2 = beta - omega * A2 * D_beta

            r1 = np.random.random(self.problem.dimension)
            r2 = np.random.random(self.problem.dimension)
            A3 = 2.0 * a_linear * r1 - a_linear
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta - x)
            X3 = delta - omega * A3 * D_delta

            y_i = (X1 + X2 + X3) / 3.0
            updated[i] = y_i

        score = self._objective_scores(fit)
        fave = float(np.mean(score)) if float(np.mean(score)) != 0.0 else 1.0
        aggregation = score / fave
        adaptive = 1.0 / (1.0 + np.exp(-(aggregation * self._theta)))
        updated *= adaptive[:, None]
        updated = np.clip(updated, lo, hi)
        new_fit = self._evaluate_population(updated)
        evals = self._n

        better_self = np.array([self.problem.is_better(float(new_fit[i]), float(selfbest_fitness[i])) for i in range(self._n)])
        selfbest_position[better_self] = updated[better_self]
        selfbest_fitness[better_self] = new_fit[better_self]

        pop = np.hstack((updated, new_fit[:, None]))
        pop = pop[self._order(pop[:, -1])]
        elite = pop[0, :].copy()

        state.step += 1
        state.evaluations += evals
        state.payload = {
            "population": pop,
            "elite": elite,
            "velocity": velocity,
            "selfbest_position": selfbest_position,
            "selfbest_fitness": selfbest_fitness,
        }
        if self.problem.is_better(float(elite[-1]), float(state.best_fitness)):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        pos = pop[:, :-1]
        fitness = pop[:, -1]
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        denom = np.linalg.norm(hi - lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "diversity": diversity,
        }

    def get_best_candidate(self, state):
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def finalize(self, state):
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
                "elapsed_time": state.elapsed_time,
                "reference": self._REFERENCE,
                "assumptions": [
                    "The sigmoid-based adaptive coefficient is implemented as phi_i = 1 / (1 + exp(-(f_i/f_avg) * theta)) to resolve the paper's ambiguous typography.",
                    "The PSO search mechanism is applied before the IMF-weighted GWO position update at every iteration.",
                ],
            },
        )

    def get_population(self, state):
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
