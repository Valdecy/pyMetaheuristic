"""pyMetaheuristic src — Glider Snake Optimization (GSO) Engine.

"""
from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineState,
    OptimizationResult,
)


class GSOGliderSnakeEngine(BaseEngine):
    """Glider Snake Optimization — chain-guided updates plus weak-agent replacement."""

    algorithm_id = "gso_glider_snake"
    algorithm_name = "Glider Snake Optimization"
    family = "swarm"
    _REFERENCE = {"doi": "10.1007/s10462-026-11504-x"}
    capabilities = CapabilityProfile(has_population=True)
    _DEFAULTS = {"population_size": 10, "mutation_rate": 0.5}

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        if "replacement_probability" in config.params and "mutation_rate" not in config.params:
            p["mutation_rate"] = config.params["replacement_probability"]
        self._n = int(p["population_size"])
        self._mutation_rate = float(p["mutation_rate"])
        if self._n < 4:
            raise ValueError("population_size must be at least 4 for Glider Snake Optimization.")
        if not (0.0 <= self._mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1].")
        if config.seed is not None:
            np.random.seed(config.seed)

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _score_ratios(self, fitness: np.ndarray) -> np.ndarray:
        fitness = np.asarray(fitness, dtype=float)
        work = fitness if self.problem.objective == "min" else -fitness
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
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload={"population": pop, "elite": elite},
        )

    def step(self, state):
        pop = np.asarray(state.payload["population"], dtype=float).copy()
        pop = pop[self._order(pop[:, -1])]
        leader = pop[0, :-1].copy()
        fitness = pop[:, -1].copy()
        ratios = self._score_ratios(fitness)
        T = max(1, self.config.max_steps or 100)
        A = 1.0 - (state.step / T)
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        new_pop = pop.copy()

        for s in range(self._n - 1, 0, -1):
            m = float(np.random.uniform(0.0, 1.0))
            rs1, rs2, rs3 = sorted(np.random.choice(self._n, size=3, replace=False).tolist())
            candidate = new_pop[s, :-1].copy()
            if self._mutation_rate > m and s > 0.5 * self._n:
                chain_index = rs1 / float(self._n)
                leader_ratio = ratios[0] / ratios[s]
                neighbour_ratio = ratios[rs2] / ratios[rs3]
                candidate = chain_index * pop[rs1, :-1] + A * (leader_ratio + neighbour_ratio)
            else:
                current = pop[s, :-1]
                d_global_leader = leader - current
                d_previous = pop[s - 1, :-1] - current
                candidate = current + A * (d_global_leader + d_previous)
            new_pop[s, :-1] = np.clip(candidate, lo, hi)

        new_pop[:, -1] = self._evaluate_population(new_pop[:, :-1])
        evals = self._n
        new_pop = new_pop[self._order(new_pop[:, -1])]
        elite = new_pop[0, :].copy()
        state.step += 1
        state.evaluations += evals
        state.payload = {"population": new_pop, "elite": elite}
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
                "aliases": {"replacement_probability": "mutation_rate"},
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
