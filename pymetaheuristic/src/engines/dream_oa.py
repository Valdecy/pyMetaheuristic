"""pyMetaheuristic src — Dream Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class DreamOAEngine(PortedPopulationEngine):
    """Dream Optimization Algorithm.

    The registry id is ``dream_oa`` to avoid colliding with the existing
    ``doa`` engine, which implements the Deer Hunting Optimization Algorithm.
    """

    algorithm_id = "dream_oa"
    algorithm_name = "Dream Optimization Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1016/j.cma.2024.117718",
        "title": "Dream Optimization Algorithm (DOA): A novel metaheuristic optimization algorithm inspired by human dreams and its applications to real-world engineering problems",
        "authors": "Yifan Lang, Yuelin Gao",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, exploration_fraction=0.9, memory_probability=0.9)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 5:
            raise ValueError("dream_oa requires population_size >= 5.")
        exploration_fraction = float(self._params.get("exploration_fraction", 0.9))
        memory_probability = float(self._params.get("memory_probability", 0.9))
        if not 0.0 < exploration_fraction < 1.0:
            raise ValueError("dream_oa exploration_fraction must be in (0, 1).")
        if not 0.0 <= memory_probability <= 1.0:
            raise ValueError("dream_oa memory_probability must be in [0, 1].")

    def _group_best(self, pop: np.ndarray, start: int, end: int) -> np.ndarray:
        group = pop[start:end]
        if group.shape[0] == 0:
            idx = self._best_index(pop[:, -1])
            return pop[idx, :-1].copy()
        idx = self._best_index(group[:, -1])
        return group[idx, :-1].copy()

    def _random_dimension_count(self, group_index: int) -> int:
        dim = self.problem.dimension
        aa = max(1, int(np.ceil(dim / 8.0 / (group_index + 1))))
        bb = int(np.ceil(dim / 3.0 / (group_index + 1))) + 1
        bb = max(aa + 1, bb)
        return max(1, min(dim, int(np.random.randint(aa, bb))))

    def _replace_out_of_bounds(self, value: float, dim_index: int, pop: np.ndarray, agent_index: int) -> float:
        if self._lo[dim_index] <= value <= self._hi[dim_index]:
            return float(value)
        if self.problem.dimension > 15 and pop.shape[0] > 1:
            choices = [k for k in range(pop.shape[0]) if k != agent_index]
            return float(pop[int(np.random.choice(choices)), dim_index])
        return float(np.random.uniform(self._lo[dim_index], self._hi[dim_index]))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, int(self.config.max_steps or 500))
        t = min(T, int(state.step) + 1)
        exploration_fraction = float(self._params.get("exploration_fraction", 0.9))
        memory_probability = float(self._params.get("memory_probability", 0.9))
        exploration_end = int(np.floor(exploration_fraction * T))

        new_pos = np.empty((n, dim), dtype=float)
        if t <= exploration_end:
            for m in range(5):
                group_start = int((m / 5.0) * n)
                group_end = int(((m + 1.0) / 5.0) * n)
                if group_start >= group_end:
                    continue
                pbest = self._group_best(pop, group_start, group_end)
                kk = self._random_dimension_count(m)
                for i in range(group_start, group_end):
                    candidate = pbest.copy()
                    dims = np.random.choice(dim, size=kk, replace=False)
                    if np.random.random() < memory_probability:
                        cos_term = (np.cos((t + T / 10.0) * np.pi / T) + 1.0) / 2.0
                        for j in dims:
                            candidate[j] = pbest[j] + (np.random.random() * self._span[j] + self._lo[j]) * cos_term
                            candidate[j] = self._replace_out_of_bounds(candidate[j], j, pop, i)
                    else:
                        for j in dims:
                            choices = [k for k in range(n) if k != i]
                            candidate[j] = pop[int(np.random.choice(choices)), j]
                    new_pos[i] = np.clip(candidate, self._lo, self._hi)
        else:
            best = pop[self._order(pop[:, -1])[0], :-1].copy()
            cos_term = (np.cos(t * np.pi / T) + 1.0) / 2.0
            for i in range(n):
                km = max(2, int(np.ceil(dim / 3.0)))
                kk = max(1, min(dim, int(np.random.randint(2, km + 1))))
                dims = np.random.choice(dim, size=kk, replace=False)
                candidate = best.copy()
                for j in dims:
                    candidate[j] = candidate[j] + (np.random.random() * self._span[j] + self._lo[j]) * cos_term
                    candidate[j] = self._replace_out_of_bounds(candidate[j], j, pop, i)
                new_pos[i] = np.clip(candidate, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        return np.hstack((new_pos, new_fit[:, None])), n, {}
