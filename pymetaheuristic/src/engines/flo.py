"""pyMetaheuristic src — Frilled Lizard Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class FLOEngine(PortedPopulationEngine):
    """Frilled Lizard Optimization.

    Two-phase bio-inspired optimizer: hunting/prey attack for exploration and
    moving up a nearby tree for local exploitation.
    """

    algorithm_id = "flo"
    algorithm_name = "Frilled Lizard Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.32604/cmc.2024.053189",
        "title": "Frilled Lizard Optimization: A Novel Bio-Inspired Optimizer for Solving Engineering Applications",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _better_indices_than(self, pop: np.ndarray, i: int) -> np.ndarray:
        if self.problem.objective == "min":
            idx = np.where(pop[:, -1] < pop[i, -1])[0]
        else:
            idx = np.where(pop[:, -1] > pop[i, -1])[0]
        idx = idx[idx != i]
        if idx.size == 0:
            idx = np.array([self._best_index(pop[:, -1])], dtype=int)
        return idx

    def _greedy_single(self, pop: np.ndarray, i: int, x: np.ndarray) -> bool:
        x = np.clip(x, self._lo, self._hi)
        fx = float(self.problem.evaluate(x))
        if self._is_better(fx, float(pop[i, -1])):
            pop[i, :-1] = x
            pop[i, -1] = fx
            return True
        return False

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, state.step + 1)
        evals = 0

        # Phase 1 — hunting strategy / attack toward a randomly selected better prey.
        for i in range(n):
            prey_idx = int(np.random.choice(self._better_indices_than(pop, i)))
            prey = pop[prey_idx, :-1]
            r = np.random.rand(dim)
            I = np.random.randint(1, 3, size=dim)
            trial = pop[i, :-1] + r * (prey - I * pop[i, :-1])
            self._greedy_single(pop, i, trial)
            evals += 1

        # Phase 2 — moving up the tree / small local movement around current position.
        for i in range(n):
            r = np.random.rand(dim)
            trial = pop[i, :-1] + (1.0 - 2.0 * r) * self._span / float(t)
            self._greedy_single(pop, i, trial)
            evals += 1

        return pop, evals, {}
