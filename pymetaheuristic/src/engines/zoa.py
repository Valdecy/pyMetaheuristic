"""pyMetaheuristic src — Zebra Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ZOAEngine(PortedPopulationEngine):
    """Zebra Optimization Algorithm — foraging and anti-predator defense strategies."""
    algorithm_id   = "zoa"
    algorithm_name = "Zebra Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2022.3172789"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # Phase 1: Foraging (Eq. 3)
        new_pos1 = np.empty_like(pop[:, :-1])
        for i in range(n):
            r1  = float(np.round(1.0 + np.random.random()))
            pos = pop[i, :-1] + np.random.random(dim) * (best_pos - r1 * pop[i, :-1])
            new_pos1[i] = np.clip(pos, self._lo, self._hi)
        new_fit1 = self._evaluate_population(new_pos1); evals += n
        mask1    = self._better_mask(new_fit1, pop[:, -1])
        pop[mask1] = np.hstack([new_pos1, new_fit1[:, None]])[mask1]

        # Phase 2: Defence against predators
        kk = np.random.randint(n)
        new_pos2 = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() < 0.5:            # S1: lion escape
                r2  = 0.1
                pos = pop[i, :-1] + r2 * (2.0 * np.random.random(dim) - 1.0) * (1.0 - t / T) * pop[i, :-1]
            else:                                   # S2: offensive
                r2  = np.random.randint(1, 3)
                pos = pop[i, :-1] + np.random.random(dim) * (pop[kk, :-1] - r2 * pop[i, :-1])
            new_pos2[i] = np.clip(pos, self._lo, self._hi)
        new_fit2 = self._evaluate_population(new_pos2); evals += n
        mask2    = self._better_mask(new_fit2, pop[:, -1])
        pop[mask2] = np.hstack([new_pos2, new_fit2[:, None]])[mask2]

        return pop, evals, {}
