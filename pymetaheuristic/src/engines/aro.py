"""pyMetaheuristic src — Artificial Rabbits Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class AROEngine(PortedPopulationEngine):
    """Artificial Rabbits Optimization — detour foraging and random hiding strategies."""
    algorithm_id   = "aro"
    algorithm_name = "Artificial Rabbits Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2022.105082"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1

        theta  = 2.0 * (1.0 - t / T)                          # Eq. 15 decay factor
        new_pos = np.empty_like(pop[:, :-1])

        for i in range(n):
            # Sparse random direction vector R  (Eq. 2)
            L   = (np.e - np.exp((t / T) ** 2)) * np.sin(2.0 * np.pi * np.random.random())
            k   = max(1, int(np.ceil(np.random.random() * dim)))
            idx = np.random.choice(dim, k, replace=False)
            temp = np.zeros(dim); temp[idx] = 1.0
            R   = L * temp

            A = 2.0 * np.log(1.0 / (np.random.random() + 1e-30)) * theta  # Eq. 15

            if A > 1.0:                                        # Detour foraging  (Eq. 1)
                j     = np.random.randint(n)
                noise = np.round(0.5 * (0.05 + np.random.random())) * np.random.normal(0, 1)
                pos   = pop[j, :-1] + R * (pop[i, :-1] - pop[j, :-1]) + noise
            else:                                              # Random hiding  (Eqs. 11–13)
                k2    = max(1, int(np.ceil(np.random.random() * dim)))
                idx2  = np.random.choice(dim, k2, replace=False)
                gr    = np.zeros(dim); gr[idx2] = 1.0          # Eq. 12
                H     = np.random.normal(0, 1) * (t / T)      # Eq. 8
                b     = pop[i, :-1] + H * gr * pop[i, :-1]    # Eq. 13
                pos   = pop[i, :-1] + R * (np.random.random() * b - pop[i, :-1])  # Eq. 11

            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
