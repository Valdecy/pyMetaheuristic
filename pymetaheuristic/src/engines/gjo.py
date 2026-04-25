"""pyMetaheuristic src — Golden Jackal Optimizer Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_gjo(n: int, dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, (n, dim))
    v = np.abs(np.random.normal(0, 1, (n, dim))) + 1e-30
    return 0.05 * u / v ** (1 / beta)

class GJOEngine(PortedPopulationEngine):
    """Golden Jackal Optimizer — cooperative predation by male and female jackal pair."""
    algorithm_id   = "gjo"
    algorithm_name = "Golden Jackal Optimizer"
    family         = "swarm"
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

        order  = self._order(pop[:, -1])
        male   = pop[order[0], :-1].copy()
        female = pop[order[1], :-1].copy() if n > 1 else male.copy()

        E1  = 1.5 * (1.0 - t / T)
        RL  = _levy_gjo(n, dim)
        new_pos = np.empty_like(pop[:, :-1])

        for i in range(n):
            male_pos   = male.copy()
            female_pos = female.copy()
            for j in range(dim):
                r1  = np.random.random()
                E0  = 2.0 * r1 - 1.0
                E   = E1 * E0
                if abs(E) < 1:                              # Exploitation
                    t1 = abs(RL[i, j] * male[j]   - pop[i, j])
                    t2 = abs(RL[i, j] * female[j] - pop[i, j])
                    male_pos[j]   = male[j]   - E * t1
                    female_pos[j] = female[j] - E * t2
                else:                                       # Exploration
                    t1 = abs(male[j]   - RL[i, j] * pop[i, j])
                    t2 = abs(female[j] - RL[i, j] * pop[i, j])
                    male_pos[j]   = male[j]   - E * t1
                    female_pos[j] = female[j] - E * t2
            new_pos[i] = np.clip((male_pos + female_pos) / 2.0, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        # GJO replaces unconditionally in original; use greedy here
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
