"""pyMetaheuristic src — Barnacles Mating Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BMOEngine(PortedPopulationEngine):
    """Barnacles Mating Optimizer — penis-length threshold crossover between permuted pairs."""
    algorithm_id   = "bmo"
    algorithm_name = "Barnacles Mating Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ICOICA.2019.8895393"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pl=5)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        pl      = min(int(self._params.get("pl", 5)), n - 1)

        k1 = np.random.permutation(n)
        k2 = np.random.permutation(n)
        temp = np.abs(k1 - k2)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if temp[i] <= pl:
                p   = np.random.random()
                pos = p * pop[k1[i], :-1] + (1.0 - p) * pop[k2[i], :-1]
            else:
                pos = np.random.random() * pop[k2[i], :-1]
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # full replacement
        return pop, n, {}
