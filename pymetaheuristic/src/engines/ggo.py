"""pyMetaheuristic src — Greylag Goose Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GGOEngine(PortedPopulationEngine):
    algorithm_id   = "ggo"
    algorithm_name = "Greylag Goose Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2023.122147"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        best_idx=self._best_index(pop[:,-1]); best_pos=pop[best_idx,:-1].copy()
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            new_pos[i]=np.clip(pop[i,:-1]+np.random.random(d)*(best_pos-pop[i,:-1]),lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
