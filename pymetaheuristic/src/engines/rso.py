"""pyMetaheuristic src — Rat Swarm Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RSOEngine(PortedPopulationEngine):
    algorithm_id   = "rso"
    algorithm_name = "Rat Swarm Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s12652-020-02580-0"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        R=int(np.floor((5-1)*np.random.random()+1))
        A=R-t*(R/max_iter)
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        fits=[float(self._evaluate_population(pop[i,:-1][None])[0]) for i in range(n)]; evals+=n
        best_idx=int(np.argmin(fits)); best_pos=pop[best_idx,:-1].copy(); best_fit=fits[best_idx]
        for i in range(n):
            for j in range(d):
                C=2*np.random.random()
                P_vec=A*pop[i,j]+abs(C*(best_pos[j]-pop[i,j]))
                pop[i,j]=best_pos[j]-P_vec
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n
        pop[:,-1]=new_fits
        return pop, evals, {}
