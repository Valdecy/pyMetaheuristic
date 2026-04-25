"""pyMetaheuristic src — Deep Sleep Optimiser Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class DSOEngine(PortedPopulationEngine):
    algorithm_id   = "dso"
    algorithm_name = "Deep Sleep Optimiser"
    family         = "human"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2023.3299804"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        T=24; H0_minus=0.17; H0_plus=0.85; a_=0.1; xs=4.2; xw=18.2
        C=np.sin(2*np.pi/T); H_min=H0_minus+a_*C; H_max=H0_plus+a_*C
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        new_pos=np.empty_like(pop[:,:-1])
        for j in range(n):
            mu=np.random.random(); mu=np.clip(mu,H_min,H_max)
            H0=pop[j,:-1]+np.random.random(d)*(best_pos-mu*pop[j,:-1])
            if np.random.random()>mu:
                H=H0*10**(-1/xs)
            else:
                H=mu+(H0-mu)*10**(-1/xw)
            new_pos[j]=np.clip(H,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
