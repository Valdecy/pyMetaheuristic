"""pyMetaheuristic src — Artemisinin Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ArtemisininOEngine(PortedPopulationEngine):
    algorithm_id   = "artemisinin_o"
    algorithm_name = "Artemisinin Optimization"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1016/j.displa.2024.102740"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        FEs=t*n; MaxFEs=max_iter*n
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        K=1-(FEs**(1/6)/MaxFEs**(1/6))
        E=np.exp(-4*(FEs/MaxFEs))
        fits=pop[:,-1].copy()
        fmin,fmax=fits.min(),fits.max()
        Fitnorm=(fits-fmin)/(fmax-fmin+1e-300)
        new_pop=pop[:,:-1].copy()
        for i in range(n):
            for j in range(d):
                if np.random.random()<K:
                    if np.random.random()<0.5:
                        new_pop[i,j]=pop[i,j]+E*pop[i,j]*(-1)**t
                    else:
                        new_pop[i,j]=pop[i,j]+E*best_pos[j]*(-1)**t
                else:
                    new_pop[i,j]=pop[i,j]
                if np.random.random()<Fitnorm[i]:
                    A=np.random.permutation(n); beta=np.random.random()/2+0.1
                    new_pop[i,j]=pop[A[2],j]+beta*(pop[A[0],j]-pop[A[1],j])
            # mutation
            for j in range(d):
                if np.random.random()<0.05: new_pop[i,j]=pop[i,j]
                if np.random.random()<0.2: new_pop[i,j]=best_pos[j]
            # boundary
            for j in range(d):
                if new_pop[i,j]>hi[j] or new_pop[i,j]<lo[j]:
                    new_pop[i,j]=best_pos[j]
        new_pos=np.clip(new_pop,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
