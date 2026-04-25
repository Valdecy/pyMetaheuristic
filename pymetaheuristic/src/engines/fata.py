"""pyMetaheuristic src — FATA Geophysics Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FATAEngine(PortedPopulationEngine):
    algorithm_id   = "fata"
    algorithm_name = "FATA Geophysics Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.neucom.2024.128289"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, arf=0.2)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        arf=float(self._params.get("arf",0.2))
        FEs=t*n; MaxFEs=max_iter*d
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        best_idx=self._best_index(pop[:,-1]); gBest=pop[best_idx,:-1].copy(); gBestScore=pop[best_idx,-1]
        fits=pop[:,-1].copy()
        order=self._order(fits); worstF=fits[order[-1]]; bestF=fits[order[0]]
        a=np.tan(-(FEs/max(MaxFEs,1))+1); b=1/np.tan(-(FEs/max(MaxFEs,1))+1)
        new_pos=pop[:,:-1].copy()
        for i in range(n):
            P1=a*np.random.random(d)-a*np.random.random(d)
            P2=b*np.random.random(d)-b*np.random.random(d)
            p=(fits[i]-worstF)/(gBestScore-worstF+1e-300)
            IP=max(min(p,1),0)
            if np.random.random()>IP:
                new_pos[i]=np.random.uniform(lo,hi)
            else:
                for j in range(d):
                    num=np.random.randint(n)
                    if np.random.random()<p:
                        new_pos[i,j]=gBest[j]+pop[i,j]*P1[j]
                    else:
                        new_pos[i,j]=pop[num,j]+P2[j]*pop[i,j]
                        new_pos[i,j]=0.5*(arf+1)*(lo[j]+hi[j])-arf*new_pos[i,j]
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
