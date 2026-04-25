"""pyMetaheuristic src — Geyser Inspired Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GEAEngine(PortedPopulationEngine):
    algorithm_id   = "gea"
    algorithm_name = "Geyser Inspired Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1007/s42235-023-00426-5"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        Nc=max(1,n//3)
        order=self._order(pop[:,-1]); pop=pop[order]
        best_pos=pop[0,:-1].copy()
        fits=pop[:,-1].copy()
        CS=np.sort(fits); dif=CS[-1]-CS[0]
        for i in range(n):
            G=(fits[i]-CS[0])/(dif+1e-300)
            it_idx=t*n
            P_i=np.sqrt(max(0,(G**(2/max(it_idx,1)))-(G**((it_idx+1)/max(it_idx,1))))*(it_idx/max(it_idx-1,1)))
            # roulette wheel from top Nc
            imp_sum=sum(fits[:Nc])
            p_arr=fits[:Nc]/(imp_sum+1e-300)
            cs=np.cumsum(p_arr); r=np.random.random()
            i1=np.searchsorted(cs,r); i1=min(i1,Nc-1)
            # nearest by cosine similarity
            D_arr=[]
            for j in range(n):
                if j!=i:
                    ni=np.linalg.norm(pop[i,:-1]); nj=np.linalg.norm(pop[j,:-1])
                    sim=np.dot(pop[i,:-1],pop[j,:-1])/(ni*nj+1e-300)
                    D_arr.append((sim,j))
            D_arr.sort(); j1=D_arr[0][1]
            new_pos=np.clip(pop[j1,:-1]+np.random.random(d)*(pop[i1,:-1]-pop[j1,:-1])+np.random.random(d)*(pop[i1,:-1]-pop[i,:-1]),lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(new_pos,new_fit); fits[i]=new_fit
                if self._is_better(new_fit,pop[0,-1]): best_pos=new_pos.copy()
            # second attempt
            p2_sum=sum(1-p_arr)
            cs2=np.cumsum(1-p_arr)/max(p2_sum,1e-300); r2=np.random.random()
            i2=np.searchsorted(cs2,r2); i2=min(i2,Nc-1)
            new_pos2=np.clip(pop[i2,:-1]+np.random.random()*(P_i-np.random.random())*np.random.random(d)*(hi-lo)+lo,lo,hi)
            new_fit2=float(self._evaluate_population(new_pos2[None])[0]); evals+=1
            if self._is_better(new_fit2,pop[i,-1]):
                pop[i]=np.append(new_pos2,new_fit2); fits[i]=new_fit2
        return pop, evals, {}
