"""pyMetaheuristic src — Ivy Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class IVYAEngine(PortedPopulationEngine):
    algorithm_id   = "ivya"
    algorithm_name = "Ivy Algorithm"
    family         = "bio"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2024.111850"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        span=self._hi-self._lo; span=np.where(span==0,1,span)
        return {"GV": pop[:,:-1]/(span)}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        span=hi-lo; span=np.where(span==0,1,span)
        GV=state.payload["GV"]
        order=self._order(pop[:,-1]); pop=pop[order]; GV=GV[order]
        new_positions=[]; new_costs=[]; new_GVs=[]
        for i in range(n):
            ii=(i+1)%n; beta_1=1+np.random.random()/2
            if pop[i,-1]<beta_1*pop[0,-1]:
                new_pos=pop[i,:-1]+np.abs(np.random.randn(d))*(pop[ii,:-1]-pop[i,:-1])+np.random.randn(d)*GV[i]
            else:
                new_pos=pop[0,:-1]*(np.random.random()+np.random.randn(d)*GV[i])
            GV[i]*=(np.random.random()**2)*np.random.randn(d)
            new_pos=np.clip(new_pos,lo,hi)
            new_gv=new_pos/span
            new_cost=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            new_positions.append(new_pos); new_costs.append(new_cost); new_GVs.append(new_gv)
        all_pos=np.vstack([pop[:,:-1],np.array(new_positions)])
        all_fit=np.concatenate([pop[:,-1],new_costs])
        all_gv=np.vstack([GV,np.array(new_GVs)])
        ord2=np.argsort(all_fit)[:n]
        pop=np.hstack([all_pos[ord2],all_fit[ord2,None]]); GV=all_gv[ord2]
        state.payload["GV"]=GV
        return pop, evals, {}
