"""pyMetaheuristic src — Poor and Rich Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class PROEngine(PortedPopulationEngine):
    algorithm_id   = "pro"
    algorithm_name = "Poor and Rich Optimization Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2019.08.025"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, pmut=0.06)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        pmut=float(self._params.get("pmut",0.06))
        nrich=n//2; npoor=n-nrich
        order=self._order(pop[:,-1])
        pop1=pop[order[:nrich],:-1].copy(); f1=pop[order[:nrich],-1].copy()
        pop2=pop[order[nrich:],:-1].copy(); f2=pop[order[nrich:],-1].copy()
        new1=pop1.copy(); new2=pop2.copy()
        for i in range(nrich):
            new1[i]+=np.random.random()*(pop1[i]-pop2[0])
            if np.random.random()<pmut: new1[i]+=np.random.randn(d)
            new1[i]=np.clip(new1[i],lo,hi)
        nf1=self._evaluate_population(new1); evals+=nrich
        meanr=np.mean(pop1,axis=0)
        pattern=(pop1[0]+meanr+pop1[-1])/3
        for j in range(npoor):
            new2[j]+=np.random.random()*(pattern-pop2[j])
            if np.random.random()<pmut: new2[j]+=np.random.randn(d)
            new2[j]=np.clip(new2[j],lo,hi)
        nf2=self._evaluate_population(new2); evals+=npoor
        all_pos=np.vstack([pop1,pop2,new1,new2])
        all_fit=np.concatenate([f1,f2,nf1,nf2])
        best_order=np.argsort(all_fit)[:n]
        pop=np.hstack([all_pos[best_order],all_fit[best_order,None]])
        return pop, evals, {}
