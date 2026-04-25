"""pyMetaheuristic src — Red-billed Blue Magpie Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RBMOEngine(PortedPopulationEngine):
    algorithm_id   = "rbmo"
    algorithm_name = "Red-billed Blue Magpie Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-024-10894-0"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, Epsilon=0.5)

    def _initialize_payload(self, pop):
        return {"X_old": pop[:,:-1].copy(), "fit_old": pop[:,-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        Epsilon=float(self._params.get("Epsilon",0.5))
        X_old=state.payload["X_old"]; fit_old=state.payload["fit_old"]
        best_idx=self._best_index(pop[:,-1]); Xfood=pop[best_idx,:-1].copy(); BestVal=pop[best_idx,-1]
        # Phase 1
        for i in range(n):
            p=min(n,max(2,np.random.randint(2,6))); idxs=np.random.choice(n,p,replace=False)
            Xpmean=np.mean(pop[idxs,:-1],axis=0); R1=np.random.randint(n)
            pop[i,:-1]+=((Xpmean-pop[R1,:-1])*np.random.random(d))
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=fits
        bi=self._best_index(pop[:,-1])
        if self._is_better(pop[bi,-1],BestVal): BestVal=pop[bi,-1]; Xfood=pop[bi,:-1].copy()
        # food storage
        Inx=fit_old<pop[:,-1]
        for k in range(n):
            if Inx[k]: pop[k,:-1]=X_old[k]; pop[k,-1]=fit_old[k]
        X_old=pop[:,:-1].copy(); fit_old=pop[:,-1].copy()
        CF=(1-t/max_iter)**(2*t/max_iter)
        # Phase 2
        for i in range(n):
            p=min(n,max(2,np.random.randint(2,6))); idxs=np.random.choice(n,p,replace=False)
            Xpmean=np.mean(pop[idxs,:-1],axis=0)
            q=min(n,max(2,np.random.randint(2,n+1))); qidxs=np.random.choice(n,q,replace=False)
            Xqmean=np.mean(pop[qidxs,:-1],axis=0)
            if np.random.random()<Epsilon:
                pop[i,:-1]=Xfood+CF*(Xpmean-pop[i,:-1])*np.random.randn(d)
            else:
                pop[i,:-1]=Xfood+CF*(Xqmean-pop[i,:-1])*np.random.randn(d)
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=fits
        bi=self._best_index(pop[:,-1])
        if self._is_better(pop[bi,-1],BestVal): BestVal=pop[bi,-1]; Xfood=pop[bi,:-1].copy()
        Inx=fit_old<pop[:,-1]
        for k in range(n):
            if Inx[k]: pop[k,:-1]=X_old[k]; pop[k,-1]=fit_old[k]
        state.payload.update({"X_old":pop[:,:-1].copy(),"fit_old":pop[:,-1].copy()})
        return pop, evals, {}
