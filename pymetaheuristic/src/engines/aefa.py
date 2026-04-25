"""pyMetaheuristic src — Artificial Electric Field Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class AEFAEngine(PortedPopulationEngine):
    algorithm_id   = "aefa"
    algorithm_name = "Artificial Electric Field Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.swevo.2019.03.013"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, fper=3.0, K0=500.0, alfa=30.0)

    def _initialize_payload(self, pop):
        return {"V": np.zeros_like(pop[:,:-1])}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        V=state.payload["V"]; fper=float(self._params.get("fper",3.0))
        K0=float(self._params.get("K0",500.0)); alfa=float(self._params.get("alfa",30.0))
        fits=pop[:,-1].copy()
        Fmax,Fmin=fits.max(),fits.min()
        if Fmax==Fmin:
            Q=np.ones(n)
        else:
            Q=np.exp((fits-Fmax)/(Fmin-Fmax+1e-300))
        Q/=Q.sum()
        cbest=max(1,round(n*(fper+(1-t/max_iter)*(100-fper))/100))
        s=np.argsort(Q)[::-1]
        E=np.zeros((n,d))
        for i in range(n):
            for ii in range(cbest):
                j=s[ii]
                if j!=i:
                    R=np.linalg.norm(pop[i,:-1]-pop[j,:-1])+1e-300
                    E[i]+=np.random.random()*Q[j]*(pop[j,:-1]-pop[i,:-1])/(R+1e-300)
        K=K0*np.exp(-alfa*t/max_iter)
        a=E*K
        V=np.random.random((n,d))*V+a
        new_pos=np.clip(pop[:,:-1]+V,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        pop=np.hstack([new_pos,new_fits[:,None]]); state.payload["V"]=V
        return pop, evals, {}
