"""pyMetaheuristic src — Lungs Performance-Based Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class LPOEngine(PortedPopulationEngine):
    algorithm_id   = "lpo"
    algorithm_name = "Lungs Performance-Based Optimization"
    family         = "bio"
    _REFERENCE     = {"doi": "10.1016/j.cma.2023.116582"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"Delta": np.random.random(pop.shape[0])*2*np.pi,
                "sigma1": [np.random.random(self.problem.dimension) for _ in range(pop.shape[0])]}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        Delta=state.payload["Delta"]; sigma1=state.payload["sigma1"]
        fits=pop[:,-1].copy(); Pos1=np.zeros(d)
        for i in range(n):
            for jj in range(1,6):
                R=fits[i]; C=(R/2)*np.sin(Delta[i])
                newsol=pop[i,:-1].copy(); newsol2=pop[i,:-1].copy()
                denom=(2*np.pi*d*R*max(C,1e-300))**2
                factor=(R**2+1/max(denom,1e-300))**-0.5
                if jj==1:
                    newsol=pop[i,:-1]+factor*np.sin(2*np.pi*d*t)*np.sin(2*np.pi*d*t+Delta[i])*pop[i,:-1]
                else:
                    newsol=pop[i,:-1]+factor*np.sin(2*np.pi*d*t)*np.sin(2*np.pi*d*t+Delta[i])*Pos1
                perm=np.random.permutation(n); a1,a2,a3=perm[0],perm[1],perm[2]
                aa1=np.sign(fits[a2]-fits[a3]) if fits[a2]!=fits[a3] else 1.0
                aa2=np.sign(fits[a1]-fits[i]) if fits[a1]!=fits[i] else 1.0
                newsol2=pop[a1,:-1]+sigma1[i]*(pop[a3,:-1]-pop[a2,:-1])
                newsol=newsol+aa2*sigma1[i]*(newsol-pop[a1,:-1])+aa1*sigma1[i]*(pop[a3,:-1]-pop[a2,:-1])
                for j in range(d):
                    Pos1[j]=newsol2[j] if np.random.random()/jj>np.random.random() else newsol[j]
                Pos1=np.clip(Pos1,lo,hi); Delta[i]=np.arctan(1/(2*np.pi*d*max(R,1e-300)*max(C,1e-300)))
                newsol=Pos1
                newCost=float(self._evaluate_population(newsol[None])[0]); evals+=1
                if self._is_better(newCost,pop[i,-1]):
                    pop[i]=np.append(newsol,newCost); fits[i]=newCost
                sigma1[i]=np.random.random(d)
        state.payload.update({"Delta":Delta,"sigma1":sigma1})
        return pop, evals, {}
