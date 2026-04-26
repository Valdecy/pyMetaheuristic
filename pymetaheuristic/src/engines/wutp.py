"""pyMetaheuristic src — Water Uptake and Transport in Plants Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WUTPEngine(PortedPopulationEngine):
    algorithm_id   = "wutp"
    algorithm_name = "Water Uptake and Transport in Plants"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1007/s00521-025-11059-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"J": 0.1*pop[:,:-1].copy(), "pbest": pop[:,:-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        J=state.payload["J"]; pbest=state.payload["pbest"]
        Lp=1e-9; Dp=1e-9; K=1e-9; rho=1000.; g=9.81; eta=0.0018; a=1.
        p=0.5; rr=0.1; lam=1-np.random.random()/2
        nu=1e-7/(1+np.exp((t-max_iter/2)/100))
        yo=pop[:,:-1].copy()
        # Water in motion
        for i in range(n):
            k=np.random.randint(n); pb=pbest[k]
            c1,c2=np.random.random(),np.random.random()
            J[i]+=lam*J[i]+c1*Lp*(pop[i,:-1]-pb)+c2*rho*g*(pop[i,:-1]-yo[i])
        # horizontal
        for i in range(n):
            for j in range(d):
                r3,r4,r5=np.random.random(),np.random.random(),np.random.random()
                if r3<p:
                    if r4>0.1: pop[i,j]-=J[i,j]/K
                    else: pop[i,j]-=nu*(lo[j]-(lo[j]-hi[j])*np.random.random())
                else:
                    if r5>0.1: pop[i,j]-=J[i,j]/(2*Dp*np.pi*a**2)
                    else: pop[i,j]-=nu*(lo[j]-(lo[j]-hi[j])*np.random.random())
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=new_fits
        mask=self._better_mask(new_fits,np.array([state.payload.get("pfit",np.full(n,np.inf))]).flatten() if False else new_fits)
        pbest_fits=np.array([float(self._evaluate_population(pbest[i][None])[0]) for i in range(n)])
        mask=self._better_mask(new_fits,pbest_fits)
        pbest[mask]=pop[mask,:-1]
        state.payload.update({"J":J,"pbest":pbest})
        return pop, evals, {}
