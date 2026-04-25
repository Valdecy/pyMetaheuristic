"""pyMetaheuristic src — Supply-Demand-Based Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SupplyDOEngine(PortedPopulationEngine):
    algorithm_id   = "supply_do"
    algorithm_name = "Supply-Demand-Based Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2019.2919408"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        a=2*(max_iter-t+1)/max_iter
        qty=pop[:,:-1].copy(); price=pop[:,:-1].copy()
        qty_fit=pop[:,-1].copy(); price_fit=pop[:,-1].copy()
        best_fit=pop[self._best_index(pop[:,-1]),-1]
        best_x=pop[self._best_index(pop[:,-1]),:-1].copy()
        F=np.abs(qty_fit-np.mean(qty_fit))+1e-15; FQ=F/F.sum()
        F2=np.abs(price_fit-np.mean(price_fit))+1e-15; FP=F2/F2.sum()
        MeanPrice=np.mean(price,axis=0)
        for i in range(n):
            k=self._roulette(FQ)
            qeq=qty[k].copy()
            Alpha=a*np.sin(2*np.pi*np.random.random(d))
            Beta=2*np.cos(2*np.pi*np.random.random(d))
            if np.random.random()>0.5:
                peq=np.random.random()*MeanPrice
            else:
                k2=self._roulette(FP)
                peq=price[k2].copy()
            new_qty=np.clip(qeq+Alpha*(price[i]-peq),lo,hi)
            new_qty_fit=float(self._evaluate_population(new_qty[None])[0]); evals+=1
            if self._is_better(new_qty_fit,qty_fit[i]):
                qty[i]=new_qty; qty_fit[i]=new_qty_fit
            new_price=np.clip(peq-Beta*(new_qty-qeq),lo,hi)
            new_price_fit=float(self._evaluate_population(new_price[None])[0]); evals+=1
            if self._is_better(new_price_fit,price_fit[i]):
                price[i]=new_price; price_fit[i]=new_price_fit
        for i in range(n):
            if self._is_better(qty_fit[i],price_fit[i]):
                price_fit[i]=qty_fit[i]; price[i]=qty[i]
            if self._is_better(price_fit[i],best_fit):
                best_x=price[i].copy(); best_fit=price_fit[i]
        pop=np.hstack([price,price_fit[:,None]])
        return pop, evals, {}

    def _roulette(self, p):
        cp=np.cumsum(p); r=np.random.random()
        idx=np.searchsorted(cp,r)
        return min(idx,len(p)-1)
