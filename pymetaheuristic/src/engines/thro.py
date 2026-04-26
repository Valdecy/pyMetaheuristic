"""pyMetaheuristic src — Tianji Horse Racing Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class THROEngine(PortedPopulationEngine):
    """Tianji Horse Racing Optimizer — binary mask swap between Tianji and King horse groups."""
    algorithm_id = "thro"; algorithm_name = "Tianji Horse Racing Optimizer"; family = "human"
    _REFERENCE     = {"doi": "10.1007/s10462-025-11269-9"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        half=n//2; p=1-t/T; evals=0
        np.random.shuffle(pop); order_t=self._order(pop[:half,-1]); order_k=self._order(pop[half:,-1])
        t_pop=pop[:half][order_t]; k_pop=pop[half:][order_k]
        t_b=np.zeros((half,dim),int); k_b=np.zeros((half,dim),int)
        for i in range(half):
            rd=np.random.permutation(dim); rn=max(1,int(np.ceil(np.sin(np.pi/2*np.random.random())*dim)))
            t_b[i,rd[:rn]]=1
            rd=np.random.permutation(dim); rn=max(1,int(np.ceil(np.sin(np.pi/2*np.random.random())*dim)))
            k_b[i,rd[:rn]]=1
        # Race — cross-update
        for i in range(min(half,half)):
            t_new=np.where(t_b[i], p*t_pop[i,:-1]+(1-p)*k_pop[i,:-1], t_pop[i,:-1])
            k_new=np.where(k_b[i], p*k_pop[i,:-1]+(1-p)*t_pop[i,:-1], k_pop[i,:-1])
            t_new=np.clip(t_new,self._lo,self._hi); k_new=np.clip(k_new,self._lo,self._hi)
            ft=float(self.problem.evaluate(t_new)); fk=float(self.problem.evaluate(k_new)); evals+=2
            if self._is_better(ft,float(t_pop[i,-1])): t_pop[i,:-1]=t_new; t_pop[i,-1]=ft
            if self._is_better(fk,float(k_pop[i,-1])): k_pop[i,:-1]=k_new; k_pop[i,-1]=fk
        pop=np.vstack([t_pop,k_pop]); return pop, evals, {}
