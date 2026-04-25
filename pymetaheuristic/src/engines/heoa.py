"""pyMetaheuristic src — Human Evolutionary Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2); den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class HEOAEngine(PortedPopulationEngine):
    algorithm_id   = "heoa"
    algorithm_name = "Human Evolutionary Optimization Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2023.122638"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        LN=0.4; EN=0.4; FN_=0.1; A=0.6
        LNn=max(1,round(n*LN)); ENn=max(1,round(n*EN)); FNn=max(1,round(n*FN_))
        order=self._order(pop[:,-1]); pop=pop[order]
        GBestX=pop[0,:-1].copy(); jump_factor=np.ptp(hi-lo)/1000
        new_pos=pop[:,:-1].copy()
        R=np.random.random()
        if t<=(1/4)*max_iter:
            for j in range(n):
                new_pos[j]=GBestX*(1-t/max_iter)+(np.mean(pop[j,:-1])-GBestX)*np.floor(np.random.random()/jump_factor)*jump_factor+0.2*(1-t/max_iter)*(pop[j,:-1]-GBestX)*_levy1d(d)
        else:
            for j in range(LNn):
                if R<A:
                    new_pos[j]=0.2*np.cos(np.pi/2*(1-t/max_iter))*pop[j,:-1]*np.exp((-t*np.random.randn())/(np.random.random()*max_iter))
                else:
                    new_pos[j]=0.2*np.cos(np.pi/2*(1-t/max_iter))*pop[j,:-1]+np.random.randn(d)
            for j in range(LNn,LNn+ENn):
                new_pos[j]=np.random.randn(d)*np.exp((pop[-1,:-1]-pop[j,:-1])/(j+1)**2)
            for j in range(LNn+ENn,LNn+ENn+FNn):
                new_pos[j]=pop[j,:-1]+0.2*np.cos(np.pi/2*(1-t/max_iter))*np.random.random(d)*(GBestX-pop[j,:-1])
            for j in range(LNn+ENn+FNn,n):
                new_pos[j]=GBestX+(GBestX-pop[j,:-1])*np.random.randn()
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
