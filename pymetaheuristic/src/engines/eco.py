"""pyMetaheuristic src — Educational Competition Optimizer Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class ECOEngine(PortedPopulationEngine):
    algorithm_id   = "eco"
    algorithm_name = "Educational Competition Optimizer"
    family         = "human"
    _REFERENCE     = {"doi": "10.3390/biomimetics10030176"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        order=self._order(pop[:,-1]); pop=pop[order]
        GBestX=pop[0,:-1].copy(); GWorstX=pop[-1,:-1].copy()
        R1=np.random.random(); R2=np.random.random()
        P=4*np.random.randn()*(1-t/max_iter)
        E=(np.pi*t)/(max(P,0.01)*max_iter)
        w=0.1*np.log(2-t/max_iter)
        new_pos=pop[:,:-1].copy()
        for j in range(n):
            St=np.random.randint(1,5)
            if St==1:
                new_pos[j]=GBestX+R1*np.exp(w)*(pop[j,:-1]-GBestX)+R2*np.exp(w)*(GWorstX-pop[j,:-1])
            elif St==2:
                new_pos[j]=pop[j,:-1]+R1*np.sin(E)*(GBestX-pop[j,:-1])+R2*np.cos(E)*(GBestX-pop[j,:-1])
            elif St==3:
                new_pos[j]=GBestX*(1-R1)+R2*(pop[j,:-1]-GBestX)
            else:
                new_pos[j]=pop[j,:-1]+_levy1d(d)*(GBestX-pop[j,:-1])
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
