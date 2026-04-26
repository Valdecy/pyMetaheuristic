"""pyMetaheuristic src — Dandelion Optimizer Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class DandelionOEngine(PortedPopulationEngine):
    algorithm_id   = "do_dandelion"
    algorithm_name = "Dandelion Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2022.105075"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        beta_arr=np.random.randn(n,d)
        alpha=np.random.random()*((1/max_iter**2)*t**2-2/max_iter*t+1)
        a=1/(max_iter**2-2*max_iter+1); b=-2*a; c_coef=1-a-b
        k=1-np.random.random()*(c_coef+a*t**2+b*t)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        # Phase 1
        if np.random.randn()<1.5:
            for i in range(n):
                lamb=np.abs(np.random.randn(d)); theta=(2*np.random.random()-1)*np.pi
                row=1/np.exp(theta); vx=row*np.cos(theta); vy=row*np.sin(theta)
                NEW=np.random.uniform(lo,hi)
                pop[i,:-1]+=alpha*vx*vy*norm.logpdf(lamb)*(NEW-pop[i,:-1])
        else:
            pop[:,:-1]*=k
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        # Phase 2
        mean=np.mean(pop[:,:-1],axis=0)
        for i in range(n):
            for j in range(d):
                pop[i,j]=pop[i,j]-beta_arr[i,j]*alpha*(mean[j]-beta_arr[i,j]*alpha*pop[i,j])
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        # Phase 3
        Step=_levy(n,d,1.5); Elite=np.tile(best_pos,(n,1))
        for i in range(n):
            for j in range(d):
                pop[i,j]=Elite[i,j]+Step[i,j]*alpha*(Elite[i,j]-pop[i,j]*(2*t/max_iter))
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=new_fits
        return pop, evals, {}
