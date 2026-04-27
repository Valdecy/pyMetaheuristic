"""pyMetaheuristic src — Superb Fairy-wren Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2); den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class SuperbFOAEngine(PortedPopulationEngine):
    algorithm_id   = "superb_foa"
    algorithm_name = "Superb Fairy-wren Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10586-024-04901-w"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        C=0.8; r1=np.random.random(); r2=np.random.random()
        w=(np.pi/2)*(t/max_iter)
        k=0.2*np.sin(np.pi/2-w)
        l=0.5*_levy(n,d,1.5)
        y_idx=np.random.randint(n)
        c1=np.random.random(); T=0.5; m=t/max_iter*2
        p=np.sin(hi-lo)*2+(hi-lo)*m
        best_idx=self._best_index(pop[:,-1]); Xb=pop[best_idx,:-1].copy()
        XG=Xb*C; new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            s=r1*20+r2*20
            if T<c1:
                new_pos[i]=pop[i,:-1]+lo+(hi-lo)*np.random.random(d)
            else:
                if s>20:
                    new_pos[i]=Xb+pop[i,:-1]*l[y_idx]*k
                else:
                    new_pos[i]=XG+(Xb-pop[i,:-1])*p
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
