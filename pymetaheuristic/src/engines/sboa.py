"""pyMetaheuristic src — Secretary Bird Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class SBOAEngine(PortedPopulationEngine):
    algorithm_id   = "sboa"
    algorithm_name = "Secretary Bird Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-024-10729-y"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        CF=(1-t/max_iter)**(2*t/max_iter)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        # Phase 1
        for i in range(n):
            if t<max_iter/3:
                r1,r2=np.random.randint(n),np.random.randint(n)
                X1=np.clip(pop[i,:-1]+(pop[r1,:-1]-pop[r2,:-1])*np.random.random(d),lo,hi)
            elif t<2*max_iter/3:
                RB=np.random.randn(d)
                X1=np.clip(best_pos+np.exp((t/max_iter)**4)*(RB-0.5)*(best_pos-pop[i,:-1]),lo,hi)
            else:
                RL=0.5*_levy1d(d)
                X1=np.clip(best_pos+CF*pop[i,:-1]*RL,lo,hi)
            f1=float(self._evaluate_population(X1[None])[0]); evals+=1
            if self._is_better(f1,pop[i,-1]): pop[i]=np.append(X1,f1)
        # Phase 2
        r=np.random.random()
        Xrandom=pop[np.random.randint(n),:-1].copy()
        for i in range(n):
            if r<0.5:
                RB=np.random.random(d)
                X2=np.clip(best_pos+(1-t/max_iter)**2*(2*RB-1)*pop[i,:-1],lo,hi)
            else:
                K=round(1+np.random.random())
                X2=np.clip(pop[i,:-1]+np.random.random(d)*(Xrandom-K*pop[i,:-1]),lo,hi)
            f2=float(self._evaluate_population(X2[None])[0]); evals+=1
            if self._is_better(f2,pop[i,-1]): pop[i]=np.append(X2,f2)
        return pop, evals, {}
