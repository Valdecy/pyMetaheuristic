"""pyMetaheuristic src — Polar Lights Optimizer Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2); den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class PLOEngine(PortedPopulationEngine):
    algorithm_id   = "plo"
    algorithm_name = "Polar Lights Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.neucom.2024.128427"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"V": np.ones_like(pop[:,:-1])}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        V=state.payload["V"]
        order=self._order(pop[:,-1]); pop=pop[order]; V=V[order]
        best_pos=pop[0,:-1].copy()
        X_mean=np.mean(pop[:,:-1],axis=0)
        w1=np.tanh((t/max_iter)**4); w2=np.exp(-(2*t/max_iter)**3)
        X_new=np.zeros_like(pop[:,:-1])
        for i in range(n):
            a=np.random.random()/2+1
            V[i]=np.exp((1-a)/100*t)
            LS=V[i]*np.ones(d)
            GS=_levy1d(d)*(X_mean-pop[i,:-1]+(lo+np.random.random(d)*(hi-lo))/2)
            X_new[i]=pop[i,:-1]+(w1*LS+w2*GS)*np.random.random(d)
        E=np.sqrt(t/max_iter); A=np.random.permutation(n)
        for i in range(n):
            for j in range(d):
                if np.random.random()<0.05 and np.random.random()<E:
                    X_new[i,j]=pop[i,j]+np.sin(np.random.random()*np.pi)*(pop[i,j]-pop[A[i],j])
            X_new[i]=np.clip(X_new[i],lo,hi)
        new_fits=self._evaluate_population(X_new); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([X_new,new_fits[:,None]])[mask]
        state.payload["V"]=V
        return pop, evals, {}
