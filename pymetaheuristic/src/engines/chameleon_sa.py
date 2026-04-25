"""pyMetaheuristic src — Chameleon Swarm Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ChameleonSAEngine(PortedPopulationEngine):
    algorithm_id   = "chameleon_sa"
    algorithm_name = "Chameleon Swarm Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.114685"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"v": 0.1*pop[:, :-1].copy(), "v0": np.zeros_like(pop[:, :-1]),
                "pbest": pop[:, :-1].copy(), "pfit": pop[:, -1].copy(),
                "gbest": pop[self._best_index(pop[:,-1]), :-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        v=state.payload["v"]; v0=state.payload["v0"]
        pbest=state.payload["pbest"]; pfit=state.payload["pfit"]; gbest=state.payload["gbest"]
        rho,alpha,beta_=1.0,4.0,3.0
        omega=(1-t/max_iter)**(rho*np.sqrt(t/max_iter))
        p1=2*np.exp(-2*(t/max_iter)**2)
        p2=2/(1+np.exp((-t+max_iter/2)/100))
        mu=2.0*np.exp(-(alpha*t/max_iter)**beta_)
        ch=np.random.randint(0,n,n)
        pos=pop[:, :-1].copy()
        for i in range(n):
            if np.random.random()>=0.1:
                pos[i]+=p1*(pbest[ch[i]]-pos[i])*np.random.random()+p2*(gbest-pos[i])*np.random.random()
            else:
                pos[i]=gbest+mu*((hi-lo)*np.random.random(d)+lo)*np.sign(np.random.random()-0.5)
        a_val=2590*(1-np.exp(-np.log(t+1)))
        v=omega*v+p1*(pbest-pos)*np.random.random()+p2*(gbest-pos)*np.random.random()
        pos=pos+(v**2-v0**2)/(2*max(a_val,1e-10))
        v0=v.copy()
        pos=np.clip(pos,lo,hi)
        new_fits=self._evaluate_population(pos); evals+=n
        mask=self._better_mask(new_fits,pfit)
        pbest[mask]=pos[mask]; pfit[mask]=new_fits[mask]
        bi=self._best_index(pfit)
        if self._is_better(pfit[bi],float(self._evaluate_population(gbest[None])[0])):
            gbest=pbest[bi].copy()
        pop=np.hstack([pos,new_fits[:,None]])
        state.payload.update({"v":v,"v0":v0,"pbest":pbest,"pfit":pfit,"gbest":gbest})
        return pop, evals, {}
