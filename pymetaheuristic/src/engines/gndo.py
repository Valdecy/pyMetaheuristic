"""pyMetaheuristic src — Generalized Normal Distribution Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GNDOEngine(PortedPopulationEngine):
    algorithm_id   = "gndo"
    algorithm_name = "Generalized Normal Distribution Optimizer"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.enconman.2020.113301"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        best_pos = pop[self._best_index(pop[:,-1]), :-1].copy()
        mo = np.mean(pop[:,:-1], axis=0)
        for i in range(n):
            idxs = [j for j in range(n) if j!=i]
            a,b,c_ = np.random.choice(idxs, 3, replace=False)
            v1 = (pop[a,:-1]-pop[i,:-1]) if pop[a,-1]<pop[i,-1] else (pop[i,:-1]-pop[a,:-1])
            v2 = (pop[b,:-1]-pop[c_,:-1]) if pop[b,-1]<pop[c_,-1] else (pop[c_,:-1]-pop[b,:-1])
            if np.random.random()<=np.random.random():
                u=(pop[i,:-1]+best_pos+mo)/3
                deta=np.sqrt((1/3)*((pop[i,:-1]-u)**2+(best_pos-u)**2+(mo-u)**2))
                vc1=np.random.random(d); vc2=np.random.random(d)
                Z1=np.sqrt(-np.log(np.clip(vc2,1e-300,None)))*np.cos(2*np.pi*vc1)
                Z2=np.sqrt(-np.log(np.clip(vc2,1e-300,None)))*np.cos(2*np.pi*vc1+np.pi)
                if np.random.random()<=np.random.random():
                    newsol=u+deta*Z1
                else:
                    newsol=u+deta*Z2
            else:
                beta=np.random.random()
                newsol=pop[i,:-1]+beta*np.abs(np.random.randn(d))*v1+(1-beta)*np.abs(np.random.randn(d))*v2
            newsol=np.clip(newsol,lo,hi)
            new_fit=float(self._evaluate_population(newsol[None])[0]); evals+=1
            if self._is_better(new_fit, pop[i,-1]):
                pop[i]=np.append(newsol,new_fit)
                if self._is_better(new_fit, pop[self._best_index(pop[:,-1]),-1]):
                    best_pos=newsol.copy()
        return pop, evals, {}
