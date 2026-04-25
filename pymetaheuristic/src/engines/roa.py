"""pyMetaheuristic src — Remora Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ROAEngine(PortedPopulationEngine):
    algorithm_id   = "roa"
    algorithm_name = "Remora Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.115665"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        H=np.random.random((1,pop.shape[0]))>0.5
        return {"H":H,"prev":pop[:,:-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        H=state.payload["H"]; prev=state.payload["prev"]
        best_idx=self._best_index(pop[:,-1]); best_pos=pop[best_idx,:-1].copy()
        a1=2-t*(2/max_iter)
        for i in range(n):
            if not H[0,i]:
                a2=-1+t*(-1/max_iter)
                l=(a2-1)*np.random.random()+1
                dist=np.abs(best_pos-pop[i,:-1])
                pop[i,:-1]=np.clip(dist*np.exp(l)*np.cos(l*2*np.pi)+best_pos,lo,hi)
            else:
                ri=np.random.randint(n); X_rand=pop[ri,:-1]
                pop[i,:-1]=np.clip(best_pos-(np.random.random(d)*(best_pos+X_rand)/2-X_rand),lo,hi)
            # attempt
            attempt=np.clip(pop[i,:-1]+(pop[i,:-1]-prev[i])*np.random.randn(d),lo,hi)
            f_att=float(self._evaluate_population(attempt[None])[0]); evals+=1
            f_cur=float(self._evaluate_population(pop[i,:-1][None])[0]); evals+=1
            if self._is_better(f_att,f_cur):
                pop[i]=np.append(attempt,f_att); H[0,i]=(np.random.random()>0.5)
            else:
                A=2*a1*np.random.random()-a1; C=0.1
                pop[i,:-1]=np.clip(pop[i,:-1]-A*(pop[i,:-1]-C*best_pos),lo,hi)
                pop[i,-1]=f_cur
        state.payload.update({"H":H,"prev":pop[:,:-1].copy()})
        return pop, evals, {}
