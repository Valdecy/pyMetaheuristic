"""pyMetaheuristic src — Pareto Sequential Sampling Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class PSSEngine(PortedPopulationEngine):
    """Pareto Sequential Sampling — acceptance-rate-guided prominent vs full-domain sampling."""
    algorithm_id = "pss"; algorithm_name = "Pareto Sequential Sampling"; family = "math"
    _REFERENCE     = {"doi": "10.1007/s00500-021-05853-8"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, acceptance_rate=0.9, steps=0.0)

    def _initialize_payload(self, pop):
        return {"new_solution": False}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        ar   = float(self._params.get("acceptance_rate",0.9))
        steps= float(self._params.get("steps",0.0))
        ns   = bool(state.payload.get("new_solution",False))
        order=self._order(pop[:,-1]); best_pos=pop[order[0],:-1].copy()

        rand_vals=np.random.random((n,dim))
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            pos=pop[i,:-1].copy()
            for k in range(dim):
                if ns:
                    dev=abs(0.5*(1-ar)*(self._span[k]))*(1-t/T)
                else:
                    dev=abs(np.random.uniform(min(0,best_pos[k]),max(0,best_pos[k])))
                rlb=max(best_pos[k]-dev, self._lo[k]); rub=min(best_pos[k]+dev, self._hi[k])
                if rub<=rlb: rub=rlb+1e-10
                if np.random.random()<=ar:
                    pos[k]=rlb+rand_vals[i,k]*(rub-rlb)
                else:
                    pos[k]=self._lo[k]+rand_vals[i,k]*self._span[k]
            if steps>0: pos=np.round(pos/steps)*steps
            new_pos[i]=np.clip(pos,self._lo,self._hi)

        new_fit=self._evaluate_population(new_pos)
        pop=np.hstack([new_pos,new_fit[:,None]])
        cur_best=float(new_fit[self._best_index(new_fit)])
        old_best=float(pop[order[0],-1]) if len(order) else cur_best
        new_sol = self._is_better(cur_best, old_best)
        return pop, n, {"new_solution": new_sol}
