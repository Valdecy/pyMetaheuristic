"""pyMetaheuristic src — Flood Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FloodAEngine(PortedPopulationEngine):
    algorithm_id   = "flood_a"
    algorithm_name = "Flood Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1007/s11227-024-06054-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, Ne=5)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        Ne=int(self._params.get("Ne",5))
        best_idx=self._best_index(pop[:,-1]); BestSolPos=pop[best_idx,:-1].copy(); BestSolFit=pop[best_idx,-1]
        Val=np.zeros_like(pop[:,:-1])
        PK=((((max_iter*(t**2)+1)**0.5+(1/((max_iter/4)*max(t,1)))*np.log((max_iter*(t**2)+1)**0.5+(max_iter/4)*t))**(-2/3))*(1.2/max(t,1)))
        for i in range(n):
            sorted_fits=np.sort(pop[:,-1])
            Pe_i=((pop[i,-1]-sorted_fits[0])/(sorted_fits[-1]-sorted_fits[0]+1e-300))**2
            others=[k for k in range(n) if k!=i]
            a=others[np.random.randint(len(others))]
            if np.random.random()>(np.random.random()+Pe_i):
                Val[i]=((PK**np.random.randn())/max(t,1))*(np.random.uniform(lo,hi))
                new_pos=np.clip(pop[i,:-1]+Val[i],lo,hi)
            else:
                new_pos=np.clip(BestSolPos+np.random.random(d)*(pop[a,:-1]-pop[i,:-1]),lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(new_pos,new_fit)
            if self._is_better(pop[i,-1],BestSolFit):
                BestSolFit=pop[i,-1]; BestSolPos=pop[i,:-1].copy()
        Pt=abs(np.sin(np.random.random()/max(t,1)))
        if np.random.random()<Pt and n>Ne:
            order=self._order(pop[:,-1])
            keep=order[:n-Ne]
            pop=pop[keep]
            for _ in range(Ne):
                new_pos=np.clip(BestSolPos+np.random.random()*(np.random.uniform(lo,hi)),lo,hi)
                new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
                pop=np.vstack([pop,np.append(new_pos,new_fit)])
        return pop, evals, {}
