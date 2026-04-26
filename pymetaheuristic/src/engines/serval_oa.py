"""pyMetaheuristic src — Serval Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SERVALOAEngine(PortedPopulationEngine):
    """Serval Optimization Algorithm — random-prey attack and chase-with-range-narrowing."""
    algorithm_id = "serval_oa"; algorithm_name = "Serval Optimization Algorithm"; family = "swarm"
    _REFERENCE     = {"doi": "10.3390/biomimetics7040204"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1; evals=0
        kk = np.random.randint(n)
        for i in range(n):
            r1 = np.random.randint(1,3,dim)
            pos = np.clip(pop[i,:-1]+np.random.random(dim)*(pop[kk,:-1]-r1*pop[i,:-1]), self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
            pos2 = np.clip(pop[i,:-1]+np.random.randint(1,3,dim)*self._span/t, self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals+=1
            if self._is_better(fit2, float(pop[i,-1])): pop[i,:-1]=pos2; pop[i,-1]=fit2
        return pop, evals, {}
