"""pyMetaheuristic src — Walrus Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WAOAEngine(PortedPopulationEngine):
    """Walrus Optimization Algorithm — feeding exploration and range-narrowing exploitation."""
    algorithm_id = "waoa"; algorithm_name = "Walrus Optimization Algorithm"; family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1; evals=0
        for i in range(n):
            kk = np.random.randint(n)
            if self._is_better(float(pop[kk,-1]),float(pop[i,-1])):
                pos = np.clip(pop[i,:-1]+np.random.random()*(pop[kk,:-1]-np.random.randint(1,3)*pop[i,:-1]), self._lo, self._hi)
            else:
                pos = np.clip(pop[i,:-1]+np.random.random()*(pop[i,:-1]-pop[kk,:-1]), self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
            LB = self._lo/t; UB = self._hi/t
            pos2 = np.clip(pop[i,:-1]+LB+(UB-np.random.random()*LB), self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals+=1
            if self._is_better(fit2, float(pop[i,-1])): pop[i,:-1]=pos2; pop[i,-1]=fit2
        return pop, evals, {}
