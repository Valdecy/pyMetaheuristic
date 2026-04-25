"""pyMetaheuristic src — Tasmanian Devil Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TDOEngine(PortedPopulationEngine):
    """Tasmanian Devil Optimization — carrion/prey feeding with neighbourhood-radius chase."""
    algorithm_id = "tdo"; algorithm_name = "Tasmanian Devil Optimization"; family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1; evals=0
        for i in range(n):
            k = np.random.choice([x for x in range(n) if x!=i])
            if self._is_better(float(pop[k,-1]),float(pop[i,-1])):
                pos = np.clip(pop[i,:-1]+np.random.random(dim)*(pop[k,:-1]-np.random.randint(1,3)*pop[i,:-1]), self._lo, self._hi)
            else:
                pos = np.clip(pop[i,:-1]+np.random.random(dim)*(pop[i,:-1]-pop[k,:-1]), self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
            rr = 0.01*(1-t/T)
            pos2 = np.clip(pop[i,:-1]+(-rr+2*rr*np.random.random(dim))*pop[i,:-1], self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals+=1
            if self._is_better(fit2, float(pop[i,-1])): pop[i,:-1]=pos2; pop[i,-1]=fit2
        return pop, evals, {}
