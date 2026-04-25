"""pyMetaheuristic src — Siberian Tiger Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class STOEngine(PortedPopulationEngine):
    """Siberian Tiger Optimization — prey hunting plus scaled range reduction."""
    algorithm_id = "sto"; algorithm_name = "Siberian Tiger Optimization"; family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1; evals=0
        for i in range(n):
            better=[j for j in range(n) if j!=i and self._is_better(float(pop[j,-1]),float(pop[i,-1]))]
            sf_pos = pop[better[np.random.randint(len(better))],:-1] if better else pop[self._best_index(pop[:,-1]),:-1]
            r1=np.random.randint(1,3)
            pos = np.clip(pop[i,:-1]+np.random.random()*(sf_pos-r1*pop[i,:-1]), self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
            pos2 = np.clip(pop[i,:-1]+np.random.random()*self._span/t, self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals+=1
            if self._is_better(fit2, float(pop[i,-1])): pop[i,:-1]=pos2; pop[i,-1]=fit2
        return pop, evals, {}
