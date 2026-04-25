"""pyMetaheuristic src — Social Spider Swarm Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SSOEngine(PortedPopulationEngine):
    """Social Spider Swarm Optimizer — two-group cosine/averaging update inspired by spiders."""
    algorithm_id = "sso"; algorithm_name = "Social Spider Swarm Optimizer"; family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        c1 = 2*np.exp(-((4*t/T)**2))
        order = self._order(pop[:,-1]); best_pos = pop[order[0],:-1].copy()
        new_pos = np.empty_like(pop[:,:-1])
        for i in range(n):
            if i < n//2:
                c2 = np.random.random(dim); c3 = np.random.random(dim)
                p1 = best_pos + c1*(self._span*c2+self._lo)
                p2 = best_pos - c1*(self._span*c2+self._lo)
                new_pos[i] = np.where(c3<0.5, p1, p2)
            else:
                new_pos[i] = (pop[i,:-1]+pop[i-1,:-1])/2.0
            new_pos[i] = np.clip(new_pos[i], self._lo, self._hi)
        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:,None]])
        mask = self._better_mask(new_fit, pop[:,-1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
