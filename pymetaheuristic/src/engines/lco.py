"""pyMetaheuristic src — Life Choice-Based Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class LCOEngine(PortedPopulationEngine):
    """Life Choice-Based Optimizer — n-best mean, gradient steps and boundary reflection."""
    algorithm_id   = "lco"
    algorithm_name = "Life Choice-Based Optimizer"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s00500-019-04443-z"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, r1=2.35)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        r1      = float(self._params.get("r1", 2.35))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        n_agents = max(2, int(np.ceil(np.sqrt(n))))   # sqrt-size n-best pool

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            prob = np.random.random()
            if prob > 0.875:          # n-best mean  (Eq. 1)
                top_n = pop[order[:n_agents], :-1]
                temp  = np.mean(np.random.random() * top_n, axis=0)
            elif prob < 0.7:          # gradient step  (Eqs. 2-6)
                f1    = 1.0 - t / T
                f2    = 1.0 - f1
                prev  = best_pos if i == 0 else pop[order[i - 1], :-1]
                bd    = f1 * r1 * (best_pos - pop[i, :-1])
                brd   = f2 * r1 * (prev     - pop[i, :-1])
                temp  = pop[i, :-1] + np.random.random() * brd + np.random.random() * bd
            else:                     # boundary reflection
                temp  = self._hi - (pop[i, :-1] - self._lo) * np.random.random()
            new_pos[i] = np.clip(temp, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
