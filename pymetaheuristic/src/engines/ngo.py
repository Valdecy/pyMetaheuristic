"""pyMetaheuristic src — Northern Goshawk Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class NGOEngine(PortedPopulationEngine):
    """Northern Goshawk Optimization — two-phase attack and pursuit hunting strategy."""
    algorithm_id   = "ngo"
    algorithm_name = "Northern Goshawk Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2021.3133286"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        evals  = 0

        new_pop = pop.copy()
        for i in range(n):
            # Phase 1 — Exploration: attack prey  (Eq. 4)
            k   = np.random.randint(n)
            prey_better = self._is_better(float(pop[k, -1]), float(pop[i, -1]))
            if prey_better:
                pos = pop[i, :-1] + np.random.random(dim) * (
                    pop[k, :-1] - np.random.randint(1, 3) * pop[i, :-1])
            else:
                pos = pop[i, :-1] + np.random.random(dim) * (
                    pop[i, :-1] - pop[k, :-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(new_pop[i, -1])):
                new_pop[i, :-1] = pos
                new_pop[i,  -1] = fit

            # Phase 2 — Exploitation: pursuit  (Eq. 7)
            R   = 0.02 * (1.0 - t / T)
            pos = new_pop[i, :-1] + (-R + 2.0 * R * np.random.random(dim)) * new_pop[i, :-1]
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(new_pop[i, -1])):
                new_pop[i, :-1] = pos
                new_pop[i,  -1] = fit

        return new_pop, evals, {}
