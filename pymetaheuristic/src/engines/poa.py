"""pyMetaheuristic src — Pelican Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class POAEngine(PortedPopulationEngine):
    """Pelican Optimization Algorithm — prey pursuit and water-surface winging phases."""
    algorithm_id   = "poa"
    algorithm_name = "Pelican Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.3390/s22030855"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        evals   = 0

        new_pop = pop.copy()
        for i in range(n):
            kk = np.random.randint(n)
            # Phase 1: Moving toward prey  (Eq. 4)
            if self._is_better(float(pop[kk, -1]), float(pop[i, -1])):
                pos = pop[i, :-1] + np.random.random() * (pop[kk, :-1] - np.random.randint(1, 3) * pop[i, :-1])
            else:
                pos = pop[i, :-1] + np.random.random() * (pop[i, :-1] - pop[kk, :-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(new_pop[i, -1])):
                new_pop[i, :-1] = pos; new_pop[i, -1] = fit

            # Phase 2: Winging on water surface  (Eq. 6)
            pos2 = new_pop[i, :-1] + 0.2 * (1.0 - t / T) * (2.0 * np.random.random(dim) - 1.0) * new_pop[i, :-1]
            pos2 = np.clip(pos2, self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals += 1
            if self._is_better(fit2, float(new_pop[i, -1])):
                new_pop[i, :-1] = pos2; new_pop[i, -1] = fit2

        return new_pop, evals, {}
