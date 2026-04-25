"""pyMetaheuristic src — Exponential-Trigonometric Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ETOEngine(PortedPopulationEngine):
    algorithm_id   = "eto"
    algorithm_name = "Exponential-Trigonometric Optimization"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2023.110148"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        a = 2 * np.exp(-4 * t / max_iter)
        for i in range(n):
            r = np.random.random(d)
            if np.random.random() < 0.5:
                # Exponential component
                C = 2 * r - 1
                X_new = best_pos - a * C * np.abs(best_pos - pop[i, :-1])
            else:
                # Trigonometric component
                theta = 2 * np.pi * r
                X_new = best_pos + a * np.sin(theta) * np.abs(best_pos - pop[i, :-1])
            X_new = np.clip(X_new, lo, hi)
            new_fit = float(self._evaluate_population(X_new[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(X_new, new_fit)
        return pop, evals, {}
