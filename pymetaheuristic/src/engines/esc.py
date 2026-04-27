"""pyMetaheuristic src — Escape Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ESCEngine(PortedPopulationEngine):
    algorithm_id   = "esc"
    algorithm_name = "Escape Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s10462-024-11008-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        lf = 1 - t / max_iter
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r = np.random.random()
            if r < 0.33:
                # Escape from worst
                worst_idx = self._worst_index(pop[:, -1]); worst = pop[worst_idx, :-1]
                new_pos[i] = pop[i, :-1] + lf * np.random.random(d) * (pop[i, :-1] - worst)
            elif r < 0.67:
                # Move toward best
                new_pos[i] = pop[i, :-1] + lf * np.random.random(d) * (best_pos - pop[i, :-1])
            else:
                # Explore randomly
                new_pos[i] = lo + np.random.random(d) * (hi - lo)
            new_pos[i] = np.clip(new_pos[i], lo, hi)
        new_fits = self._evaluate_population(new_pos); evals += n
        mask = self._better_mask(new_fits, pop[:, -1])
        pop[mask] = np.hstack([new_pos, new_fits[:, None]])[mask]
        return pop, evals, {}
