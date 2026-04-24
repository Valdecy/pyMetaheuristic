"""pyMetaheuristic src — Differential Evolution MTS Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class HDEEngine(PortedPopulationEngine):
    """Hybrid Differential Evolution with Multiple Trajectory Search local refinement."""
    algorithm_id = "hde"
    algorithm_name = "Differential Evolution MTS"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, differential_weight=0.5, crossover_probability=0.9, local_searches=5, search_range=0.1)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        trials = [de_trial(self, pop, i, float(self._params.get("differential_weight", 0.5)), float(self._params.get("crossover_probability", 0.9))) for i in range(n)]
        trial_pop = self._pop_from_positions(np.asarray(trials)); evals = n
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1]); pop[mask] = trial_pop[mask]
        b = self._best_index(pop[:, -1]); x = pop[b, :-1].copy(); fx = float(pop[b, -1])
        rng = float(self._params.get("search_range", 0.1)) * self._span
        for _ in range(int(self._params.get("local_searches", 5))):
            improved = False
            for d in range(dim):
                for s in (1.0, -1.0):
                    y = x.copy(); y[d] = np.clip(y[d] + s * rng[d], self._lo[d], self._hi[d])
                    fy = float(self.problem.evaluate(y)); evals += 1
                    if self._is_better(fy, fx):
                        x, fx, improved = y, fy, True
            if not improved: rng *= 0.5
        pop[b, :-1], pop[b, -1] = x, fx
        return pop, evals, {}
