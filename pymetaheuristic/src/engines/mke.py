"""pyMetaheuristic src — Monkey King Evolution Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class MKEEngine(PortedPopulationEngine):
    """Monkey King Evolution V1 — king-guided fluctuation and population-rate jumps."""
    algorithm_id = "mke"
    algorithm_name = "Monkey King Evolution V1"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, fluctuation_coeff=0.7, population_rate=0.3, c=3.0, fc=0.5)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        king = pop[self._best_index(pop[:, -1]), :-1]
        pr = float(self._params.get("population_rate", 0.3))
        fc = float(self._params.get("fluctuation_coeff", 0.7))
        c = float(self._params.get("c", 3.0))
        trials = []
        for i in range(n):
            if np.random.rand() < pr:
                y = pop[i, :-1] + np.random.rand(dim) * (king - pop[i, :-1]) + fc * np.random.normal(0, 1, dim) * self._span / max(c, 1e-12)
            else:
                peers = pop[self._rand_indices(n, i, 2), :-1]
                y = king + np.random.rand(dim) * (peers[0] - peers[1])
            trials.append(np.clip(y, self._lo, self._hi))
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        return pop, n, {}
