"""pyMetaheuristic src — Lyrebird Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class LOALyrebirdEngine(PortedPopulationEngine):
    """
    Lyrebird Optimization Algorithm (LOA).
    """

    algorithm_id = "loa_lyrebird"
    algorithm_name = "Lyrebird Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.cma.2023.116436",
        "title": "Lyrebird optimization algorithm: A new bio-inspired metaheuristic algorithm for solving optimization problems",
        "authors": "Seyedali Mirjalili et al.",
        "year": 2023,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("loa_lyrebird requires population_size >= 2.")

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        order = self._order(pop[:, -1])
        gbest = pop[order[0], :-1].copy()
        current = pop.copy()
        trials = np.zeros((n, dim), dtype=float)
        for i in range(n):
            rp = np.random.rand()
            better = [j for j in order if j != i and self._is_better(current[j, -1], current[i, -1])]
            if rp <= 0.5:
                target_idx = int(np.random.choice(better)) if better else int(order[0])
                target = current[target_idx, :-1]
                r = np.random.rand(dim)
                I = np.random.randint(1, 3, size=dim)
                trials[i] = current[i, :-1] + r * (target - I * current[i, :-1])
            else:
                r = np.random.rand(dim)
                trials[i] = current[i, :-1] + (1.0 - 2.0 * r) * (self._span / float(t))
            if not np.all(np.isfinite(trials[i])):
                trials[i] = gbest.copy()
        trial_pop = self._pop_from_positions(np.clip(trials, self._lo, self._hi))
        mask = self._better_mask(trial_pop[:, -1], current[:, -1])
        current[mask] = trial_pop[mask]
        return current, n, {}
