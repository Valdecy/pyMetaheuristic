"""pyMetaheuristic src — Chernobyl Disaster Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class CDOChornobylEngine(PortedPopulationEngine):
    """
    Chernobyl Disaster Optimizer.

    """

    algorithm_id = "cdo_chernobyl"
    algorithm_name = "Chernobyl Disaster Optimizer"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.compstruc.2023.107488",
        "title": "Chernobyl Disaster Optimizer (CDO)",
        "authors": "Mohammad Dehghani et al.",
        "year": 2023,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, radiation_floor=1.0, max_walk_speed=3.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        walk_speed = max(0.0, float(self._params.get("max_walk_speed", 3.0)) * (1.0 - float(state.step + 1) / float(T)))
        order = self._order(pop[:, -1])
        alpha = pop[order[0], :-1].copy()
        beta = pop[order[min(1, n - 1)], :-1].copy()
        gamma = pop[order[min(2, n - 1)], :-1].copy()

        trials = np.zeros((n, dim), dtype=float)
        for i in range(n):
            xi = pop[i, :-1]
            out = []
            for leader, scale, limit in ((alpha, 0.25, 16000), (beta, 0.50, 270000), (gamma, 1.00, 300000)):
                S = np.log(np.random.randint(1, limit + 1))
                q = np.random.rand(dim) * (S - walk_speed * np.random.rand(dim))
                A = scale * np.random.rand(dim) * 2.0
                D = A * leader - xi
                out.append(leader - q * D)
            trials[i] = (out[0] + out[1] + out[2]) / 3.0
        trial_pop = self._pop_from_positions(np.clip(trials, self._lo, self._hi))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        return pop, n, {}
