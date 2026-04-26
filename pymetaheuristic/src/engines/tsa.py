"""pyMetaheuristic src — Tunicate Swarm Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TSAEngine(PortedPopulationEngine):
    """Tunicate Swarm Algorithm — coefficient-driven best-neighbourhood pursuit."""
    algorithm_id   = "tsa"
    algorithm_name = "Tunicate Swarm Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2020.103541"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        order   = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        pmin, pmax = 1, 4

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            c1 = np.random.random(dim)
            c2 = np.random.random(dim)
            c3 = np.random.random(dim)
            M  = float(int(pmin + np.random.random() * (pmax - pmin)))
            A  = (c2 + c3 - 2.0 * c1) / M
            t1 = best_pos + A * np.abs(best_pos - c2 * pop[i, :-1])
            t2 = best_pos - A * np.abs(best_pos - c2 * pop[i, :-1])
            pos = np.where(c3 >= 0.5, t1, t2)
            if i > 0:
                pos = (pos + pop[i-1, :-1]) / 2.0
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # full replacement
        return pop, n, {}
