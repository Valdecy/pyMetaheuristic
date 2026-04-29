"""pyMetaheuristic src — Success History Intelligent Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class SHIOSuccessEngine(PortedPopulationEngine):
    """
    Success History Intelligent Optimizer.

    """

    algorithm_id = "shio_success"
    algorithm_name = "Success History Intelligent Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.cma.2024.117272",
        "title": "Success History Intelligent Optimizer",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        idx = self._order(pop[:, -1])[:3]
        hist = pop[idx, :-1].copy()
        if hist.shape[0] < 3:
            hist = np.vstack([hist, np.tile(hist[-1], (3 - hist.shape[0], 1))])
        return {"history_top3": hist}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        hist = np.asarray(state.payload.get("history_top3"), dtype=float).reshape(-1, dim)
        current = pop.copy()
        evals = 0

        a = max(0.0, 2.0 - 2.0 * float(state.step + 1) / float(T))
        sv = max(0.0, 1.5 - 1.5 * float(state.step + 1) / float(T))

        for i in range(n):
            xi = current[i, :-1]
            deltas = []
            for c in hist[:3]:
                r = np.random.rand(dim)
                deltas.append(c + (sv * 2.0 * r - a) * np.abs(r * c - xi))
            mean_delta = (deltas[0] + deltas[1] + deltas[2]) / 3.0
            candidate = xi + np.random.rand(dim) * (mean_delta - xi)
            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit

        idx = self._order(current[:, -1])[:3]
        hist = current[idx, :-1].copy()
        if hist.shape[0] < 3:
            hist = np.vstack([hist, np.tile(hist[-1], (3 - hist.shape[0], 1))])
        return current, evals, {"history_top3": hist}
