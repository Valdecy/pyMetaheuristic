"""pyMetaheuristic src — Dolphin Echolocation Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class DEODolphinEngine(PortedPopulationEngine):
    """
    Dolphin Echolocation Optimization.

    """

    algorithm_id = "deo_dolphin"
    algorithm_name = "Dolphin Echolocation Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.advengsoft.2016.05.002",
        "title": "Dolphin Echolocation Optimization",
        "authors": "Kaveh and Farhoudi",
        "year": 2016,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, search_radius=0.25, convergence_power=2.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        progress = min(1.0, float(state.step + 1) / float(T))
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        current = pop.copy()
        evals = 0
        radius = float(self._params.get("search_radius", 0.25)) * (1.0 - progress)
        power = float(self._params.get("convergence_power", 2.0))

        ranks = np.empty(n, dtype=float)
        ranks[order] = np.linspace(1.0, 0.0, n)
        probs = ranks ** max(power, 1.0)

        for i in range(n):
            xi = current[i, :-1]
            ref_idx = int(np.random.choice(n, p=(probs / (np.sum(probs) + 1e-12))))
            ref = current[ref_idx, :-1]
            gate = np.random.rand(dim) < np.clip(probs[ref_idx] + progress * 0.25, 0.05, 0.95)
            candidate = xi.copy()
            move = ref + np.random.uniform(-radius, radius, dim) * self._span
            candidate[gate] = move[gate]
            candidate[~gate] = xi[~gate] + np.random.randn(np.sum(~gate)) * radius * 0.5 * self._span[~gate]
            candidate = 0.5 * candidate + 0.5 * best
            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
