"""pyMetaheuristic src — Market Game Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class MGOAMarketEngine(PortedPopulationEngine):
    """Market Game Optimization Algorithm with attraction and collaboration moves."""

    algorithm_id = "mgoa_market"
    algorithm_name = "Market Game Optimization Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1016/j.asoc.2024.112466",
        "title": "Market Game Optimization Algorithm: a strategy-inspired optimizer",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        strategy_amplitude=np.pi,
        frequency=2.0,
        decay=2.0,
        collaboration_rate=0.5,
    )

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        evals = 0
        T = max(1, self.config.max_evaluations or self.config.max_steps or 100)
        fes = max(1, state.evaluations)
        progress = min(1.0, float(fes) / float(T))
        A = float(self._params.get("strategy_amplitude", np.pi))
        freq = float(self._params.get("frequency", 2.0))
        decay = float(self._params.get("decay", 2.0))
        S = A * np.sin(2.0 * np.pi * progress * freq) * np.exp(-decay * progress)
        best = pop[self._best_index(pop[:, -1]), :-1].copy()
        current = pop.copy()

        for i in range(n):
            xi = current[i, :-1]
            k = max(1, int(np.random.randint(1, dim + 1)))
            dims = np.random.choice(dim, size=k, replace=False)
            attract = xi.copy()
            attract[dims] = xi[dims] - S * (best[dims] - xi[dims])

            ids = self._rand_indices(n, i, 3)
            a, b, c = current[ids[0], :-1], current[ids[1], :-1], current[ids[2], :-1]
            r1, r2 = np.random.rand(), np.random.rand()
            collab = xi + r1 * (a - b) + r2 * (c - xi)

            candidate = 0.5 * (attract + collab) if np.random.rand() < float(self._params.get("collaboration_rate", 0.5)) else attract
            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
