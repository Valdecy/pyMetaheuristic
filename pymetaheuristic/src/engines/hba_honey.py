"""pyMetaheuristic src — Honey Badger Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, safe_norm 


class HBAHoneyEngine(PortedPopulationEngine):
    """
    Honey Badger Algorithm (HBA).

    """

    algorithm_id = "hba_honey"
    algorithm_name = "Honey Badger Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.matcom.2021.08.013",
        "title": "Honey badger algorithm: New metaheuristic algorithm for solving optimization problems",
        "authors": "Hashim et al.",
        "year": 2022,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, beta=6.0, density_constant=2.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        alpha = float(self._params.get("density_constant", 2.0)) * np.exp(-float(state.step + 1) / float(T))
        beta = float(self._params.get("beta", 6.0))
        best = pop[self._best_index(pop[:, -1]), :-1].copy()
        current = pop.copy()
        evals = 0

        positions = current[:, :-1]
        shift = np.roll(positions, -1, axis=0)
        S = np.sum((positions - shift) ** 2, axis=1) + 1.0e-12

        for i in range(n):
            xi = current[i, :-1]
            d = best - xi
            intensity = np.random.rand() * S[i] / (4.0 * np.pi * (safe_norm(d) ** 2))
            F = 1.0 if np.random.rand() < 0.5 else -1.0
            if np.random.rand() < 0.5:
                r3, r4, r5 = np.random.rand(), np.random.rand(), np.random.rand()
                osc = abs(np.cos(2.0 * np.pi * r4) * (1.0 - np.cos(2.0 * np.pi * r5)))
                candidate = best + F * beta * intensity * best + F * r3 * alpha * d * osc
            else:
                r7 = np.random.rand(dim)
                candidate = best + F * r7 * alpha * d
            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
