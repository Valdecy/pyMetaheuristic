"""pyMetaheuristic src — Archerfish Hunting Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight, safe_norm


class AHOEngine(PortedPopulationEngine):
    """Archerfish Hunting Optimizer with shooting/jumping phases and stagnation rescue."""

    algorithm_id = "aho"
    algorithm_name = "Archerfish Hunting Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.engappai.2024.108081",
        "title": "Archerfish Hunting Optimizer",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, omega=0.10, levy_beta=1.5, stagnation_generations=5)

    def _initialize_payload(self, pop):
        return {"stagnation": 0}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        current = pop.copy()
        evals = 0
        improved_any = False
        omega_scale = float(self._params.get("omega", 0.10)) * np.mean(self._span)
        best_idx = self._best_index(current[:, -1])

        for i in range(n):
            xi = current[i, :-1]
            k = self._rand_indices(n, i, 1)[0]
            xk = current[k, :-1]
            alpha = np.random.rand()
            b = 1 if np.random.rand() < 0.5 else 0
            theta0 = ((-1) ** b) * alpha * np.pi
            prey = xk.copy()
            mod_dims = np.random.choice(dim, size=(2 if abs(np.sin(theta0)) > 0.5 and dim > 1 else 1), replace=False)
            prey[mod_dims[0]] += omega_scale * np.sin(2.0 * theta0)
            if len(mod_dims) > 1:
                prey[mod_dims[1]] += omega_scale * (np.sin(theta0) ** 2)
            prey += np.random.normal(0.0, 0.01, dim) * self._span
            attraction = np.exp(-(safe_norm(prey - xi) ** 2) / (np.mean(self._span) ** 2 + 1.0e-12))
            candidate = xi + attraction * (prey - xi)
            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
                improved_any = True

        stagnation = int(state.payload.get("stagnation", 0))
        if not improved_any:
            stagnation += 1
        else:
            stagnation = 0

        if stagnation >= int(self._params.get("stagnation_generations", 5)):
            half = max(1, n // 2)
            idx = np.random.choice(n, size=half, replace=False)
            for i in idx:
                candidate = current[i, :-1] + levy_flight(dim, beta=float(self._params.get("levy_beta", 1.5)), scale=0.02) * self._span
                candidate = np.clip(candidate, self._lo, self._hi)
                fit = float(self.problem.evaluate(candidate))
                evals += 1
                if self._is_better(fit, current[i, -1]):
                    current[i, :-1] = candidate
                    current[i, -1] = fit
            stagnation = 0

        return current, evals, {"stagnation": stagnation}
