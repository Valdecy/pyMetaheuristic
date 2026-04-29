"""pyMetaheuristic src — Atom Search Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class ASOAtomEngine(PortedPopulationEngine):
    """
    Atom Search Optimization (ASO).

    """

    algorithm_id = "aso_atom"
    algorithm_name = "Atom Search Optimization"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.knosys.2018.08.030",
        "title": "Atom Search Optimization: a metaheuristic algorithm for global optimization",
        "authors": "Zhao et al.",
        "year": 2019,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, alpha=50.0, beta=0.2)

    def _initialize_payload(self, pop):
        return {"velocity": np.zeros((pop.shape[0], self.problem.dimension), dtype=float)}

    def _masses(self, fit):
        fit = np.asarray(fit, dtype=float)
        best = np.min(fit) if self.problem.objective == "min" else np.max(fit)
        worst = np.max(fit) if self.problem.objective == "min" else np.min(fit)
        denom = abs(best - worst) + 1.0e-12
        if self.problem.objective == "min":
            m = np.exp(-(fit - best) / denom)
        else:
            m = np.exp(-(best - fit) / denom)
        return m / (np.sum(m) + 1.0e-12)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        t = state.step + 1
        velocity = np.asarray(state.payload.get("velocity", np.zeros((n, dim))), dtype=float)
        masses = self._masses(pop[:, -1])
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        K = max(2, int(n - (n - 2) * (t / T) ** 0.5))
        beta = float(self._params.get("beta", 0.2))
        alpha = float(self._params.get("alpha", 50.0)) * np.exp(-20.0 * t / T)

        new_pop = pop.copy()
        for i in range(n):
            xi = pop[i, :-1]
            force = np.zeros(dim, dtype=float)
            for j in order[:K]:
                if j == i:
                    continue
                xj = pop[j, :-1]
                dist = np.linalg.norm(xj - xi) + 1.0e-12
                h = dist / (np.mean(self._span) + 1.0e-12)
                if h < beta:
                    potential = alpha * (12.0 * (-h) ** -13 - 6.0 * (-h) ** -7) if h > 1.0e-6 else alpha
                else:
                    potential = -alpha / (h ** 2 + 1.0e-12)
                direction = (xj - xi) / dist
                force += np.random.rand(dim) * potential * direction * masses[j]
            acc = force / (masses[i] + 1.0e-12) + np.random.rand(dim) * (best - xi)
            velocity[i] = np.random.rand(dim) * velocity[i] + acc
            new_pop[i, :-1] = np.clip(xi + velocity[i], self._lo, self._hi)
        fit = self._evaluate_population(new_pop[:, :-1])
        new_pop[:, -1] = fit
        return new_pop, n, {"velocity": velocity}
