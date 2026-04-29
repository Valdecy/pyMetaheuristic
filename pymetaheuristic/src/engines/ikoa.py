"""pyMetaheuristic src — Improved Kepler Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight, safe_norm


class IKOAEngine(PortedPopulationEngine):
    """
    Improved Kepler Optimization Algorithm.

    """

    algorithm_id = "ikoa"
    algorithm_name = "Improved Kepler Optimization Algorithm"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.eswa.2025.128216",
        "title": "Improved Kepler Optimization Algorithm",
        "authors": "paper PDF bundle",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, mu0=2.0, gamma=3.0, branch_probability=0.6)

    def _initialize_payload(self, pop):
        return {"velocity": np.zeros((pop.shape[0], self.problem.dimension), dtype=float)}

    def _masses(self, fit):
        fit = np.asarray(fit, dtype=float)
        best = np.min(fit) if self.problem.objective == "min" else np.max(fit)
        worst = np.max(fit) if self.problem.objective == "min" else np.min(fit)
        denom = abs(best - worst) + 1.0e-12
        if self.problem.objective == "min":
            m = (worst - fit) / denom
        else:
            m = (fit - worst) / denom
        m = np.clip(m, 0.0, None) + 1.0e-12
        return m / np.sum(m)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        t = state.step + 1
        mu = float(self._params.get("mu0", 2.0)) * np.exp(-float(self._params.get("gamma", 3.0)) * t / T)
        velocities = np.asarray(state.payload.get("velocity", np.zeros((n, dim))), dtype=float)
        masses = self._masses(pop[:, -1])
        order = self._order(pop[:, -1])
        star = pop[order[0], :-1].copy()
        lprob = float(self._params.get("branch_probability", 0.6))

        current = pop.copy()
        evals = 0
        for i in range(n):
            xi = current[i, :-1]
            ids = self._rand_indices(n, i, 3)
            xa, xb, xc = current[ids[0], :-1], current[ids[1], :-1], current[ids[2], :-1]
            grav = masses[i] * mu * (star - xi) / (safe_norm(star - xi) ** 2 + 1.0e-12)
            peer = np.random.rand(dim) * (xa - xb)
            velocities[i] = np.random.rand(dim) * velocities[i] + grav + peer

            if np.random.rand() < lprob:
                candidate = xi + velocities[i] + np.random.rand(dim) * (star - xi)
            else:
                candidate = xi + velocities[i] + np.random.rand(dim) * (xa - xc)

            better_ab = xa if self._is_better(current[ids[0], -1], current[ids[1], -1]) else xb
            better_bc = xb if self._is_better(current[ids[1], -1], current[ids[2], -1]) else xc
            v1 = better_ab - xi
            v2 = better_bc - xi
            if np.random.rand() < 0.5:
                improve = star + np.random.rand() * levy_flight(dim, beta=1.5, scale=0.02) * self._span + (1.0 - np.random.rand()) * np.random.randn(dim) * 0.05 * self._span + 0.5 * (v1 + v2)
            else:
                improve = star + np.random.rand() * np.random.randn(dim) * 0.05 * self._span + (1.0 - np.random.rand()) * levy_flight(dim, beta=1.5, scale=0.02) * self._span + 0.5 * (v1 + v2)

            candidate = np.clip(candidate, self._lo, self._hi)
            improve = np.clip(improve, self._lo, self._hi)

            fit_c = float(self.problem.evaluate(candidate))
            fit_i = float(self.problem.evaluate(improve))
            evals += 2
            if self._is_better(fit_i, fit_c):
                candidate, fit_c = improve, fit_i
            if self._is_better(fit_c, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit_c
        return current, evals, {"velocity": velocities}
