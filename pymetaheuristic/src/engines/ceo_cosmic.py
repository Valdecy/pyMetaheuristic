"""pyMetaheuristic src — Cosmic Evolution Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, safe_norm, weighted_mean_rows



class CEOCosmicEngine(PortedPopulationEngine):
    """Cosmic Evolution Optimization with exploration, attraction, collision, and resonance."""

    algorithm_id = "ceo_cosmic"
    algorithm_name = "Cosmic Evolution Optimization"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1007/s00521-025-11234-6",
        "title": "Cosmic Evolution Optimization",
        "year": 2025,
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
        center_count=5,
        alpha=0.25,
        base_collision_probability=0.20,
    )

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        progress = min(1.0, float(state.step + 1) / float(T))
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()

        C = max(2, min(n, int(self._params.get("center_count", 5))))
        centers = pop[order[:C], :-1]
        weights = np.linspace(C, 1, C)
        system_center = weighted_mean_rows(centers, weights)
        Rc = np.mean(np.linalg.norm(centers - system_center, axis=1)) + 1.0e-12

        Vep = (1.0 - 0.001 * (state.step + 1) / T) * np.exp(-4.0 * (state.step + 1) / T)
        alpha_t = float(self._params.get("alpha", 0.25)) * (1.0 + 0.0005 * (state.step + 1) / T)
        p_base = float(self._params.get("base_collision_probability", 0.20))
        Pglobal = p_base * (1.0 - 0.0005 * (state.step + 1) / T) * (1.0 - progress)
        Preson = 0.1 * np.exp(-3.0 * (state.step + 1) / T)

        current = pop.copy()
        evals = 0
        fit_best = float(current[order[0], -1])
        denom = abs(fit_best) + 1.0e-12

        for i in range(n):
            xi = current[i, :-1]
            exploration = Vep * np.random.randn(dim) * self._span

            force_terms = []
            force_weights = []
            for c in range(C):
                oc = centers[c]
                Uc = (oc - xi) / safe_norm(oc - xi)
                Fc = Uc * np.exp(-safe_norm(oc - xi) / Rc)
                wc = np.exp(-abs(float(current[i, -1]) - float(current[order[c], -1])) / denom)
                force_terms.append(Fc)
                force_weights.append(wc)
            force_terms = np.asarray(force_terms, dtype=float)
            force_weights = np.asarray(force_weights, dtype=float)
            force = weighted_mean_rows(force_terms, force_weights)

            align = alpha_t * (best - xi) * np.random.randn(dim)
            candidate = xi + exploration + force + align

            if np.random.rand() < Pglobal:
                candidate = candidate + np.random.randn(dim) * Rc
            if np.random.rand() < Preson:
                candidate = candidate + 0.01 * np.random.randn(dim) * self._span

            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
