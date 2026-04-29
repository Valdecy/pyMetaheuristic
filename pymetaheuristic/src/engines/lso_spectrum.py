"""pyMetaheuristic src — Light Spectrum Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, safe_norm


class LSOSpectrumEngine(PortedPopulationEngine):
    """
    Light Spectrum Optimizer.

    """

    algorithm_id = "lso_spectrum"
    algorithm_name = "Light Spectrum Optimizer"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.asoc.2024.112318",
        "title": "Light Spectrum Optimizer",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, q=0.5, ps=0.5, pe=0.5, beta=0.5)

    def _normalize(self, x):
        x = np.asarray(x, dtype=float)
        return x / safe_norm(x)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        progress = min(1.0, float(state.step + 1) / float(T))
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        worst_fit = float(pop[order[-1], -1])
        best_fit = float(pop[order[0], -1])
        mean_x = np.mean(pop[:, :-1], axis=0)

        q = float(self._params.get("q", 0.5))
        ps = float(self._params.get("ps", 0.5))
        pe = float(self._params.get("pe", 0.5))
        beta = float(self._params.get("beta", 0.5))

        current = pop.copy()
        evals = 0
        for i in range(n):
            xi = current[i, :-1]
            xr = current[np.random.randint(n), :-1]
            xp = current[np.random.choice(order[: max(2, n // 3)]), :-1]

            xnA = self._normalize(xr)
            xnB = self._normalize(xp)
            xnC = self._normalize(best)
            xL0 = self._normalize(mean_x)

            kr = np.random.uniform(0.2, 1.0)
            xL1 = (xL0 - xnA * np.dot(xnA, xL0)) / max(kr, 1.0e-12)
            xL2 = xL1 - 2.0 * xnB * np.dot(xL1, xnB)
            xL3 = kr * (xL2 - xnC * np.dot(xL2, xnC))

            a = np.random.rand() * (1.0 - progress)
            GI = np.clip(np.random.randn() * a, -1.0, 1.0)
            if np.random.rand() <= q:
                x1 = xi + GI * (xL1 - xL3) * (current[np.random.randint(n), :-1] - current[np.random.randint(n), :-1])
            else:
                x1 = xi + GI * (xL2 - xL3) * (current[np.random.randint(n), :-1] - current[np.random.randint(n), :-1])

            diff_norm = abs((float(current[i, -1]) - best_fit) / (abs(best_fit - worst_fit) + 1.0e-12))
            if np.random.rand() < ps or diff_norm < np.random.rand():
                if np.random.rand() < pe:
                    x2 = xi + np.random.rand(dim) * (current[np.random.randint(n), :-1] - current[np.random.randint(n), :-1]) + np.random.randn(dim) * (np.random.rand() < beta) * (best - xi)
                else:
                    x2 = 2.0 * np.cos(np.pi * np.random.rand()) * best * xi
            else:
                U = (np.random.rand(dim) < 0.5).astype(float)
                x2 = (xp + abs(np.random.randn(dim)) * (current[np.random.randint(n), :-1] - current[np.random.randint(n), :-1])) * U + (1.0 - U) * xi

            candidate = np.clip(0.5 * (x1 + x2), self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
