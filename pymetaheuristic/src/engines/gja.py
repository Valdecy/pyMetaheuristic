"""pyMetaheuristic src — Gekko Japonicus Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight, safe_norm


class GJAEngine(PortedPopulationEngine):
    """
    Gekko Japonicus Algorithm.

 
    """

    algorithm_id = "gja"
    algorithm_name = "Gekko Japonicus Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.eswa.2025.127982",
        "title": "Gekko Japonicus Algorithm",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, beta_start=1.9, beta_end=1.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(2, self.config.max_steps or 100)
        t = state.step + 1
        beta_s = float(self._params.get("beta_start", 1.9))
        beta_e = float(self._params.get("beta_end", 1.0))
        beta = beta_s - (beta_s - beta_e) * (np.cosh((t / T) * 5.0) - 1.0) / (np.cosh(5.0) - 1.0)
        alpha = 0.6 * beta
        dy = (1.0 - (t - 1) / (T - 1)) ** 2
        best = pop[self._best_index(pop[:, -1]), :-1].copy()

        current = pop.copy()
        evals = 0
        for i in range(n):
            xi = current[i, :-1]
            if np.random.rand() < 0.5:
                step = alpha * levy_flight(dim, beta=max(1.05, min(beta, 1.95)), scale=0.02) * self._span
            else:
                step = alpha * np.random.randn(dim) * self._span / np.sqrt(float(t))
            candidate = xi + step

            df = np.sign(best - candidate)
            candidate = candidate + 0.3 * np.random.rand(dim) * df * self._span

            s = max(1, int(np.floor((np.random.rand() ** 2) * n)))
            group = current[np.random.choice(n, size=s, replace=False), :-1]
            group_best = group[np.argmin([self.problem.evaluate(g.tolist()) for g in group]) if self.problem.objective == "min" else np.argmax([self.problem.evaluate(g.tolist()) for g in group])]
            candidate = candidate + 0.5 * dy * (group_best - candidate)

            if safe_norm(candidate - best) > 0.1 * safe_norm(self._span):
                mb = 1.0 - dy
                mask = np.random.rand(dim) < mb
                candidate[mask] = best[mask] + np.random.randn(np.sum(mask)) * 0.1 * self._span[mask]

            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit
        return current, evals, {}
