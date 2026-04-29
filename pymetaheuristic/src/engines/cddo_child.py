"""pyMetaheuristic src — Child Drawing Development Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class CDDOChildEngine(PortedPopulationEngine):
    """
    Child Drawing Development Optimization.

  
    """

    algorithm_id = "cddo_child"
    algorithm_name = "Child Drawing Development Optimization Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1016/j.knosys.2024.111558",
        "title": "Child Drawing Development Optimization Algorithm",
        "authors": "paper PDF bundle",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, pattern_memory_size=5, creativity_rate=0.1, local_group_size=4)

    def _initialize_payload(self, pop):
        m = min(int(self._params.get("pattern_memory_size", 5)), pop.shape[0])
        keep = self._order(pop[:, -1])[:m]
        return {"pattern_memory": pop[keep, :-1].copy()}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        pm = np.asarray(state.payload.get("pattern_memory", pop[self._order(pop[:, -1])[:5], :-1]), dtype=float).reshape(-1, dim)
        gbest = pop[self._best_index(pop[:, -1]), :-1].copy()
        current = pop.copy()
        evals = 0
        cr = float(self._params.get("creativity_rate", 0.1))
        local_group_size = max(2, int(self._params.get("local_group_size", 4)))

        for i in range(n):
            xi = current[i, :-1]
            j = int(np.random.randint(dim))
            hp = float(xi[j])
            rhp = float(np.random.uniform(self._lo[j], self._hi[j]))

            local_ids = self._rand_indices(n, i, min(local_group_size, n - 1))
            local_rows = np.vstack((current[i:i+1], current[local_ids]))
            local_best = local_rows[self._best_index(local_rows[:, -1]), :-1]

            L = int(np.random.randint(dim))
            W = int(np.random.randint(dim))
            gr = (xi[L] + xi[W]) / (abs(xi[L]) + 1.0e-12)
            sr = np.random.rand(dim)
            lr = np.random.rand(dim)

            if abs(rhp) < abs(hp):
                candidate = gr + sr * (local_best - xi) + lr * (gbest - xi)
            elif abs(abs(rhp) - abs(hp)) <= 0.1 * max(1.0, abs(hp)):
                pm_idx = int(np.random.randint(pm.shape[0]))
                candidate = pm[pm_idx] + cr * gbest
            else:
                candidate = xi + np.random.normal(0.0, 0.05, dim) * self._span

            candidate = np.clip(candidate, self._lo, self._hi)
            fit = float(self.problem.evaluate(candidate))
            evals += 1
            if self._is_better(fit, current[i, -1]):
                current[i, :-1] = candidate
                current[i, -1] = fit

        keep = self._order(current[:, -1])[: min(pm.shape[0], current.shape[0])]
        pm = current[keep, :-1].copy()
        return current, evals, {"pattern_memory": pm}
