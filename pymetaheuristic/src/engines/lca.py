"""pyMetaheuristic src — Liver Cancer Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class LCAEngine(PortedPopulationEngine):
    algorithm_id   = "lca"
    algorithm_name = "Liver Cancer Algorithm"
    family         = "bio"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2023.111039"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, pc=0.2, pm=0.1)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        pc = float(self._params.get("pc", 0.2)); pm = float(self._params.get("pm", 0.1))
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        rate = 1 - t / max_iter
        for i in range(n):
            # Replication (move toward best)
            Xi = pop[i, :-1].copy()
            if np.random.random() < pc:
                r = np.random.random(d)
                Xi = pop[i, :-1] + r * (best_pos - pop[i, :-1])
            # Invasion (crossover with random cell)
            j = i
            while j == i: j = np.random.randint(n)
            mask = np.random.random(d) < 0.5
            Xi[mask] = pop[j, :-1][mask]
            # Angiogenesis (random mutation)
            for k in range(d):
                if np.random.random() < pm:
                    Xi[k] = lo[k] + np.random.random() * (hi[k] - lo[k])
            Xi = np.clip(Xi, lo, hi)
            new_fit = float(self._evaluate_population(Xi[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(Xi, new_fit)
        return pop, evals, {}
