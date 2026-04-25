"""pyMetaheuristic src — Political Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class PoliticalOEngine(PortedPopulationEngine):
    algorithm_id   = "political_o"
    algorithm_name = "Political Optimizer"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2020.106376"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, NC=5)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        NC = int(self._params.get("NC", 5))  # number of constituencies
        # Assign constituents to parties (best NC are party leaders)
        order = self._order(pop[:, -1])
        leaders = [pop[order[min(i, n-1)], :-1].copy() for i in range(NC)]
        # Electoral campaign — move toward nearest leader
        for i in range(n):
            dists = [np.linalg.norm(pop[i, :-1] - leaders[k]) for k in range(NC)]
            k = int(np.argmin(dists))
            r = np.random.random(d)
            new_pos = np.clip(pop[i, :-1] + r * (leaders[k] - pop[i, :-1]), lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
        # Parliamentary phase — leaders compete
        for k in range(NC):
            idx = order[k]
            j = np.random.randint(n)
            best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
            new_pos = np.clip(leaders[k] + np.random.randn(d) * (best_pos - leaders[k]) * (1 - t / max_iter), lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[idx, -1]):
                pop[idx] = np.append(new_pos, new_fit)
        return pop, evals, {}
