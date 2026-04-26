"""pyMetaheuristic src — Multiple Trajectory Search Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class MTSEngine(PortedPopulationEngine):
    """Multiple Trajectory Search — coordinate local searches over multiple enabled trajectories."""
    algorithm_id = "mts"
    algorithm_name = "Multiple Trajectory Search"
    family = "trajectory"
    _REFERENCE     = {"doi": "10.5555/1689599.1689856"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, num_tests=5, num_searches=5, num_searches_best=5, num_enabled=17, bonus1=10, bonus2=1, search_range=0.2)

    def _initialize_payload(self, pop):
        return {"search_range": np.full((pop.shape[0], self.problem.dimension), float(self._params.get("search_range", 0.2))) * self._span, "grade": np.zeros(pop.shape[0])}

    def _local(self, x, fx, sr, loops):
        evals = 0
        for _ in range(loops):
            improved = False
            dims = np.random.permutation(self.problem.dimension)[:max(1, int(self._params.get("num_tests", 5)))]
            for d in dims:
                for sign in (1.0, -1.0):
                    y = x.copy(); y[d] = np.clip(y[d] + sign * sr[d], self._lo[d], self._hi[d])
                    fy = float(self.problem.evaluate(y)); evals += 1
                    if self._is_better(fy, fx):
                        x, fx, improved = y, fy, True
                        break
            if not improved: sr *= 0.5
        return x, fx, sr, evals

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        sr = np.asarray(state.payload.get("search_range", np.tile(0.2*self._span, (n,1))), dtype=float)
        grade = np.asarray(state.payload.get("grade", np.zeros(n)), dtype=float)
        if sr.shape != (n, self.problem.dimension): sr = np.tile(0.2*self._span, (n,1))
        if grade.shape[0] != n: grade = np.zeros(n)
        enabled = self._order(pop[:, -1])[:min(n, int(self._params.get("num_enabled", 17)))]
        evals = 0
        for i in enabled:
            loops = int(self._params.get("num_searches_best", 5)) if i == self._best_index(pop[:, -1]) else int(self._params.get("num_searches", 5))
            x, fx, new_sr, e = self._local(pop[i, :-1].copy(), float(pop[i, -1]), sr[i].copy(), loops)
            evals += e
            if self._is_better(fx, pop[i, -1]):
                pop[i, :-1], pop[i, -1] = x, fx; grade[i] += float(self._params.get("bonus1", 10))
            else:
                grade[i] -= float(self._params.get("bonus2", 1))
            sr[i] = new_sr
        return pop, evals, {"search_range": sr, "grade": grade}
