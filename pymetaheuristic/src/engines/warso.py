"""pyMetaheuristic src — War Strategy Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WARSOEngine(PortedPopulationEngine):
    """War Strategy Optimization — rank-sorted troop update with win-history weighting."""
    algorithm_id   = "warso"
    algorithm_name = "War Strategy Optimization"
    family         = "human"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, rr=0.1)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        return {"wl": np.ones(n), "wg": np.zeros(n, int)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        rr      = float(self._params.get("rr", 0.1))

        wl = np.asarray(state.payload.get("wl", np.ones(n)), dtype=float)
        wg = np.asarray(state.payload.get("wg", np.zeros(n, int)), dtype=int)

        order     = self._order(pop[:, -1])
        sorted_pop = pop[order]
        wl_sorted  = wl[order]
        wg_sorted  = wg[order]

        perm = np.random.permutation(n)

        new_pop = pop.copy()
        for i in range(n):
            r1 = np.random.random()
            si = order[i]                   # original index of sorted[i]
            if r1 < rr:
                pos = (2 * r1 * (self._order(pop[:, -1])[0] and pop[order[0], :-1] - pop[perm[i], :-1]) +
                       wl_sorted[i] * np.random.random() * (sorted_pop[i, :-1] - pop[si, :-1]))
                pos = 2 * r1 * (pop[order[0], :-1] - pop[perm[i], :-1]) + \
                      wl_sorted[i] * np.random.random() * (sorted_pop[i, :-1] - pop[si, :-1])
            else:
                pos = 2 * r1 * (sorted_pop[i, :-1] - pop[order[0], :-1]) + \
                      np.random.random() * (wl_sorted[i] * pop[order[0], :-1] - pop[si, :-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos))
            if self._is_better(fit, float(pop[si, -1])):
                new_pop[si, :-1] = pos; new_pop[si, -1] = fit
                wg[si] += 1
                wl[si] = wl[si] * (1.0 - wg[si] / T) ** 2

        return new_pop, n, {"wl": wl, "wg": wg}
