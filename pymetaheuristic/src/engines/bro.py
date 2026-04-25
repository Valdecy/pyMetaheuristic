"""pyMetaheuristic src — Battle Royale Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BROEngine(PortedPopulationEngine):
    """Battle Royale Optimization — nearest-neighbour combat with damage accumulation."""
    algorithm_id   = "bro"
    algorithm_name = "Battle Royale Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s00521-020-05004-4"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, threshold=3)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        return {"damage": np.zeros(n, dtype=int),
                "dyn_delta": max(1, n // 5),
                "lb_dyn": self._lo.copy(),
                "ub_dyn": self._hi.copy()}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim     = pop.shape[0], self.problem.dimension
        threshold  = int(self._params.get("threshold", 3))
        damage     = np.asarray(state.payload.get("damage", np.zeros(n, int)), dtype=int)
        dyn_delta  = int(state.payload.get("dyn_delta", max(1, n // 5)))
        lb_d       = np.asarray(state.payload.get("lb_dyn", self._lo.copy()), dtype=float)
        ub_d       = np.asarray(state.payload.get("ub_dyn", self._hi.copy()), dtype=float)
        evals      = 0
        t          = state.step + 1

        for i in range(n):
            # Find nearest neighbour
            dists = [np.linalg.norm(pop[i, :-1] - pop[j, :-1]) if j != i else np.inf for j in range(n)]
            j = int(np.argmin(dists))

            dam, vic = (j, i) if self._is_better(float(pop[i, -1]), float(pop[j, -1])) else (i, j)

            if damage[dam] < threshold:
                pos = np.random.random(dim) * (np.maximum(pop[dam, :-1], pop[0, :-1]) -
                                                np.minimum(pop[dam, :-1], pop[0, :-1])) + \
                      np.maximum(pop[dam, :-1], pop[0, :-1])
                pos = np.clip(pos, lb_d, ub_d)
                fit = float(self.problem.evaluate(pos)); evals += 1
                pop[dam, :-1] = pos; pop[dam, -1] = fit
                damage[dam] += 1; damage[vic] = 0
            else:
                pos = np.random.uniform(lb_d, ub_d)
                fit = float(self.problem.evaluate(pos)); evals += 1
                pop[dam, :-1] = pos; pop[dam, -1] = fit
                damage[dam] = 0

        # Dynamic bound contraction
        if t >= dyn_delta:
            std = np.std(pop[:, :-1], axis=0) + 1e-30
            order = self._order(pop[:, -1])
            best  = pop[order[0], :-1]
            lb_d  = np.clip(best - std, self._lo, self._hi)
            ub_d  = np.clip(best + std, self._lo, self._hi)
            # Ensure lb < ub
            swap  = lb_d > ub_d
            lb_d[swap], ub_d[swap] = ub_d[swap], lb_d[swap]
            dyn_delta += max(1, dyn_delta // 2)

        return pop, evals, {"damage": damage, "dyn_delta": dyn_delta,
                            "lb_dyn": lb_d, "ub_dyn": ub_d}
