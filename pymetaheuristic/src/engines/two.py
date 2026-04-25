"""pyMetaheuristic src — Tug of War Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TWOEngine(PortedPopulationEngine):
    """Tug of War Optimization — teams compete via force-displacement mechanics."""
    algorithm_id   = "two"
    algorithm_name = "Tug of War Optimization"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.procs.2020.03.063"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, muy_s=1.0, muy_k=1.0, delta_t=1.0,
                     alpha=0.99, beta=0.1)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        # Assign initial weights proportional to quality (rank-based)
        n = pop.shape[0]
        order  = self._order(pop[:, -1])
        weight = np.empty(n)
        for rank, idx in enumerate(order):
            weight[idx] = (n - rank) / n     # best gets highest weight
        return {"weight": weight}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        muy_s   = float(self._params.get("muy_s", 1.0))
        muy_k   = float(self._params.get("muy_k", 1.0))
        delta_t = float(self._params.get("delta_t", 1.0))
        alpha   = float(self._params.get("alpha", 0.99))
        beta    = float(self._params.get("beta", 0.1))

        weight  = np.asarray(state.payload.get("weight", np.ones(n)), dtype=float)
        order   = self._order(pop[:, -1])
        g       = 2.0 * (1.0 - t / T)           # gravitational-like decay

        new_pos = pop[:, :-1].copy()
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                fw    = max(weight[i] * muy_s, weight[j] * muy_s)
                rf    = fw - weight[i] * muy_k
                if weight[i] * muy_k == 0:
                    continue
                acc   = rf * g / (weight[i] * muy_k)
                dx    = 0.5 * acc * delta_t**2 + (alpha**t) * beta * \
                        np.random.random() * (pop[j, :-1] - pop[i, :-1])
                new_pos[i] += dx
            new_pos[i] = np.clip(new_pos[i], self._lo, self._hi)

        # Update weights by rank
        new_fit = self._evaluate_population(new_pos)
        new_order = self._order(new_fit)
        new_w     = np.empty(n)
        for rank, idx in enumerate(new_order):
            new_w[idx] = (n - rank) / n
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"weight": new_w}
