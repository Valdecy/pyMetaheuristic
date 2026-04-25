"""pyMetaheuristic src — Social Ski-Driver Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SSDOEngine(PortedPopulationEngine):
    """Social Ski-Driver Optimization — sine/cosine velocity update around top-3 mean."""
    algorithm_id   = "ssdo"
    algorithm_name = "Social Ski-Driver Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s00521-019-04159-z"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"velocity": np.zeros_like(pop[:, :-1]),
                "local":    pop[:, :-1].copy()}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        c       = 2.0 - t * (2.0 / T)

        order   = self._order(pop[:, -1])
        top3    = pop[order[:3], :-1]
        pos_mean = top3.mean(axis=0)

        vel   = np.asarray(state.payload.get("velocity", np.zeros((n, dim))), dtype=float)
        local = np.asarray(state.payload.get("local",    pop[:, :-1].copy()), dtype=float)

        r1, r2 = np.random.random(), np.random.random()
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if r2 <= 0.5:
                vel[i] = (c * np.sin(r1) * (local[i] - pop[i, :-1])
                          + (2.0 - c) * np.sin(r1) * (pos_mean - pop[i, :-1]))
            else:
                vel[i] = (c * np.cos(r1) * (local[i] - pop[i, :-1])
                          + (2.0 - c) * np.cos(r1) * (pos_mean - pop[i, :-1]))
            pos = np.random.normal(0, 1, dim) * pop[i, :-1] + np.random.random() * vel[i]
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i, -1])):
                local[i] = pop[i, :-1].copy()
                pop[i, :-1] = new_pos[i]; pop[i, -1] = new_fit[i]

        return pop, n, {"velocity": vel, "local": local}
