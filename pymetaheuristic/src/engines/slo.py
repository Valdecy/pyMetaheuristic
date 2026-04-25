"""pyMetaheuristic src — Sea Lion Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SLOEngine(PortedPopulationEngine):
    """Sea Lion Optimization — spiral and linear encircling prey hunting."""
    algorithm_id   = "slo"
    algorithm_name = "Sea Lion Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.14569/IJACSA.2019.0100548"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        c        = 2.0 - 2.0 * t / T

        t0 = np.random.random()
        v1 = np.sin(2.0 * np.pi * t0)
        v2 = np.sin(2.0 * np.pi * (1.0 - t0))
        denom = v2 if abs(v2) > 1e-12 else 1e-12
        SP_leader = abs(v1 * (1.0 + v2) / denom)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if SP_leader < 0.25:
                if c < 1.0:
                    pos = best_pos - c * np.abs(2.0 * np.random.random() * best_pos - pop[i, :-1])
                else:
                    ri  = np.random.choice([k for k in range(n) if k != i])
                    pos = pop[ri, :-1] - c * np.abs(2.0 * np.random.random() * pop[ri, :-1] - pop[i, :-1])
            else:
                pos = np.abs(best_pos - pop[i, :-1]) * np.cos(2.0 * np.pi * np.random.uniform(-1, 1)) + best_pos
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
