"""pyMetaheuristic src — Seagull Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SOAEngine(PortedPopulationEngine):
    """Seagull Optimization Algorithm — spiral-attack with linearly decreasing frequency."""
    algorithm_id   = "soa"
    algorithm_name = "Seagull Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2018.11.024"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, fc=2.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        fc      = float(self._params.get("fc", 2.0))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        A   = fc - t * fc / T          # Eq. 6 — linearly decreased
        uu  = vv = 1.0

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            B   = 2.0 * A ** 2 * np.random.random()          # Eq. 8
            M   = B * (best_pos - pop[i, :-1])               # Eq. 7
            C   = A * pop[i, :-1]                            # Eq. 5
            D   = np.abs(C + M)                              # Eq. 9
            k   = np.random.uniform(0, 2.0 * np.pi)
            r   = uu * np.exp(k * vv)
            xx  = r * np.cos(k); yy = r * np.sin(k); zz = r * k
            pos = xx * yy * zz * D + np.random.normal(0, 1, dim) * best_pos  # Eq. 14
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
