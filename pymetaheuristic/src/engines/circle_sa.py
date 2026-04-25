"""pyMetaheuristic src — Circle-Based Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CIRCLESAEngine(PortedPopulationEngine):
    """Circle-Based Search Algorithm — tangent-guided exploitation with c_factor phase switch."""
    algorithm_id   = "circle_sa"
    algorithm_name = "Circle-Based Search Algorithm"
    family         = "math"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, c_factor=0.8)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        cf      = float(self._params.get("c_factor", 0.8))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        threshold = cf * T

        a  = np.pi - np.pi * (t / T) ** 2          # Eq. 8
        p  = 1.0 - 0.9 * (t / T) ** 0.5

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            w = a * np.random.random() - a
            if t > threshold:
                pos = best_pos + (best_pos - pop[i, :-1]) * np.tan(w * np.random.random())
            else:
                pos = best_pos - (best_pos - pop[i, :-1]) * np.tan(w * p)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # full replacement (original)
        return pop, n, {}
