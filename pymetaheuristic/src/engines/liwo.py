"""pyMetaheuristic src — Leaf in Wind Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class LiWOEngine(PortedPopulationEngine):
    """Leaf in Wind Optimization (LiWO).

    Implements the light-breeze and strong-wind movement operators described in
    the LiWO paper, including translational motion, spiral perturbation, and
    dimension-wise strong-wind reset.
    """

    algorithm_id = "liwo"
    algorithm_name = "Leaf in Wind Optimization"
    family = "physics"
    _REFERENCE = {"doi": "10.1109/ACCESS.2024.3390670"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, spiral_probability=0.1, reset_probability=0.1, breeze_probability=0.3)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 1000)))
        p_spiral = float(self._params.get("spiral_probability", 0.1))
        p_reset = float(self._params.get("reset_probability", 0.1))
        p_breeze = float(self._params.get("breeze_probability", 0.3))
        omega_max = float(self._params.get("omega_max", 0.9))
        omega_min = float(self._params.get("omega_min", 0.0))
        omega = omega_max - t * ((omega_max - omega_min) / max_iter)
        C1 = np.exp(omega ** 5) - 1.0
        best = np.asarray(state.best_position, dtype=float)
        old_pos = pop[:, :-1].copy()
        trial = old_pos.copy()

        for i in range(n):
            # Breeze-driven translation and optional spiral motion.
            m1 = np.random.randint(n)
            breeze = old_pos[i] + np.random.random(dim) * (best - old_pos[m1])
            for j in range(dim):
                if np.random.random() < p_spiral:
                    m2 = np.random.randint(n)
                    d_wind_s = best[j] - old_pos[m2, j]
                    phi = 2.0 * np.pi * np.random.random()
                    breeze[j] = breeze[j] + C1 * d_wind_s * np.sin(phi) * phi
            out = (breeze < self._lo) | (breeze > self._hi)
            breeze[out] = old_pos[i, out]

            # Strong-wind one-dimensional displacement and optional reset.
            strong = old_pos[i].copy()
            j = np.random.randint(dim)
            m3 = np.random.randint(n)
            d_wind_st = old_pos[m3, j] - old_pos[i, j]
            strong[j] = strong[j] + d_wind_st * np.random.random()
            if np.random.random() < p_reset:
                strong[j] = self._lo[j] + np.random.random() * (self._hi[j] - self._lo[j])
            if strong[j] < self._lo[j] or strong[j] > self._hi[j]:
                strong[j] = old_pos[i, j]

            trial[i] = breeze if np.random.random() < p_breeze else strong

        trial = np.clip(trial, self._lo, self._hi)
        fit = self._evaluate_population(trial)
        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1] = trial[mask]
        pop[mask, -1] = fit[mask]
        return pop, n, {}
