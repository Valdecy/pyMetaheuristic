"""pyMetaheuristic src — Numeric Crunch Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class NCAEngine(PortedPopulationEngine):
    """Numeric Crunch Algorithm (NCA).

    Population-distribution optimizer with exploration, exploitation, and
    hyperbolic acceleration phases.
    """

    algorithm_id = "nca"
    algorithm_name = "Numeric Crunch Algorithm"
    family = "math"
    _REFERENCE = {"doi": "10.1007/s00500-023-08925-z"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _safe_uniform(self, lo, hi, shape):
        lo_arr = np.minimum(lo, hi)
        hi_arr = np.maximum(lo, hi)
        lo_arr = np.maximum(lo_arr, self._lo)
        hi_arr = np.minimum(hi_arr, self._hi)
        bad = lo_arr >= hi_arr
        if np.any(bad):
            lo_arr = np.where(bad, self._lo, lo_arr)
            hi_arr = np.where(bad, self._hi, hi_arr)
        return np.random.uniform(lo_arr, hi_arr, shape)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        k = max(1, min(n - 1, int(round(n / 2))))
        eta = max(1.0, float(k))
        trials = pop[:, :-1].copy()
        evals = 0

        # Exploration: first half creates row-wise adaptive bounds from its own
        # coordinate distribution and samples inside those intervals.
        for i in range(k):
            row = pop[i, :-1]
            Li, Ui = float(np.min(row)), float(np.max(row))
            r1 = max(np.random.random(), 1e-8)
            r2 = max(np.random.random(), 1e-8)
            new_l = Li * (1.0 - r1) / r1
            new_u = Ui * (1.0 - r2) / r2
            center = 0.5 * (row + np.asarray(state.best_position, dtype=float))
            width = np.abs(new_u - new_l)
            if not np.isfinite(width) or width <= 1e-12:
                width = float(np.mean(self._span))
            local_lo = center - 0.5 * width * np.random.random(dim)
            local_hi = center + 0.5 * width * np.random.random(dim)
            trials[i] = self._safe_uniform(local_lo, local_hi, (dim,))

        first_fit = self._evaluate_population(trials[:k]); evals += k
        first_pop = np.hstack([trials[:k], first_fit[:, None]])
        lbest = first_pop[self._best_index(first_pop[:, -1]), :-1]

        # Exploitation: second half is crunched around the best discovered in
        # exploration using algebraic alpha/beta combinations.
        for i in range(k, n):
            xi = pop[i, :-1]
            r = np.maximum(np.random.random(dim), 1e-8)
            if np.random.randint(0, n) <= eta:
                alpha = xi * lbest + r * lbest - (r * lbest) / eta
                beta = xi * lbest - (2.0 * xi / r) / eta
            else:
                alpha = xi * lbest + (2.0 * lbest / r) / eta
                beta = xi * lbest - (2.0 * xi / r) / eta
            d0 = np.sqrt((beta - alpha) ** 2) + 1e-12
            cand = np.where((alpha + beta < 0.0) & (alpha + d0 <= beta), beta / d0, alpha / d0)
            cand = np.where((alpha + beta > 0.0) & (beta + d0 <= alpha), beta / d0, cand)
            bad = (cand < self._lo) | (cand > self._hi) | ~np.isfinite(cand)
            if np.any(bad):
                cand[bad] = np.random.uniform(self._lo[bad], self._hi[bad])
            trials[i] = np.clip(cand, self._lo, self._hi)

        # Acceleration: hyperbolic contraction for a random subset of components.
        j_scale = max(1, state.step + 1)
        pa = np.random.random((n, dim))
        accel = trials.copy()
        accel_candidate = j_scale * (trials * np.tanh(np.e / (j_scale ** 2)))
        accel[pa <= 0.5] = accel_candidate[pa <= 0.5]
        accel = np.clip(accel, self._lo, self._hi)

        fit = self._evaluate_population(accel); evals += n
        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1] = accel[mask]
        pop[mask, -1] = fit[mask]
        return pop, evals, {}
