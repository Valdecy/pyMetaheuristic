"""pyMetaheuristic src — Rain-Cloud Condensation Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class RCCOEngine(PortedPopulationEngine):
    """Rain-Cloud Condensation Optimizer.

    This implementation follows the fixed-schedule RCCO described in the paper.
    The paper also proposes optional adaptive coefficient and operator-selection
    modules; those are intentionally disabled here to preserve the paper's
    reproducible baseline.
    """

    algorithm_id = "rcco"
    algorithm_name = "Rain-Cloud Condensation Optimizer"
    family = 'physics'
    _REFERENCE = {
        "doi": "10.3390/eng6100281",
        "title": "Rain-Cloud Condensation Optimizer: Novel Nature-Inspired Metaheuristic for Solving Engineering Design Problems",
        "authors": "Sandi Fakhouri, Amjad Hudaib, Azzam Sleit, Hussam N. Fakhouri",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, mirror_probability=0.35, gust_probability=0.18)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("rcco requires population_size >= 2.")
        for key in ("mirror_probability", "gust_probability"):
            value = float(self._params.get(key))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rcco {key} must be in [0, 1].")

    def _wrap_reflect(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo, hi = self._lo, self._hi
        span = hi - lo
        out = x.copy()
        zero_mask = span <= 0.0
        if np.any(zero_mask):
            out[..., zero_mask] = lo[zero_mask]
        active = ~zero_mask
        if np.any(active):
            doubled = 2.0 * span[active]
            y = np.mod(out[..., active] - lo[active], doubled)
            out[..., active] = lo[active] + np.where(y <= span[active], y, doubled - y)
        return out

    def _new_positions(self, n: int | None = None) -> np.ndarray:
        n_points = self._n if n is None else int(n)
        dim = self.problem.dimension
        total = n_points * dim
        z = np.random.uniform(1.0e-6, 1.0 - 1.0e-6)
        seq = np.empty(total, dtype=float)
        for k in range(total):
            z = np.sin(np.pi * z)
            seq[k] = np.clip(z, 0.0, 1.0)
        return self._lo + seq.reshape(n_points, dim) * self._span

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        order = self._order(pop[:, -1])
        ranked = pop[order].copy()
        positions = ranked[:, :-1].copy()
        fitness = ranked[:, -1].copy()
        best = positions[0].copy()
        best_fit = float(fitness[0])
        q = max(2, int(np.floor(0.2 * n)))
        weights = np.arange(q, 0, -1, dtype=float)
        core = np.sum(weights[:, None] * positions[:q], axis=0) / (np.sum(weights) + 1.0e-12)
        Tmax = max(1, int(self.config.max_steps or max(50, state.step + 1)))
        t = min(Tmax, state.step + 1)
        decay = max(0.0, 1.0 - t / Tmax)
        beta = 0.70 * decay + 0.05 * np.random.rand()
        sigma = 0.30 * decay
        tau = 0.04 * decay * self._span
        evals = 0

        condensation = positions + beta * (best - positions) + sigma * (core - positions) + np.random.normal(0.0, tau, size=positions.shape)
        condensation = self._wrap_reflect(condensation)
        cond_fit = self._evaluate_population(condensation)
        evals += n
        improved = self._better_mask(cond_fit, fitness)
        positions[improved] = condensation[improved]
        fitness[improved] = cond_fit[improved]
        best_index = self._best_index(fitness)
        best = positions[best_index].copy()
        best_fit = float(fitness[best_index])

        mirror_probability = float(self._params.get("mirror_probability", 0.35))
        gust_probability = float(self._params.get("gust_probability", 0.18))
        for i in range(n):
            parents = self._rand_indices(n, i, 2)
            xp, xq = positions[parents[0]], positions[parents[1]]
            lower = np.minimum(xp, xq)
            upper = np.maximum(xp, xq)
            y = lower + np.random.rand(dim) * (upper - lower)
            if np.random.rand() < mirror_probability:
                y = self._lo + self._hi - y
            if np.random.rand() < gust_probability:
                u = np.random.rand(dim)
                y = y + decay * 0.02 * self._span * np.tan(np.pi * (u - 0.5))
            y = self._wrap_reflect(y)
            fy = float(self.problem.evaluate(y))
            evals += 1
            if self._is_better(fy, fitness[i]):
                positions[i] = y
                fitness[i] = fy
                if self._is_better(fy, best_fit):
                    best = y.copy()
                    best_fit = fy

        new_ranked = np.hstack((positions, fitness[:, None]))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(n)
        restored = new_ranked[inverse]
        return restored, evals, {}
