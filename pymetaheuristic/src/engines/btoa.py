"""pyMetaheuristic src — Basketball Team Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class BTOAEngine(PortedPopulationEngine):
    """Basketball Team Optimization Algorithm."""

    algorithm_id = "btoa"
    algorithm_name = "Basketball Team Optimization Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1038/s41598-025-05477-0",
        "title": "Basketball team optimization algorithm (BTOA): a novel sport-inspired meta-heuristic optimizer for engineering applications",
        "authors": "Yujie Chen, Guangyu Wang, Baichuan Yin, Chongyun Ma, Zhiqiao Wu, Ming Gao",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=50,
        strongnum_fraction=0.125,
        kappa=5.0,
        lambda_factor=1.0,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 3:
            raise ValueError("btoa requires population_size >= 3.")
        frac = float(self._params.get("strongnum_fraction", 0.125))
        if not 0.0 < frac <= 1.0:
            raise ValueError("btoa strongnum_fraction must be in (0, 1].")
        if float(self._params.get("kappa", 5.0)) <= 0.0:
            raise ValueError("btoa kappa must be positive.")
        if float(self._params.get("lambda_factor", 1.0)) < 0.0:
            raise ValueError("btoa lambda_factor must be non-negative.")

    def _binary_mask(self, dim: int, count: int) -> np.ndarray:
        count = max(1, min(dim, int(count)))
        mask = np.zeros(dim, dtype=float)
        mask[np.random.choice(dim, size=count, replace=False)] = 1.0
        return mask

    def _rank_alpha(self, rank: int, n: int, dim: int) -> int:
        if n <= 1:
            return dim
        return int(np.ceil(dim / 2.0 + ((rank - 1) / (n - 1)) * dim / 2.0))

    def _boundary_reentry(self, candidate: np.ndarray, old: np.ndarray) -> np.ndarray:
        x = np.asarray(candidate, dtype=float).copy()
        q = np.random.rand(x.size)
        p = np.random.rand(x.size)
        hi_mask = x > self._hi
        lo_mask = x < self._lo
        x[hi_mask] = q[hi_mask] * self._hi[hi_mask] + (1.0 - q[hi_mask]) * old[hi_mask]
        x[lo_mask] = p[lo_mask] * self._lo[lo_mask] + (1.0 - p[lo_mask]) * old[lo_mask]
        return np.clip(x, self._lo, self._hi)

    def _dynamic_position_candidate(self, current: np.ndarray, best: np.ndarray, n: int) -> tuple[np.ndarray, float, int]:
        dim = current.size
        prand = np.random.randint(n)
        drand = np.random.randint(dim)
        reference = self._work_positions[prand]
        denom = self._span[drand]
        ratio = 0.0 if denom <= 1.0e-12 else (reference[drand] - self._lo[drand]) / denom
        lrand = self._lo + ratio * self._span
        q1 = lrand + np.random.rand(dim) * (best - lrand)
        delta = np.random.randint(0, 2, size=dim)
        c1 = q1 + np.random.normal(0.0, 1.0, dim) * (q1 - delta * current)
        c1 = self._boundary_reentry(c1, current)

        dbest = np.random.randint(dim)
        denom_best = self._span[dbest]
        ratio_best = 0.0 if denom_best <= 1.0e-12 else (best[dbest] - self._lo[dbest]) / denom_best
        lbest = self._lo + ratio_best * self._span
        r = self._lo + np.random.rand(dim) * self._span
        c2 = np.empty(dim, dtype=float)
        rnd = np.random.rand(dim)
        below = r < lbest
        c2[below] = lbest[below] + rnd[below] * (self._hi[below] - lbest[below])
        c2[~below] = self._lo[~below] + rnd[~below] * (lbest[~below] - self._lo[~below])
        c2 = self._boundary_reentry(c2, current)

        f1 = float(self.problem.evaluate(c1))
        f2 = float(self.problem.evaluate(c2))
        if self._is_better(f1, f2):
            return c1, f1, 2
        return c2, f2, 2

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        order = self._order(pop[:, -1])
        ranked = pop[order].copy()
        positions = ranked[:, :-1].copy()
        fitness = ranked[:, -1].copy()
        self._work_positions = positions
        best = positions[0].copy()
        strongnum = max(1, int(np.ceil(float(self._params.get("strongnum_fraction", 0.125)) * n)))
        Tmax = max(1, int(self.config.max_steps or max(50, state.step + 1)))
        t = min(Tmax, state.step + 1)
        gamma_t = 0.1 + 0.8 * np.sqrt(max(0.0, 1.0 - t / Tmax))
        kappa = float(self._params.get("kappa", 5.0))
        lambda_factor = float(self._params.get("lambda_factor", 1.0))

        new_positions = positions.copy()
        new_fitness = fitness.copy()
        evals = 0
        for pos_idx in range(n):
            xi = positions[pos_idx].copy()
            rank = pos_idx + 1
            mask = self._binary_mask(dim, self._rank_alpha(rank, n, dim))
            if pos_idx < strongnum:
                beta_i = rank / strongnum
                peers = self._rand_indices(n, pos_idx, 2)
                pj = positions[peers[0]]
                pg = positions[peers[1]]
                candidate = best + (gamma_t * (pj - xi) + beta_i * (pg - xi)) * mask
                candidate = self._boundary_reentry(candidate, xi)
                fit = float(self.problem.evaluate(candidate))
                evals += 1
            else:
                eta = kappa * np.exp(-lambda_factor * rank / max(1, n)) * np.log(1.0 / max(np.random.rand(), 1.0e-12))
                if eta > 1.0:
                    peers = self._rand_indices(n, pos_idx, 2)
                    pj = positions[peers[0]]
                    pg = positions[peers[1]]
                    f = 2.0 - (t / Tmax) ** 2
                    candidate = xi + (np.random.rand(dim) * f * (best - xi) + np.random.rand(dim) * (1.0 - t / Tmax) * (pj - pg)) * mask
                    candidate = self._boundary_reentry(candidate, xi)
                    fit = float(self.problem.evaluate(candidate))
                    evals += 1
                else:
                    candidate, fit, extra = self._dynamic_position_candidate(xi, best, n)
                    evals += extra
            if self._is_better(fit, fitness[pos_idx]):
                new_positions[pos_idx] = candidate
                new_fitness[pos_idx] = fit

        new_ranked = np.hstack((new_positions, new_fitness[:, None]))
        inverse = np.empty_like(order)
        inverse[order] = np.arange(n)
        restored = new_ranked[inverse]
        return restored, evals, {}
