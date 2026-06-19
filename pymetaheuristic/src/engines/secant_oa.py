"""pyMetaheuristic src — Secant Optimization Algorithm Engine."""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class SecantOAEngine(PortedPopulationEngine):
    """Secant Optimization Algorithm (SOA) with secant-based and stochastic update phases.

    The canonical package ID is ``secant_oa`` because ``soa`` is already used by
    the Seagull Optimization Algorithm in this package.
    """

    algorithm_id = "secant_oa"
    algorithm_name = "Secant Optimization Algorithm"
    family = "math"
    _REFERENCE = {
        "doi": "10.1038/s41598-026-36691-z",
        "title": "Secant Optimization Algorithm for efficient global optimization",
        "authors": "Mohammed Q. Ibrahim, Mohammed Qaraad, Nazar K. Hussein, M. A. Farag, David Guinovart",
        "year": 2026,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(population_size=40, mutation_rate=0.2, beta=1.5, secant_epsilon=1e-12)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._operator_labels = [
            "secant_oa.secant_update",
            "secant_oa.stochastic_exploitation",
            "secant_oa.mutation_gate",
            "secant_oa.selection",
        ]
        self._last_operator_contributions = {k: 0.0 for k in self._operator_labels}
        self._last_operator_counts = {k: 0 for k in self._operator_labels}

    def _blank(self):
        return {k: 0.0 for k in self._operator_labels}, {k: 0 for k in self._operator_labels}

    def _levy_z(self, dim: int) -> np.ndarray:
        beta = float(self._params.get("beta", 1.5))
        sigma = (
            math.gamma(1.0 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)
        v1 = np.random.normal(0.0, sigma, dim)
        v2 = np.random.normal(0.0, 1.0, dim)
        return 0.01 * v1 / (np.abs(v2) ** (1.0 / beta) + 1.0e-30)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        mean = np.mean(pop[:, :-1], axis=0)
        dist = np.linalg.norm(pop[:, :-1] - mean, axis=1)
        closest = pop[int(np.argmin(dist)), :-1].copy()
        farthest = pop[int(np.argmax(dist)), :-1].copy()
        mutation_rate = float(self._params.get("mutation_rate", 0.2))
        eps = float(self._params.get("secant_epsilon", 1.0e-12))
        contrib, counts = self._blank()
        new_pos = np.empty((n, dim), dtype=float)

        for i in range(n):
            xi = pop[i, :-1]
            xr = pop[self._rand_indices(n, i, 1)[0], :-1]
            f_best = float(pop[order[0], -1])
            f_i = float(pop[i, -1])
            # Locate xr fitness without an extra objective call; fall back to nearest row if duplicated.
            xr_idx = int(np.argmin(np.linalg.norm(pop[:, :-1] - xr, axis=1)))
            f_r = float(pop[xr_idx, -1])
            denom = (f_r - f_i) - (f_best - f_i)
            denom = denom if abs(denom) > eps else math.copysign(eps, denom if denom != 0 else 1.0)
            # Vectorized form of the paper's modified secant update using stored fitness differences.
            x_sec = best - ((f_best - f_i) / denom) * (xi - best)
            x_sec = np.clip(x_sec, self._lo, self._hi)

            x_rand = np.random.uniform(self._lo, self._hi, dim)
            s1 = pop[self._rand_indices(n, i, 1)[0], :-1]
            s2 = pop[self._rand_indices(n, i, 1)[0], :-1]
            z = self._levy_z(dim)
            x_stage = xi + z * (closest - x_rand) + np.random.random(dim) * (farthest - s1)
            x_stage = best + z * (x_sec - xi) + np.random.random(dim) * (xi - s2) + 0.25 * (x_stage - xi)
            x_stage = np.clip(x_stage, self._lo, self._hi)
            if np.random.random() < mutation_rate:
                x_stage = xi.copy()
            new_pos[i] = x_stage

        fit = self._evaluate_population(new_pos)
        mask = self._better_mask(fit, pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - fit[mask]
        else:
            gains[mask] = fit[mask] - pop[mask, -1]
        gain = float(np.sum(np.maximum(gains, 0.0)))
        if gain > 0.0:
            contrib["secant_oa.secant_update"] = gain / 3.0
            contrib["secant_oa.stochastic_exploitation"] = gain / 3.0
            contrib["secant_oa.selection"] = gain / 3.0
        counts["secant_oa.secant_update"] = n
        counts["secant_oa.stochastic_exploitation"] = n
        counts["secant_oa.mutation_gate"] = n
        counts["secant_oa.selection"] = int(np.count_nonzero(mask))
        pop[mask, :-1] = new_pos[mask]
        pop[mask, -1] = fit[mask]
        self._last_operator_contributions = contrib
        self._last_operator_counts = counts
        return pop, n, {}

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        return obs
