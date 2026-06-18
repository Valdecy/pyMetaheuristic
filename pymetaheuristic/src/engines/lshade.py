"""pyMetaheuristic src — L-SHADE Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class LSHADEEngine(PortedPopulationEngine):
    """Linear Population Size Reduction SHADE.

    This engine implements the L-SHADE structure from Tanabe and Fukunaga:
    SHADE's success-history F/CR memories, current-to-pbest/1/bin mutation with
    an external archive, greedy selection, and linear population size reduction.
    """

    algorithm_id = "lshade"
    algorithm_name = "L-SHADE"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2014.6900380",
        "title": "Improving the Search Performance of SHADE Using Linear Population Size Reduction",
        "authors": "Ryoji Tanabe and Alex S. Fukunaga",
        "year": 2014,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        has_archive=True,
    )
    _DEFAULTS = dict(
        population_size=100,
        min_population_size=4,
        extern_arc_rate=2.6,
        pbest_factor=0.11,
        hist_mem_size=6,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        if self._n < 4:
            raise ValueError("population_size must be >= 4 for L-SHADE.")
        self._last_operator_contributions = self._blank_operator_contribs()
        self._last_operator_counts = {key: 0 for key in self._last_operator_contributions}

    def _blank_operator_contribs(self) -> dict[str, float]:
        return {
            "lshade.mutation": 0.0,
            "lshade.crossover": 0.0,
            "lshade.selection": 0.0,
            "lshade.archive_update": 0.0,
            "lshade.success_history_update": 0.0,
            "lshade.population_reduction": 0.0,
        }

    @staticmethod
    def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        return float(np.sum(weights * values * values) / (np.sum(weights * values) + 1.0e-30))

    def _sample_f(self, center: float) -> float:
        for _ in range(100):
            value = float(np.random.standard_cauchy() * 0.1 + center)
            if value > 0.0:
                return min(value, 1.0)
        return 0.5

    def _sample_cr(self, center: float) -> float:
        if not np.isfinite(center):
            return 0.0
        return float(np.clip(np.random.normal(center, 0.1), 0.0, 1.0))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> np.ndarray:
        donor = np.asarray(donor, dtype=float).copy()
        below = donor < self._lo
        above = donor > self._hi
        donor[below] = (self._lo[below] + parent[below]) / 2.0
        donor[above] = (self._hi[above] + parent[above]) / 2.0
        return np.clip(donor, self._lo, self._hi)

    def _target_population_size(self, current_evaluations: int, initial_n: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            progress = min(1.0, max(0.0, float(current_evaluations) / float(self.config.max_evaluations)))
        else:
            horizon = max(1, int(self.config.max_steps or 100))
            progress = min(1.0, max(0.0, float(self._current_step_for_lpsr + 1) / float(horizon)))
        return max(min_n, int(round(initial_n + (min_n - initial_n) * progress)))

    def _initialize_payload(self, pop):
        h = int(self._params.get("hist_mem_size", 6))
        if h <= 0:
            raise ValueError("hist_mem_size must be positive for L-SHADE.")
        return {
            "M_F": np.full(h, 0.5, dtype=float),
            "M_CR": np.full(h, 0.5, dtype=float),
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
        }

    def _step_impl(self, state, pop):
        self._current_step_for_lpsr = int(state.step)
        n, dim = pop.shape[0], self.problem.dimension
        if n < 4:
            raise ValueError("L-SHADE requires at least 4 population members.")

        M_F = np.asarray(state.payload.get("M_F", np.full(6, 0.5)), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR", np.full(6, 0.5)), dtype=float)
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        k = int(state.payload.get("k", 0)) % len(M_F)
        initial_n = int(state.payload.get("initial_n", n))
        p = float(self._params.get("pbest_factor", 0.11))
        p = float(np.clip(p, 2.0 / max(2, n), 1.0))
        pnum = max(2, int(round(p * n)))
        order = self._order(pop[:, -1])
        union = np.vstack((pop[:, :-1], archive)) if archive.size else pop[:, :-1]

        trials: list[np.ndarray] = []
        Fs: list[float] = []
        CRs: list[float] = []
        for i in range(n):
            r = int(np.random.randint(len(M_F)))
            F = self._sample_f(float(M_F[r]))
            CR = self._sample_cr(float(M_CR[r]))
            pbest = pop[np.random.choice(order[:pnum]), :-1]
            r1_idx = self._rand_indices(n, i, 1)[0]
            r2_idx = int(np.random.randint(union.shape[0]))
            parent = pop[i, :-1]
            donor = parent + F * (pbest - parent) + F * (pop[r1_idx, :-1] - union[r2_idx])
            donor = self._repair_bound(donor, parent)
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trials.append(np.where(cross, donor, parent))
            Fs.append(F)
            CRs.append(CR)

        trial_pop = self._pop_from_positions(np.asarray(trials, dtype=float))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        strict_mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - trial_pop[mask, -1]
        else:
            gains[mask] = trial_pop[mask, -1] - pop[mask, -1]

        contrib = self._blank_operator_contribs()
        counts = {key: 0 for key in contrib}
        accepted_gain = float(np.sum(np.maximum(gains, 0.0)))
        if accepted_gain > 0.0:
            contrib["lshade.mutation"] = accepted_gain / 3.0
            contrib["lshade.crossover"] = accepted_gain / 3.0
            contrib["lshade.selection"] = accepted_gain / 3.0
        counts["lshade.mutation"] = n
        counts["lshade.crossover"] = n
        counts["lshade.selection"] = int(np.count_nonzero(mask))

        if np.any(mask):
            archive = np.vstack((archive, pop[strict_mask, :-1])) if archive.size else pop[strict_mask, :-1].copy()
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            contrib["lshade.archive_update"] = 0.0
            counts["lshade.archive_update"] = int(np.count_nonzero(strict_mask))

            sf = np.asarray(Fs, dtype=float)[mask]
            scr = np.asarray(CRs, dtype=float)[mask]
            df = np.abs(pop[mask, -1] - trial_pop[mask, -1])
            weights = df / (float(np.sum(df)) + 1.0e-30)
            M_F[k] = self._weighted_lehmer(sf, weights)
            if np.max(scr) <= 0.0 or not np.isfinite(M_CR[k]):
                M_CR[k] = np.nan
            else:
                M_CR[k] = self._weighted_lehmer(scr, weights)
            k = (k + 1) % len(M_F)
            contrib["lshade.success_history_update"] = 0.0
            counts["lshade.success_history_update"] = 1
            pop[mask] = trial_pop[mask]

        target_n = self._target_population_size(state.evaluations + n, initial_n)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = pop.shape[0] - target_n
            pop = pop[keep]
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * target_n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            contrib["lshade.population_reduction"] = 0.0
            counts["lshade.population_reduction"] = int(removed)

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, n, {"M_F": M_F, "M_CR": M_CR, "k": k, "archive": archive, "initial_n": initial_n}

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        return obs
