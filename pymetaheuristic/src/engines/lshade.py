"""pyMetaheuristic src — L-SHADE Engine.

Native implementation of Tanabe and Fukunaga's L-SHADE: SHADE 1.1
success-history adaptation, current-to-pbest/1/bin, external archive, and
linear population size reduction (LPSR).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30
_TERMINAL_CR = np.nan


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used by SHADE-family success memories."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


def _is_terminal_cr(value: float) -> bool:
    return (not np.isfinite(float(value))) or float(value) < 0.0


class LSHADEEngine(PortedPopulationEngine):
    """L-SHADE: SHADE with Linear Population Size Reduction."""

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
        has_archive=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper tuned setting: Ninit = round(18 * D).  Explicit population_size
        # still overrides this package default for user-controlled experiments.
        population_size=None,
        initial_population_multiplier=18.0,
        min_population_size=4,
        extern_arc_rate=2.6,
        pbest_factor=0.11,
        hist_mem_size=6,
        cauchy_scale=0.1,
        cr_sigma=0.1,
        resampling_attempts=100,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            multiplier = self._params.get("r_Ninit", self._params.get("r_ninit", self._params.get("initial_population_multiplier", 18.0)))
            self._n = max(4, int(round(float(multiplier) * float(self.problem.dimension))))
            self._params["population_size"] = self._n
        else:
            self._n = max(4, int(pop_param))
            self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = self._blank_contribs()
        self._last_operator_counts = self._blank_counts()

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        return [
            "lshade.parameter_sampling",
            "lshade.current_to_pbest_mutation",
            "lshade.midpoint_bound_repair",
            "lshade.binomial_crossover",
            "lshade.greedy_selection",
            "lshade.external_archive_update",
            "lshade.success_history_update",
            "lshade.linear_population_size_reduction",
        ]

    def _blank_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._operator_labels}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._operator_labels}

    def _add(self, contrib: dict[str, float], counts: dict[str, int], suffix: str, value: float = 0.0, count: int = 0) -> None:
        label = f"{self.algorithm_id}.{suffix}"
        if label in contrib:
            contrib[label] += float(value)
            counts[label] += int(count)

    # ------------------------------------------------------------------
    # Native SHADE/L-SHADE primitives
    # ------------------------------------------------------------------
    def _sample_f(self, center: float) -> float:
        scale = float(self._params.get("cauchy_scale", 0.1))
        attempts = max(1, int(self._params.get("resampling_attempts", 100)))
        for _ in range(attempts):
            value = float(np.random.standard_cauchy() * scale + float(center))
            if value > 0.0:
                return min(value, 1.0)
        # Defensive fallback for pathological random streams; the paper says to
        # resample until positive and truncate to one.
        return min(max(float(center), 0.5, _EPS), 1.0)

    def _sample_cr(self, center: float) -> float:
        if _is_terminal_cr(center):
            return 0.0
        sigma = float(self._params.get("cr_sigma", 0.1))
        return float(np.clip(np.random.normal(float(center), sigma), 0.0, 1.0))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, int]:
        donor = np.asarray(donor, dtype=float).copy()
        parent = np.asarray(parent, dtype=float)
        below = donor < self._lo
        above = donor > self._hi
        count = int(np.count_nonzero(below) + np.count_nonzero(above))
        donor[below] = (self._lo[below] + parent[below]) / 2.0
        donor[above] = (self._hi[above] + parent[above]) / 2.0
        return np.clip(donor, self._lo, self._hi), count

    def _select_r2_from_population_or_archive(self, pop: np.ndarray, archive: np.ndarray, i: int, r1_idx: int) -> np.ndarray:
        """Select x_r2 from P ∪ A while respecting JADE/SHADE index exclusions."""
        n = pop.shape[0]
        archive = np.asarray(archive, dtype=float).reshape(-1, self.problem.dimension)
        total = n + archive.shape[0]
        if total <= 0:
            return pop[r1_idx, :-1]
        for _ in range(100):
            idx = int(np.random.randint(total))
            if idx < n:
                if idx == i or idx == r1_idx:
                    continue
                return pop[idx, :-1]
            return archive[idx - n]
        valid_pop = [j for j in range(n) if j not in (i, r1_idx)]
        candidates = []
        if valid_pop:
            candidates.extend([pop[j, :-1] for j in valid_pop])
        if archive.size:
            candidates.extend([row for row in archive])
        if not candidates:
            return pop[r1_idx, :-1]
        return np.asarray(candidates[int(np.random.randint(len(candidates)))], dtype=float)

    def _target_population_size(self, current_evaluations: int, initial_n: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            progress = float(np.clip(float(current_evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        else:
            horizon = max(1, int(self.config.max_steps or 100))
            progress = float(np.clip(float(self._current_step_for_lpsr + 1) / float(horizon), 0.0, 1.0))
        return max(min_n, int(round(float(initial_n) + float(min_n - initial_n) * progress)))

    def _survival_mask(self, trial_fit: np.ndarray, parent_fit: np.ndarray) -> np.ndarray:
        if self.problem.objective == "min":
            return np.asarray(trial_fit) <= np.asarray(parent_fit)
        return np.asarray(trial_fit) >= np.asarray(parent_fit)

    def _strict_improvement_mask(self, trial_fit: np.ndarray, parent_fit: np.ndarray) -> np.ndarray:
        if self.problem.objective == "min":
            return np.asarray(trial_fit) < np.asarray(parent_fit)
        return np.asarray(trial_fit) > np.asarray(parent_fit)

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(1, int(self._params.get("hist_mem_size", 6)))
        self._params["hist_mem_size"] = h
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
        p = float(np.clip(float(self._params.get("pbest_factor", 0.11)), _EPS, 1.0))
        pnum = min(n, max(2, int(round(p * n))))
        order = self._order(pop[:, -1])

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        trials: list[np.ndarray] = []
        Fs: list[float] = []
        CRs: list[float] = []

        for i in range(n):
            r = int(np.random.randint(len(M_F)))
            F = self._sample_f(float(M_F[r]))
            CR = self._sample_cr(float(M_CR[r]))
            pbest = pop[np.random.choice(order[:pnum]), :-1]
            r1_idx = int(self._rand_indices(n, i, 1)[0])
            r2 = self._select_r2_from_population_or_archive(pop, archive, i, r1_idx)
            parent = pop[i, :-1]
            donor = parent + F * (pbest - parent) + F * (pop[r1_idx, :-1] - r2)
            donor, repair_count = self._repair_bound(donor, parent)
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trials.append(np.where(cross, donor, parent))
            Fs.append(F)
            CRs.append(CR)
            self._add(contrib, counts, "parameter_sampling", count=1)
            self._add(contrib, counts, "current_to_pbest_mutation", count=1)
            self._add(contrib, counts, "midpoint_bound_repair", count=repair_count)
            self._add(contrib, counts, "binomial_crossover", count=1)

        trial_pop = self._pop_from_positions(np.asarray(trials, dtype=float))
        survivor_mask = self._survival_mask(trial_pop[:, -1], pop[:, -1])
        strict_mask = self._strict_improvement_mask(trial_pop[:, -1], pop[:, -1])

        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[strict_mask] = pop[strict_mask, -1] - trial_pop[strict_mask, -1]
        else:
            gains[strict_mask] = trial_pop[strict_mask, -1] - pop[strict_mask, -1]
        accepted_gain = float(np.sum(np.maximum(gains, 0.0)))
        if accepted_gain > 0.0:
            contrib["lshade.current_to_pbest_mutation"] += accepted_gain / 3.0
            contrib["lshade.binomial_crossover"] += accepted_gain / 3.0
            contrib["lshade.greedy_selection"] += accepted_gain / 3.0
        counts["lshade.greedy_selection"] = int(np.count_nonzero(survivor_mask))

        if np.any(strict_mask):
            archive = np.vstack((archive, pop[strict_mask, :-1])) if archive.size else pop[strict_mask, :-1].copy()
            max_arc = max(1, int(round(float(self._params.get("extern_arc_rate", 2.6)) * float(n))))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            counts["lshade.external_archive_update"] = int(np.count_nonzero(strict_mask))

            sf = np.asarray(Fs, dtype=float)[strict_mask]
            scr = np.asarray(CRs, dtype=float)[strict_mask]
            df = np.abs(pop[strict_mask, -1] - trial_pop[strict_mask, -1])
            weights = df / (float(np.sum(df)) + _EPS)
            if _is_terminal_cr(M_CR[k]) or np.max(scr) <= 0.0:
                M_CR[k] = _TERMINAL_CR
            else:
                M_CR[k] = _weighted_lehmer(scr, weights)
            M_F[k] = _weighted_lehmer(sf, weights)
            k = (k + 1) % len(M_F)
            counts["lshade.success_history_update"] = 1

        if np.any(survivor_mask):
            pop[survivor_mask] = trial_pop[survivor_mask]

        target_n = self._target_population_size(state.evaluations + n, initial_n)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            max_arc = max(1, int(round(float(self._params.get("extern_arc_rate", 2.6)) * float(target_n))))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            counts["lshade.linear_population_size_reduction"] = removed

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
