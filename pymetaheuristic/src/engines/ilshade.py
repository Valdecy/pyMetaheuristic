"""pyMetaheuristic src — paper-faithful iL-SHADE engine.

This module implements iL-SHADE as described by Brest, Sepesy Maucec, and
Boskovic (CEC 2016): L-SHADE with iL-SHADE's CR/F memory changes, a fixed
0.9 memory cell, early F/CR schedule restrictions, linear p-best decay, external
archive, and linear population-size reduction.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used by SHADE-family success-history memories."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """Sample Cauchy(center, scale), resampling non-positive F and capping at upper."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(center, 0.5, _EPS), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Sample a clipped Gaussian CR; negative/terminal memory gives CR=0."""
    if not np.isfinite(center) or center < 0.0:
        return 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


class ILSHADEEngine(PortedPopulationEngine):
    """iL-SHADE: Improved Linear Population Size Reduction SHADE."""

    algorithm_id = "ilshade"
    algorithm_name = "Improved L-SHADE"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2016.7743922",
        "title": "iL-SHADE: Improved L-SHADE Algorithm for Single Objective Real-Parameter Optimization",
        "authors": "Janez Brest, Mirjam Sepesy Maucec, and Borko Boskovic",
        "year": 2016,
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
        # Paper default: N_init = rNinit * D with rNinit=12.  Passing an
        # explicit integer population_size overrides this package default.
        population_size=None,
        population_size_factor=12,
        min_population_size=4,
        hist_mem_size=6,
        memory_f_init=0.5,
        memory_cr_init=0.8,
        reserved_memory_value=0.9,
        extern_arc_rate=2.6,
        pbest_start=0.2,
        pbest_end=0.1,
        f_scale=0.1,
        cr_sigma=0.1,
    )

    def __init__(self, problem, config) -> None:
        raw_params = dict(config.params or {})
        pop_param = raw_params.get("population_size", self._DEFAULTS.get("population_size"))
        if pop_param in (None, 0, "paper", "auto"):
            factor = float(raw_params.get("population_size_factor", self._DEFAULTS["population_size_factor"]))
            auto_n = max(4, int(round(factor * int(problem.dimension))))
            patched = {**raw_params, "population_size": auto_n}
            config = replace(config, params=patched)
        super().__init__(problem, config)
        if self._n < 4:
            raise ValueError("population_size must be >= 4 for iL-SHADE.")
        self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = self._blank_contribs()
        self._last_operator_counts = self._blank_counts()

    # ------------------------------------------------------------------
    # EvoMapX telemetry helpers
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        return [
            f"{self.algorithm_id}.current_to_pbest_mutation",
            f"{self.algorithm_id}.binomial_crossover",
            f"{self.algorithm_id}.greedy_selection",
            f"{self.algorithm_id}.external_archive_update",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.linear_population_size_reduction",
            f"{self.algorithm_id}.pbest_schedule",
            f"{self.algorithm_id}.fixed_memory_cell",
            f"{self.algorithm_id}.early_parameter_control",
            f"{self.algorithm_id}.midpoint_bound_repair",
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
    # Paper-specific schedules and sampling
    # ------------------------------------------------------------------
    def _progress(self, state, *, after_evaluations: int = 0) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip((state.evaluations + after_evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        return float(np.clip(state.step / float(horizon), 0.0, 1.0))

    def _p_value(self, progress: float, n: int) -> float:
        pmax = float(self._params.get("pbest_start", 0.2))
        pmin = float(self._params.get("pbest_end", 0.1))
        p = pmax + (pmin - pmax) * float(progress)
        return float(np.clip(p, 2.0 / max(2, n), 1.0))

    def _target_population_size(self, state, initial_n: int, current_n: int, evaluated_this_step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(state, after_evaluations=evaluated_this_step)
        target = initial_n + (min_n - initial_n) * progress
        return int(np.clip(round(target), min_n, max(min_n, initial_n, current_n)))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, int]:
        donor = np.asarray(donor, dtype=float).copy()
        parent = np.asarray(parent, dtype=float)
        below = donor < self._lo
        above = donor > self._hi
        repairs = int(np.count_nonzero(below) + np.count_nonzero(above))
        donor[below] = 0.5 * (self._lo[below] + parent[below])
        donor[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(donor, self._lo, self._hi), repairs

    def _sample_f_cr(self, M_F: np.ndarray, M_CR: np.ndarray, memory_index: int, progress: float) -> tuple[float, float, bool]:
        reserved = memory_index == len(M_F) - 1
        if reserved:
            center_f = center_cr = float(self._params.get("reserved_memory_value", 0.9))
        else:
            center_f = float(M_F[memory_index])
            center_cr = float(M_CR[memory_index])
        CR = _normal_cr(center_cr, float(self._params.get("cr_sigma", 0.1)))
        if progress < 0.25:
            CR = max(CR, 0.5)
        elif progress < 0.5:
            CR = max(CR, 0.25)
        F = _positive_cauchy(center_f, float(self._params.get("f_scale", 0.1)), upper=1.0)
        if progress < 0.25:
            F = min(F, 0.7)
        elif progress < 0.5:
            F = min(F, 0.8)
        elif progress < 0.75:
            F = min(F, 0.9)
        return float(F), float(CR), bool(reserved)

    def _choice_r2_from_population_or_archive(self, pop: np.ndarray, archive: np.ndarray, i: int, r1_idx: int) -> np.ndarray:
        n = pop.shape[0]
        union_n = n + int(archive.shape[0])
        for _ in range(100):
            j = int(np.random.randint(union_n)) if union_n > 0 else 0
            if j < n:
                if j != i and j != r1_idx:
                    return pop[j, :-1].copy()
            else:
                return archive[j - n].copy()
        candidates = [j for j in range(n) if j not in {i, r1_idx}]
        if candidates:
            return pop[int(np.random.choice(candidates)), :-1].copy()
        if archive.size:
            return archive[int(np.random.randint(archive.shape[0]))].copy()
        return pop[r1_idx, :-1].copy()

    def _replacement_masks(self, trial_fit: np.ndarray, current_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.problem.objective == "min":
            replace_mask = trial_fit <= current_fit
            success_mask = trial_fit < current_fit
        else:
            replace_mask = trial_fit >= current_fit
            success_mask = trial_fit > current_fit
        return replace_mask, success_mask

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(2, int(self._params.get("hist_mem_size", 6)))
        self._params["hist_mem_size"] = h
        reserved = float(self._params.get("reserved_memory_value", 0.9))
        m_f = np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float)
        m_cr = np.full(h, float(self._params.get("memory_cr_init", 0.8)), dtype=float)
        # The final memory cell is sampled but never overwritten by the update.
        m_f[-1] = reserved
        m_cr[-1] = reserved
        return {
            "M_F": m_f,
            "M_CR": m_cr,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
        }

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        if n < 4:
            raise ValueError("iL-SHADE requires at least 4 population members.")

        M_F = np.asarray(state.payload.get("M_F"), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float)
        h = len(M_F)
        if h < 2:
            raise ValueError("hist_mem_size must be at least 2 because iL-SHADE reserves one fixed memory cell.")
        reserved = float(self._params.get("reserved_memory_value", 0.9))
        M_F[-1] = reserved
        M_CR[-1] = reserved
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        k = int(state.payload.get("k", 0)) % (h - 1)
        initial_n = int(state.payload.get("initial_n", n))

        progress = self._progress(state)
        p = self._p_value(progress, n)
        pnum = max(2, int(round(p * n)))
        order = self._order(pop[:, -1])

        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        used_reserved = 0
        bound_repairs = 0

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        self._add(contrib, counts, "pbest_schedule", 0.0, n)

        for i in range(n):
            r = int(np.random.randint(h))
            F, CR, is_reserved = self._sample_f_cr(M_F, M_CR, r, progress)
            used_reserved += int(is_reserved)
            parent = pop[i, :-1]
            pbest = pop[int(np.random.choice(order[:pnum])), :-1]
            r1_idx = int(self._rand_indices(n, i, 1)[0])
            r2 = self._choice_r2_from_population_or_archive(pop, archive, i, r1_idx)
            donor = parent + F * (pbest - parent) + F * (pop[r1_idx, :-1] - r2)
            donor, repaired = self._repair_bound(donor, parent)
            bound_repairs += repaired
            cross = np.random.rand(dim) < CR
            cross[int(np.random.randint(dim))] = True
            trials[i] = np.where(cross, donor, parent)
            Fs[i] = F
            CRs[i] = CR

        self._add(contrib, counts, "current_to_pbest_mutation", 0.0, n)
        self._add(contrib, counts, "binomial_crossover", 0.0, n)
        self._add(contrib, counts, "fixed_memory_cell", 0.0, used_reserved)
        self._add(contrib, counts, "early_parameter_control", 0.0, n if progress < 0.75 else 0)
        self._add(contrib, counts, "midpoint_bound_repair", 0.0, bound_repairs)

        trial_pop = self._pop_from_positions(trials)
        evals = n
        replace_mask, success_mask = self._replacement_masks(trial_pop[:, -1], pop[:, -1])

        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[success_mask] = pop[success_mask, -1] - trial_pop[success_mask, -1]
        else:
            gains[success_mask] = trial_pop[success_mask, -1] - pop[success_mask, -1]
        accepted_gain = float(np.sum(np.maximum(gains, 0.0)))
        if accepted_gain > 0.0:
            self._add(contrib, counts, "current_to_pbest_mutation", accepted_gain / 3.0, 0)
            self._add(contrib, counts, "binomial_crossover", accepted_gain / 3.0, 0)
            self._add(contrib, counts, "greedy_selection", accepted_gain / 3.0, 0)
        self._add(contrib, counts, "greedy_selection", 0.0, int(np.count_nonzero(replace_mask)))

        if np.any(success_mask):
            old_success = pop[success_mask, :-1].copy()
            archive = np.vstack((archive, old_success)) if archive.size else old_success
            arc_rate = float(self._params.get("extern_arc_rate", self._params.get("archive_rate", 2.6)))
            max_arc = max(1, int(round(arc_rate * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "external_archive_update", 0.0, int(np.count_nonzero(success_mask)))

            sf = Fs[success_mask]
            scr = CRs[success_mask]
            df = np.abs(pop[success_mask, -1] - trial_pop[success_mask, -1])
            weights = df / (float(np.sum(df)) + _EPS)
            old_mf = float(M_F[k])
            old_mcr = float(M_CR[k])
            M_F[k] = 0.5 * (_weighted_lehmer(sf, weights) + old_mf)
            if np.max(scr) <= 0.0 or old_mcr < 0.0 or not np.isfinite(old_mcr):
                M_CR[k] = 0.0
            else:
                M_CR[k] = 0.5 * (_weighted_lehmer(scr, weights) + old_mcr)
            k = (k + 1) % (h - 1)
            self._add(contrib, counts, "success_history_update", 0.0, 1)

        pop[replace_mask] = trial_pop[replace_mask]

        target_n = self._target_population_size(state, initial_n, pop.shape[0], evals)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = pop.shape[0] - target_n
            pop = pop[keep]
            arc_rate = float(self._params.get("extern_arc_rate", self._params.get("archive_rate", 2.6)))
            max_arc = max(1, int(round(arc_rate * target_n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "linear_population_size_reduction", 0.0, int(removed))

        M_F[-1] = reserved
        M_CR[-1] = reserved
        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {"M_F": M_F, "M_CR": M_CR, "k": k, "archive": archive, "initial_n": initial_n}

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        obs["pbest_value"] = self._p_value(self._progress(state), state.payload["population"].shape[0])
        obs["archive_size"] = int(np.asarray(state.payload.get("archive", np.empty((0, self.problem.dimension)))).reshape(-1, self.problem.dimension).shape[0])
        return obs
