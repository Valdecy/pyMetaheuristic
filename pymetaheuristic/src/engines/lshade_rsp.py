"""pyMetaheuristic src — faithful LSHADE-RSP Engine.

This module implements the LSHADE-RSP algorithm from Stanovov, Akhmedova,
and Semenkin (CEC 2018). LSHADE-RSP is an L-SHADE/jSO-family Differential
Evolution variant with rank-based selective pressure in the current-to-pbest/r
mutation, jSO-style weighted pbest term, increasing pbest pool, linear
population-size reduction, success-history F/CR adaptation, high-CR propagation,
early F capping, and an external archive whose size follows the population.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """SHADE-style Cauchy F sample: resample while F <= 0 and cap at upper."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(center, 0.5, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian CR sample clipped to [0, 1], with negative-memory guard."""
    if not np.isfinite(center) or center < 0.0:
        return 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


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


class LSHADERspEngine(PortedPopulationEngine):
    """LSHADE-RSP: paper-faithful L-SHADE with rank-based selective pressure."""

    algorithm_id = "lshade_rsp"
    algorithm_name = "LSHADE-RSP"
    family = "evolutionary"
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _REFERENCE = {
        "doi": "10.1109/CEC.2018.8477957",
        "title": "LSHADE Algorithm with Rank-Based Selective Pressure Strategy for Solving CEC 2017 Benchmark Problems",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2018,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper default: Nmax = 75 * D^(2/3), Nmin = 4.
        # None/"paper"/0 activates the dimension-dependent default.
        population_size=None,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.3,
        memory_cr_init=0.8,
        reserved_memory_value=0.9,
        rank_greediness=3.0,
        archive_rate=1.0,  # paper: archive size equals current N and is dynamic
        pbest_base=0.085,  # pb = 0.085 + 0.085 * NFE/NFEmax
        f_scale=0.1,
        cr_sigma=0.1,
        memory_update_mixing=0.5,
        resampling_attempts=100,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            d = max(2, int(self.problem.dimension))
            self._n = max(4, int(round(75.0 * np.power(float(d), 2.0 / 3.0))))
            self._params["population_size"] = self._n
        else:
            self._n = max(4, int(pop_param))
            self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        return [
            f"{self.algorithm_id}.mutation",
            f"{self.algorithm_id}.weighted_pbest_scaling",
            f"{self.algorithm_id}.crossover",
            f"{self.algorithm_id}.selection",
            f"{self.algorithm_id}.archive_update",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.population_reduction",
            f"{self.algorithm_id}.rank_selective_pressure",
            f"{self.algorithm_id}.bound_resampling",
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
    # Initialization and schedules
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(2, int(self._params.get("hist_mem_size", 5)))
        self._params["hist_mem_size"] = h
        reserved = float(self._params.get("reserved_memory_value", 0.9))
        m_f = np.full(h, float(self._params.get("memory_f_init", 0.3)), dtype=float)
        m_cr = np.full(h, float(self._params.get("memory_cr_init", 0.8)), dtype=float)
        # The LSHADE-RSP paper states that one memory cell always keeps
        # mu_F=mu_CR=0.9. We reserve the last memory cell permanently.
        m_f[-1] = reserved
        m_cr[-1] = reserved
        return {
            "M_F": m_f,
            "M_CR": m_cr,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
            "pb_value": float(self._params.get("pbest_base", 0.085)),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _pb_value(self, progress: float) -> float:
        base = float(self._params.get("pbest_base", 0.085))
        return float(np.clip(base * (1.0 + float(progress)), 1.0e-12, 1.0))

    def _pbest_count(self, n: int, progress: float) -> int:
        pb = self._pb_value(progress)
        return int(np.clip(round(pb * n), 2, max(2, n)))

    def _weighted_pbest_factor(self, F: float, progress: float) -> float:
        # Eq. (6) uses Fw; the paper imports the jSO piecewise schedule.
        if progress < 0.2:
            return 0.7 * float(F)
        if progress < 0.4:
            return 0.8 * float(F)
        return 1.2 * float(F)

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        target = round(((min_n - initial_n) * progress) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------
    def _rank_probabilities(self, n: int, order: np.ndarray) -> np.ndarray:
        # order is best-to-worst. Paper: Rank_i = k * (N - i) + 1 for the
        # sorted rank i. With zero-based sorted rank s: k*(n-1-s)+1.
        k = float(self._params.get("rank_greediness", 3.0))
        sorted_ranks = k * (np.arange(n - 1, -1, -1, dtype=float)) + 1.0
        sorted_ranks /= float(np.sum(sorted_ranks)) + _EPS
        weights = np.empty(n, dtype=float)
        weights[order] = sorted_ranks
        return weights

    def _choice_population_index(self, n: int, exclude: set[int], order: np.ndarray, *, rank_based: bool = False) -> int:
        candidates = np.array([idx for idx in range(n) if idx not in exclude], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        if rank_based:
            p = self._rank_probabilities(n, order)[candidates]
            p = p / (float(np.sum(p)) + _EPS)
            return int(np.random.choice(candidates, p=p))
        return int(np.random.choice(candidates))

    def _choose_pbest_index(self, order: np.ndarray, pnum: int, i: int) -> int:
        top = np.asarray(order[:pnum], dtype=int)
        top = top[top != int(i)]
        if top.size == 0:
            top = np.asarray(order[:pnum], dtype=int)
        return int(np.random.choice(top))

    def _choose_r2_vector(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        i: int,
        pbest_idx: int,
        r1_idx: int,
        order: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Select r2 from population with rank pressure or uniformly from archive.

        The LSHADE archive participates in r2 selection as part of P union A;
        when r2 is drawn from the population, the LSHADE-RSP rank probabilities
        are used for the selective-pressure part of Eq. (6).
        """
        n = int(pop.shape[0])
        if archive.shape[0] > 0:
            archive_probability = archive.shape[0] / float(n + archive.shape[0])
            if np.random.random() < archive_probability:
                return archive[int(np.random.randint(archive.shape[0]))].copy(), True
        r2_idx = self._choice_population_index(n, {int(i), int(pbest_idx), int(r1_idx)}, order, rank_based=True)
        return pop[r2_idx, :-1].copy(), False

    def _sample_f_cr(self, M_F: np.ndarray, M_CR: np.ndarray, mem_idx: int, progress: float) -> tuple[float, float]:
        F = _positive_cauchy(float(M_F[mem_idx]), float(self._params.get("f_scale", 0.1)))
        # First 0.6 NFEmax: F is not allowed above 0.7; afterward it remains <=1.
        if progress < 0.6 and F > 0.7:
            F = 0.7
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        # jSO-style high-CR propagation used by LSHADE-RSP.
        if progress < 0.25:
            CR = max(CR, 0.7)
        elif progress < 0.5:
            CR = max(CR, 0.6)
        return float(F), float(CR)

    def _midpoint_repair(self, trial: np.ndarray, parent: np.ndarray) -> np.ndarray:
        repaired = np.asarray(trial, dtype=float).copy()
        parent = np.asarray(parent, dtype=float)
        below = repaired < self._lo
        above = repaired > self._hi
        repaired[below] = 0.5 * (self._lo[below] + parent[below])
        repaired[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(repaired, self._lo, self._hi)

    def _trial_once(
        self,
        parent: np.ndarray,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        F: float,
        CR: float,
        progress: float,
        pnum: int,
    ) -> tuple[np.ndarray, bool]:
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        pbest_idx = self._choose_pbest_index(order, pnum, i)
        r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)}, order, rank_based=True)
        r2, archive_used = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx, order)
        Fw = self._weighted_pbest_factor(F, progress)
        donor = parent + Fw * (pop[pbest_idx, :-1] - parent) + float(F) * (pop[r1_idx, :-1] - r2)
        cross = np.random.rand(dim) < float(CR)
        cross[np.random.randint(dim)] = True
        trial = np.where(cross, donor, parent)
        return trial, bool(archive_used)

    def _make_trial(
        self,
        parent: np.ndarray,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        F: float,
        CR: float,
        progress: float,
        pnum: int,
    ) -> tuple[np.ndarray, int, int]:
        """Construct current-to-pbest/r/bin and apply a bounded repeat mechanism."""
        max_attempts = max(1, int(self._params.get("resampling_attempts", 100)))
        archive_used_total = 0
        last_trial = None
        for attempt in range(max_attempts):
            trial, archive_used = self._trial_once(parent, pop, archive, order, i, F, CR, progress, pnum)
            archive_used_total += int(archive_used)
            last_trial = trial
            if np.all((trial >= self._lo) & (trial <= self._hi)):
                return trial, archive_used_total, attempt
        return self._midpoint_repair(np.asarray(last_trial, dtype=float), parent), archive_used_total, max_attempts

    # ------------------------------------------------------------------
    # Main native step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        if n < 4:
            raise ValueError(f"{self.algorithm_id} requires at least four population members.")

        M_F = np.asarray(state.payload.get("M_F"), dtype=float).copy()
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float).copy()
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim).copy()
        h = int(M_F.shape[0])
        if h < 2:
            raise ValueError("hist_mem_size must be at least 2 because LSHADE-RSP reserves the last memory cell at 0.9.")
        update_h = h - 1
        k_mem = int(state.payload.get("k", 0)) % update_h
        initial_n = int(state.payload.get("initial_n", n))
        reserved = float(self._params.get("reserved_memory_value", 0.9))
        M_F[-1] = reserved
        M_CR[-1] = reserved

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state.evaluations, state.step)
        order = self._order(pop[:, -1])
        pnum = self._pbest_count(n, progress)
        pb_value = self._pb_value(progress)

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        archive_r2_count = 0
        repeat_attempts = 0

        for i in range(n):
            mem_idx = int(np.random.randint(h))
            if mem_idx == h - 1:
                M_F[mem_idx] = reserved
                M_CR[mem_idx] = reserved
            F, CR = self._sample_f_cr(M_F, M_CR, mem_idx, progress)
            trial, used_archive_count, attempt_count = self._make_trial(
                parent=pop[i, :-1],
                pop=pop,
                archive=archive,
                order=order,
                i=i,
                F=F,
                CR=CR,
                progress=progress,
                pnum=pnum,
            )
            trials[i] = trial
            Fs[i] = F
            CRs[i] = CR
            archive_r2_count += int(used_archive_count > 0)
            repeat_attempts += int(attempt_count)

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        if self.problem.objective == "min":
            mask = trial_fit <= pop[:, -1]
            strict_gain = np.maximum(pop[:, -1] - trial_fit, 0.0)
        else:
            mask = trial_fit >= pop[:, -1]
            strict_gain = np.maximum(trial_fit - pop[:, -1], 0.0)
        accepted_gain = float(np.sum(strict_gain[mask]))

        self._add(contrib, counts, "mutation", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(n))
        self._add(contrib, counts, "weighted_pbest_scaling", 0.0, int(n))
        self._add(contrib, counts, "crossover", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(n))
        self._add(contrib, counts, "selection", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(np.count_nonzero(mask)))
        # Both r1 and population-selected r2 are selected via rank probabilities.
        self._add(contrib, counts, "rank_selective_pressure", 0.0, int(2 * n - archive_r2_count))
        self._add(contrib, counts, "bound_resampling", 0.0, int(repeat_attempts))

        strict_mask = mask & (strict_gain > 0.0)
        if np.any(mask):
            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        if np.any(strict_mask):
            old_success = parent_before[strict_mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(strict_mask)))

            sf = Fs[strict_mask]
            scr = CRs[strict_mask]
            df = np.abs(parent_fit_before[strict_mask] - trial_fit[strict_mask])
            weights = df / (float(np.sum(df)) + _EPS)
            new_mf = _weighted_lehmer(sf, weights)
            new_mcr = _weighted_lehmer(scr, weights)
            c = float(self._params.get("memory_update_mixing", 0.5))
            # LSHADE-RSP/iL-SHADE style: average old memory and new Lehmer mean.
            M_F[k_mem] = float(np.clip(c * M_F[k_mem] + (1.0 - c) * new_mf, 0.0, 1.0))
            M_CR[k_mem] = float(np.clip(c * M_CR[k_mem] + (1.0 - c) * new_mcr, 0.0, 1.0))
            k_mem = (k_mem + 1) % update_h
            M_F[-1] = reserved
            M_CR[-1] = reserved
            self._add(contrib, counts, "success_history_update", 0.0, 1)

        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "population_reduction", 0.0, removed)

        # Archive size equals the current population size and is dynamically adjusted.
        max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * pop.shape[0])))
        if archive.shape[0] > max_arc:
            archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "initial_n": initial_n,
            "pb_value": pb_value,
            "last_bound_resampling_attempts": int(repeat_attempts),
            "last_archive_r2_count": int(archive_r2_count),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "direct_improvement"
        obs["evomapx_fidelity"] = "native"
        if "M_F" in state.payload and "M_CR" in state.payload:
            obs["mean_memory_f"] = float(np.nanmean(np.asarray(state.payload["M_F"], dtype=float)))
            obs["mean_memory_cr"] = float(np.nanmean(np.asarray(state.payload["M_CR"], dtype=float)))
            obs["reserved_memory_f"] = float(np.asarray(state.payload["M_F"], dtype=float)[-1])
            obs["reserved_memory_cr"] = float(np.asarray(state.payload["M_CR"], dtype=float)[-1])
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        obs["pb_value"] = float(state.payload.get("pb_value", self._params.get("pbest_base", 0.085)))
        obs["last_bound_resampling_attempts"] = int(state.payload.get("last_bound_resampling_attempts", 0))
        obs["last_archive_r2_count"] = int(state.payload.get("last_archive_r2_count", 0))
        return obs
