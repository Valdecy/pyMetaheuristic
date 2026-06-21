"""pyMetaheuristic src — faithful jSO Differential Evolution Engine.

This module implements the jSO algorithm from Brest, Sepesy Maucec, and
Boskovic (CEC 2017).  jSO is an iL-SHADE descendant with the weighted
current-to-pBest-w/1/bin mutation, dimension-dependent initial population,
H=5 success-history memories, high-CR propagation, early caps/floors for F/CR,
linear population-size reduction, a linearly decreasing p-best pool, and the
L-SHADE/iL-SHADE external archive.
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


class JSODEEngine(PortedPopulationEngine):
    """jSO: paper-faithful improved iL-SHADE with weighted p-best mutation."""

    algorithm_id = "jso_de"
    algorithm_name = "jSO Differential Evolution"
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
        "doi": "10.1109/CEC.2017.7969362",
        "title": "Single Objective Real-Parameter Optimization: Algorithm jSO",
        "authors": "Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic",
        "year": 2017,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults are dimension-dependent. None/"paper"/0 means:
        # NP_init = round(25 * log(D) * sqrt(D)).
        population_size=None,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.3,
        memory_cr_init=0.8,
        reserved_memory_value=0.9,
        # jSO keeps L-SHADE/iL-SHADE archive machinery; 1.4 is the common
        # jSO source-code/archive-size setting and remains user-configurable.
        archive_rate=1.4,
        pbest_max=0.25,
        pbest_min=None,  # None means pbest_max / 2.
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
            self._n = max(4, int(round(25.0 * np.log(float(d)) * np.sqrt(float(d)))))
            self._params["population_size"] = self._n
        else:
            self._n = max(4, int(pop_param))
            self._params["population_size"] = self._n
        if self._params.get("pbest_min", None) is None:
            self._params["pbest_min"] = 0.5 * float(self._params.get("pbest_max", 0.25))
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
        # Algorithm 1 lines 10--13: if the sampled index is H, MF_H=MCR_H=0.9.
        # We reserve that last cell permanently, as in iL-SHADE/jSO practice.
        m_f[-1] = reserved
        m_cr[-1] = reserved
        return {
            "M_F": m_f,
            "M_CR": m_cr,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
            "p_value": float(self._params.get("pbest_max", 0.25)),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _p_value(self, progress: float) -> float:
        # jSO: p linearly decreases from pmax=0.25 to pmin=pmax/2.
        pmax = float(self._params.get("pbest_max", 0.25))
        pmin = float(self._params.get("pbest_min", 0.125))
        return float(np.clip(pmax + (pmin - pmax) * float(progress), pmin, pmax))

    def _pbest_count(self, n: int, progress: float) -> int:
        p = self._p_value(progress)
        return int(np.clip(round(p * n), 2, max(2, n)))

    def _weighted_pbest_factor(self, F: float, progress: float) -> float:
        # Eq. (4) in the jSO paper.
        if progress < 0.2:
            return 0.7 * float(F)
        if progress < 0.4:
            return 0.8 * float(F)
        return 1.2 * float(F)

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        target = round(initial_n - progress * (initial_n - min_n))
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------
    def _choice_population_index(self, n: int, exclude: set[int]) -> int:
        candidates = np.array([idx for idx in range(n) if idx not in exclude], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
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
    ) -> tuple[np.ndarray, bool]:
        """Select r2 from P union A using L-SHADE-style archive participation."""
        n = int(pop.shape[0])
        if archive.shape[0] > 0:
            archive_probability = archive.shape[0] / float(n + archive.shape[0])
            if np.random.random() < archive_probability:
                return archive[int(np.random.randint(archive.shape[0]))].copy(), True
        r2_idx = self._choice_population_index(n, {int(i), int(pbest_idx), int(r1_idx)})
        return pop[r2_idx, :-1].copy(), False

    def _sample_f_cr(self, M_F: np.ndarray, M_CR: np.ndarray, mem_idx: int, progress: float) -> tuple[float, float]:
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        # Algorithm 1: high CR floors in the first half of the run.
        if progress < 0.25:
            CR = max(CR, 0.7)
        elif progress < 0.5:
            CR = max(CR, 0.6)
        F = _positive_cauchy(float(M_F[mem_idx]), float(self._params.get("f_scale", 0.1)))
        # Algorithm 1: do not allow very high F early.
        if progress < 0.6 and F > 0.7:
            F = 0.7
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
        r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)})
        r2, archive_used = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx)
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
        """Construct current-to-pBest-w/1/bin and apply jSO's repeat mechanism.

        The paper states that a repeat mechanism is applied when trial variables
        leave the bounds.  We retry donor/crossover generation with the same F/CR
        up to ``resampling_attempts`` times, then use midpoint repair as a safe
        deterministic fallback so the engine cannot loop forever.
        """
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
            raise ValueError("hist_mem_size must be at least 2 because jSO reserves the last memory cell at 0.9.")
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
        p_value = self._p_value(progress)

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        archive_r2_count = 0
        repeat_attempts = 0

        for i in range(n):
            mem_idx = int(np.random.randint(h))
            # The reserved last index always holds high F/CR values.
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
        self._add(contrib, counts, "bound_resampling", 0.0, int(repeat_attempts))

        if np.any(mask):
            old_success = parent_before[mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.4)) * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            # jSO updates memories only from strictly improved candidates.
            strict_success = strict_gain[mask] > 0.0
            if np.any(strict_success):
                sf = Fs[mask][strict_success]
                scr = CRs[mask][strict_success]
                df = np.abs(parent_fit_before[mask][strict_success] - trial_fit[mask][strict_success])
                weights = df / (float(np.sum(df)) + _EPS)
                new_mf = _weighted_lehmer(sf, weights)
                new_mcr = _weighted_lehmer(scr, weights)
                c = float(self._params.get("memory_update_mixing", 0.5))
                # iL-SHADE/jSO memory update: old value and new Lehmer mean are
                # weighted equally by default.
                M_F[k_mem] = float(np.clip(c * M_F[k_mem] + (1.0 - c) * new_mf, 0.0, 1.0))
                M_CR[k_mem] = float(np.clip(c * M_CR[k_mem] + (1.0 - c) * new_mcr, 0.0, 1.0))
                k_mem = (k_mem + 1) % update_h
                M_F[-1] = reserved
                M_CR[-1] = reserved
                self._add(contrib, counts, "success_history_update", 0.0, 1)

            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "population_reduction", 0.0, removed)

        max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.4)) * pop.shape[0])))
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
            "p_value": p_value,
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
        obs["p_value"] = float(state.payload.get("p_value", self._params.get("pbest_max", 0.25)))
        obs["last_bound_resampling_attempts"] = int(state.payload.get("last_bound_resampling_attempts", 0))
        obs["last_archive_r2_count"] = int(state.payload.get("last_archive_r2_count", 0))
        return obs
