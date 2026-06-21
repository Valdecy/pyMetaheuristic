"""pyMetaheuristic src — faithful NL-SHADE-RSP Engine.

This module implements the NL-SHADE-RSP algorithm proposed by Stanovov,
Akhmedova, and Semenkin (CEC 2021). The implementation is a native NumPy port
of the paper mechanics: NLPSR, current-to-pbest/1 with rank-based selective
pressure on population-selected r2, adaptive archive-use probability, mixed
binomial/exponential crossover with distinct CR control, sorted exponential CRs,
SHADE-style success-history adaptation, and random bound repair.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """SHADE-style Cauchy sample for F: resample while F < 0, then cap at upper."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value >= 0.0:
            return min(value, upper)
    return min(max(center, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian CR sample clipped to [0, 1]."""
    if not np.isfinite(center):
        center = 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used by L-SHADE/SHADE success-history updates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


class NLSHADERspEngine(PortedPopulationEngine):
    """NL-SHADE-RSP with adaptive archive usage and rank-selective r2 pressure."""

    algorithm_id = "nlshade_rsp"
    algorithm_name = "NL-SHADE-RSP"
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
        "doi": "10.1109/CEC45853.2021.9504959",
        "title": "NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2021,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults are dimension-dependent. None/0/"paper" activates NPmax = 30D and H = 20D.
        population_size=None,
        min_population_size=4,
        hist_mem_size=None,
        memory_f_init=0.2,
        memory_cr_init=0.2,
        memory_update_mixing=0.5,
        archive_rate=2.1,
        archive_use_probability=0.5,
        archive_probability_min=0.1,
        archive_probability_max=0.9,
        pbest_max=0.4,
        pbest_min=0.2,
        f_scale=0.1,
        cr_sigma=0.1,
        rank_pressure_coefficient=1.0,
        binomial_crossover_probability=0.5,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            self._n = max(4, 30 * int(self.problem.dimension))
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
            f"{self.algorithm_id}.crossover_binomial",
            f"{self.algorithm_id}.crossover_exponential",
            f"{self.algorithm_id}.crossover_rate_sorting",
            f"{self.algorithm_id}.selection",
            f"{self.algorithm_id}.archive_update",
            f"{self.algorithm_id}.adaptive_archive_probability",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.nonlinear_population_reduction",
            f"{self.algorithm_id}.rank_selective_pressure",
            f"{self.algorithm_id}.bound_random_repair",
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
        h_param = self._params.get("hist_mem_size", None)
        if h_param in (None, 0, "paper", "auto"):
            h = max(1, 20 * int(self.problem.dimension))
            self._params["hist_mem_size"] = h
        else:
            h = max(1, int(h_param))
            self._params["hist_mem_size"] = h
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.2)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.2)), dtype=float),
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
            "p_archive": float(self._params.get("archive_use_probability", 0.5)),
            "pbest_value": float(self._params.get("pbest_max", 0.4)),
            "binomial_cr": 0.0,
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _pbest_value(self, progress: float) -> float:
        pmax = float(self._params.get("pbest_max", 0.4))
        pmin = float(self._params.get("pbest_min", 0.2))
        return float(np.clip(pmax + (pmin - pmax) * float(progress), 1.0e-12, 1.0))

    def _pbest_count(self, n: int, progress: float) -> int:
        return int(np.clip(round(self._pbest_value(progress) * n), 2, max(2, n)))

    def _binomial_cr(self, progress: float) -> float:
        # Eq. (15): CRb=0 during the first half, then linearly increases to 1.
        if progress < 0.5:
            return 0.0
        return float(np.clip(2.0 * (float(progress) - 0.5), 0.0, 1.0))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        if progress <= 0.0:
            target = initial_n
        else:
            # NLPSR borrowed from AGSK: (NPmin-NPmax)*NFE_r^(1-NFE_r)+NPmax.
            target = round((min_n - initial_n) * (progress ** (1.0 - progress)) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    def _archive_capacity(self, n: int) -> int:
        return max(1, int(round(float(self._params.get("archive_rate", 2.1)) * int(n))))

    # ------------------------------------------------------------------
    # Selection primitives
    # ------------------------------------------------------------------
    def _rank_probabilities(self, n: int, order: np.ndarray) -> np.ndarray:
        # The NL-SHADE-RSP paper applies RSP only to population-selected r2.
        # It describes the rank weights as R_i = exp(-i/NP), with i sorted by fitness.
        coeff = float(self._params.get("rank_pressure_coefficient", 1.0))
        sorted_positions = np.arange(1, n + 1, dtype=float)
        sorted_weights = np.exp(-coeff * sorted_positions / float(max(1, n)))
        sorted_weights /= float(np.sum(sorted_weights)) + _EPS
        weights = np.empty(n, dtype=float)
        weights[order] = sorted_weights
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
        p_archive: float,
    ) -> tuple[np.ndarray, bool]:
        n = int(pop.shape[0])
        if archive.shape[0] > 0 and np.random.random() < float(p_archive):
            return archive[int(np.random.randint(archive.shape[0]))].copy(), True
        r2_idx = self._choice_population_index(n, {int(i), int(pbest_idx), int(r1_idx)}, order, rank_based=True)
        return pop[r2_idx, :-1].copy(), False

    # ------------------------------------------------------------------
    # Crossover and repair
    # ------------------------------------------------------------------
    def _binomial_crossover(self, parent: np.ndarray, donor: np.ndarray, cr: float) -> np.ndarray:
        dim = int(self.problem.dimension)
        mask = np.random.rand(dim) < float(cr)
        mask[np.random.randint(dim)] = True
        return np.where(mask, donor, parent)

    def _exponential_crossover(self, parent: np.ndarray, donor: np.ndarray, cr: float) -> np.ndarray:
        dim = int(self.problem.dimension)
        start = int(np.random.randint(dim))
        length = 1
        while length < dim and np.random.random() < float(cr):
            length += 1
        mask = np.zeros(dim, dtype=bool)
        for offset in range(length):
            mask[(start + offset) % dim] = True
        return np.where(mask, donor, parent)

    def _random_bound_repair(self, trial: np.ndarray) -> tuple[np.ndarray, int]:
        trial = np.asarray(trial, dtype=float).copy()
        bad = (trial < self._lo) | (trial > self._hi)
        count = int(np.count_nonzero(bad))
        if count:
            trial[bad] = np.random.uniform(self._lo[bad], self._hi[bad])
        return trial, count

    def _archive_insert(self, archive: np.ndarray, vector: np.ndarray, capacity: int) -> np.ndarray:
        capacity = max(1, int(capacity))
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        if archive.shape[0] < capacity:
            return np.vstack((archive, vector)) if archive.size else vector.copy()
        archive = archive.copy()
        archive[int(np.random.randint(archive.shape[0])), :] = vector[0]
        return archive

    def _truncate_archive(self, archive: np.ndarray, capacity: int) -> np.ndarray:
        capacity = max(1, int(capacity))
        if archive.shape[0] <= capacity:
            return archive
        keep = np.random.choice(archive.shape[0], capacity, replace=False)
        return archive[keep].copy()

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
        k_mem = int(state.payload.get("k", 0)) % h
        initial_n = int(state.payload.get("initial_n", n))
        p_archive = float(state.payload.get("p_archive", self._params.get("archive_use_probability", 0.5)))
        p_archive = float(np.clip(
            p_archive,
            float(self._params.get("archive_probability_min", 0.1)),
            float(self._params.get("archive_probability_max", 0.9)),
        ))

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state.evaluations, state.step)
        order = self._order(pop[:, -1])
        pnum = self._pbest_count(n, progress)
        pbest_value = self._pbest_value(progress)
        cr_bin = self._binomial_cr(progress)

        mem_indices = np.random.randint(0, h, size=n)
        Fs = np.array([_positive_cauchy(M_F[idx], float(self._params.get("f_scale", 0.1))) for idx in mem_indices], dtype=float)
        raw_cr = np.array([_normal_cr(M_CR[idx], float(self._params.get("cr_sigma", 0.1))) for idx in mem_indices], dtype=float)
        CRs = raw_cr.copy()
        # Smaller exponential CR values go to better individuals, larger to worse ones.
        CRs[order] = np.sort(raw_cr)
        self._add(contrib, counts, "crossover_rate_sorting", 0.0, n)

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        used_archive = np.zeros(n, dtype=bool)
        used_binomial = np.zeros(n, dtype=bool)
        used_exponential = np.zeros(n, dtype=bool)
        repair_count = 0

        binomial_probability = float(self._params.get("binomial_crossover_probability", 0.5))
        for i in range(n):
            pbest_idx = self._choose_pbest_index(order, pnum, i)
            r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)}, order, rank_based=False)
            r2_vec, archive_flag = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx, order, p_archive)
            used_archive[i] = bool(archive_flag)
            F = float(Fs[i])
            donor = pop[i, :-1] + F * (pop[pbest_idx, :-1] - pop[i, :-1]) + F * (pop[r1_idx, :-1] - r2_vec)

            if np.random.random() < binomial_probability:
                trial = self._binomial_crossover(pop[i, :-1], donor, cr_bin)
                used_binomial[i] = True
            else:
                trial = self._exponential_crossover(pop[i, :-1], donor, float(CRs[i]))
                used_exponential[i] = True
            trial, fixed = self._random_bound_repair(trial)
            repair_count += fixed
            trials[i] = trial

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        mask = self._better_mask(trial_fit, pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - trial_fit[mask]
        else:
            gains[mask] = trial_fit[mask] - pop[mask, -1]
        positive_gains = np.maximum(gains, 0.0)
        accepted_gain = float(np.sum(positive_gains))

        # Direct-improvement telemetry. Control mechanisms receive counts even if
        # their direct one-step attribution is delayed.
        mutation_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        crossover_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        selection_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        self._add(contrib, counts, "mutation", mutation_gain, n)
        self._add(contrib, counts, "crossover_binomial", crossover_gain * (np.count_nonzero(used_binomial) / max(1, n)), int(np.count_nonzero(used_binomial)))
        self._add(contrib, counts, "crossover_exponential", crossover_gain * (np.count_nonzero(used_exponential) / max(1, n)), int(np.count_nonzero(used_exponential)))
        self._add(contrib, counts, "selection", selection_gain, int(np.count_nonzero(mask)))
        self._add(contrib, counts, "rank_selective_pressure", 0.0, int(np.count_nonzero(~used_archive)))
        self._add(contrib, counts, "bound_random_repair", 0.0, repair_count)

        if np.any(mask):
            capacity = self._archive_capacity(n)
            for old_x in parent_before[mask]:
                archive = self._archive_insert(archive, old_x, capacity)
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            sf = Fs[mask]
            scr = CRs[mask]
            df = np.abs(parent_fit_before[mask] - trial_fit[mask])
            weights = df / (float(np.sum(df)) + _EPS)
            c = float(self._params.get("memory_update_mixing", 0.5))
            new_f = _weighted_lehmer(sf, weights)
            new_cr = _weighted_lehmer(scr, weights)
            M_F[k_mem] = float(np.clip(c * M_F[k_mem] + (1.0 - c) * new_f, 0.0, 1.0))
            M_CR[k_mem] = float(np.clip(c * M_CR[k_mem] + (1.0 - c) * new_cr, 0.0, 1.0))
            k_mem = (k_mem + 1) % h
            self._add(contrib, counts, "success_history_update", 0.0, 1)

            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        # Adaptive archive probability: compare mean improvement of offspring
        # generated with archive r2 and with population-selected r2.
        n_archive = int(np.count_nonzero(used_archive))
        n_population = int(n - n_archive)
        avg_archive = float(np.sum(positive_gains[used_archive]) / max(1, n_archive))
        avg_population = float(np.sum(positive_gains[~used_archive]) / max(1, n_population))
        if avg_archive + avg_population > _EPS:
            p_archive = avg_archive / (avg_archive + avg_population)
            p_archive = float(np.clip(
                p_archive,
                float(self._params.get("archive_probability_min", 0.1)),
                float(self._params.get("archive_probability_max", 0.9)),
            ))
            self._add(contrib, counts, "adaptive_archive_probability", 0.0, 1)

        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "nonlinear_population_reduction", 0.0, removed)

        archive = self._truncate_archive(archive, self._archive_capacity(pop.shape[0]))

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "initial_n": initial_n,
            "p_archive": p_archive,
            "pbest_value": pbest_value,
            "binomial_cr": cr_bin,
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
        obs["p_archive"] = float(state.payload.get("p_archive", self._params.get("archive_use_probability", 0.5)))
        obs["pbest_value"] = float(state.payload.get("pbest_value", self._params.get("pbest_max", 0.4)))
        obs["binomial_cr"] = float(state.payload.get("binomial_cr", 0.0))
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        return obs
