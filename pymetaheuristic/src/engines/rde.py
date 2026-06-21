"""pyMetaheuristic src — faithful Reconstructed Differential Evolution Engine.

This module implements the Reconstructed Differential Evolution (RDE) algorithm
of Tao, Zhao, Wang, and Gao (arXiv:2404.16280, 2024).  The implementation is a
native NumPy port of the mechanisms described in the paper: the hybrid resource
allocation between DE/current-to-pbest/1 and DE/current-to-order-pbest/1,
extended rank-based selective pressure for pbest/r1/r2 and archive sampling,
jSO-style success-history parameter adaptation, early F/CR restrictions, linear
p reduction, linear population size reduction, an external archive, and Cauchy
perturbation during crossover.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """Cauchy sample for F: resample while negative, then cap at ``upper``."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value >= 0.0:
            return min(value, upper)
    return min(max(center, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian crossover-rate sample clipped to [0, 1]."""
    if not np.isfinite(center):
        center = 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _cauchy_around(center: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Componentwise Cauchy perturbation centered on a target vector."""
    return np.asarray(center, dtype=float) + np.random.standard_cauchy(np.asarray(center).shape) * float(scale)


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray, fallback: float = 0.5) -> float:
    """Weighted Lehmer mean used in SHADE-family success-history updates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float(fallback)
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


class RDEEngine(PortedPopulationEngine):
    """Reconstructed DE with two mutation strategies and extended RSP."""

    algorithm_id = "rde"
    algorithm_name = "Reconstructed Differential Evolution"
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
        "doi": "10.48550/arXiv.2404.16280",
        "title": "An Efficient Reconstructed Differential Evolution Variant by Some of the Current State-of-the-art Strategies for Solving Single Objective Bound Constrained Problems",
        "authors": "Sichen Tao, Ruihan Zhao, Kaiyu Wang, Shangce Gao",
        "year": 2024,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults.  None/0/"paper" activates NPmax = 18D.
        population_size=None,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.3,
        memory_cr_init=0.8,
        memory_reserved_value=0.9,
        pmax=0.25,
        rank_greediness=3.0,
        archive_rate=1.0,
        gamma_order_init=0.5,
        cauchy_perturbation_rate=0.2,
        cauchy_perturbation_scale=0.1,
        f_scale=0.1,
        cr_sigma=0.1,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            self._n = max(4, 18 * int(self.problem.dimension))
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
            f"{self.algorithm_id}.mutation_current_to_pbest",
            f"{self.algorithm_id}.mutation_current_to_order_pbest",
            f"{self.algorithm_id}.strategy_resource_allocation",
            f"{self.algorithm_id}.extended_rank_selective_pressure",
            f"{self.algorithm_id}.crossover",
            f"{self.algorithm_id}.cauchy_target_perturbation",
            f"{self.algorithm_id}.selection",
            f"{self.algorithm_id}.archive_update",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.linear_population_reduction",
            f"{self.algorithm_id}.bound_repair",
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
        reserved = float(self._params.get("memory_reserved_value", 0.9))
        m_f = np.full(h, float(self._params.get("memory_f_init", 0.3)), dtype=float)
        m_cr = np.full(h, float(self._params.get("memory_cr_init", 0.8)), dtype=float)
        m_f[-1] = reserved
        m_cr[-1] = reserved
        gamma = float(self._params.get("gamma_order_init", 0.5))
        gamma = float(np.clip(gamma, 0.0, 1.0))
        return {
            "M_F": m_f,
            "M_CR": m_cr,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension + 1), dtype=float),
            "initial_n": int(pop.shape[0]),
            "gamma_order": gamma,
            "gamma_current": 1.0 - gamma,
            "p_value": float(self._params.get("pmax", 0.25)),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _p_value(self, progress: float) -> float:
        # Eq. (20): p(k) = pmax * (1 - 0.5 * nfes / max_nfes).
        pmax = float(self._params.get("pmax", 0.25))
        return float(np.clip(pmax * (1.0 - 0.5 * float(progress)), 1.0e-12, 1.0))

    def _pbest_count(self, n: int, progress: float) -> int:
        return int(np.clip(np.ceil(self._p_value(progress) * n), 2, max(2, n)))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step)
        target = round((min_n - initial_n) * progress + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    def _archive_capacity(self, n: int) -> int:
        return max(1, int(round(float(self._params.get("archive_rate", 1.0)) * int(n))))

    # ------------------------------------------------------------------
    # Selection primitives
    # ------------------------------------------------------------------
    def _rank_weights_from_fitness(self, fitness: np.ndarray) -> np.ndarray:
        """RDE extended RSP weights: Rank_i = kr * (N - rank_position) + 1."""
        fitness = np.asarray(fitness, dtype=float)
        n = int(fitness.size)
        if n <= 0:
            return np.array([], dtype=float)
        order = self._order(fitness)
        kr = float(self._params.get("rank_greediness", 3.0))
        ranked = np.arange(n, 0, -1, dtype=float)
        sorted_weights = kr * ranked + 1.0
        weights = np.empty(n, dtype=float)
        weights[order] = sorted_weights
        weights = np.maximum(weights, _EPS)
        return weights / (float(np.sum(weights)) + _EPS)

    def _choice_weighted_index(self, fitness: np.ndarray, exclude: set[int] | None = None) -> int:
        n = int(len(fitness))
        if n <= 0:
            raise ValueError("Cannot select from an empty set.")
        exclude = exclude or set()
        candidates = np.array([idx for idx in range(n) if idx not in exclude], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        probs = self._rank_weights_from_fitness(np.asarray(fitness, dtype=float))[candidates]
        probs = probs / (float(np.sum(probs)) + _EPS)
        return int(np.random.choice(candidates, p=probs))

    def _choose_pbest_index(self, pop: np.ndarray, pnum: int, i: int) -> int:
        order = self._order(pop[:, -1])
        top = np.asarray(order[:pnum], dtype=int)
        top = top[top != int(i)]
        if top.size == 0:
            top = np.asarray(order[:pnum], dtype=int)
        # Extended RSP also affects pbest selection.
        top_fit = pop[top, -1]
        probs = self._rank_weights_from_fitness(top_fit)
        probs = probs / (float(np.sum(probs)) + _EPS)
        return int(np.random.choice(top, p=probs))

    def _combined_rows_for_r2(self, pop: np.ndarray, archive: np.ndarray) -> np.ndarray:
        if archive.size:
            return np.vstack((pop, archive))
        return pop.copy()

    def _choose_r2_combined(self, combined: np.ndarray, pop_size: int, exclude_pop: set[int]) -> tuple[np.ndarray, float, bool]:
        candidates = np.array(
            [idx for idx in range(combined.shape[0]) if not (idx < pop_size and idx in exclude_pop)],
            dtype=int,
        )
        if candidates.size == 0:
            candidates = np.arange(combined.shape[0], dtype=int)
        probs_all = self._rank_weights_from_fitness(combined[:, -1])
        probs = probs_all[candidates]
        probs = probs / (float(np.sum(probs)) + _EPS)
        idx = int(np.random.choice(candidates, p=probs))
        return combined[idx, :-1].copy(), float(combined[idx, -1]), bool(idx >= pop_size)

    def _select_three_for_order(self, pop: np.ndarray, archive: np.ndarray, pnum: int, i: int) -> tuple[list[np.ndarray], list[float], bool]:
        pbest_idx = self._choose_pbest_index(pop, pnum, i)
        r1_idx = self._choice_weighted_index(pop[:, -1], {int(i), int(pbest_idx)})
        combined = self._combined_rows_for_r2(pop, archive)
        r2_vec, r2_fit, r2_from_archive = self._choose_r2_combined(
            combined,
            pop.shape[0],
            {int(i), int(pbest_idx), int(r1_idx)},
        )
        return (
            [pop[pbest_idx, :-1].copy(), pop[r1_idx, :-1].copy(), r2_vec],
            [float(pop[pbest_idx, -1]), float(pop[r1_idx, -1]), float(r2_fit)],
            r2_from_archive,
        )

    # ------------------------------------------------------------------
    # Operators and repair
    # ------------------------------------------------------------------
    def _current_to_pbest_donor(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        i: int,
        pnum: int,
        F: float,
    ) -> tuple[np.ndarray, bool]:
        pbest_idx = self._choose_pbest_index(pop, pnum, i)
        r1_idx = self._choice_weighted_index(pop[:, -1], {int(i), int(pbest_idx)})
        combined = self._combined_rows_for_r2(pop, archive)
        r2_vec, _, r2_from_archive = self._choose_r2_combined(
            combined,
            pop.shape[0],
            {int(i), int(pbest_idx), int(r1_idx)},
        )
        donor = pop[i, :-1] + float(F) * (pop[pbest_idx, :-1] - pop[i, :-1]) + float(F) * (pop[r1_idx, :-1] - r2_vec)
        return donor, r2_from_archive

    def _current_to_order_pbest_donor(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        i: int,
        pnum: int,
        F: float,
    ) -> tuple[np.ndarray, bool]:
        vecs, fits, r2_from_archive = self._select_three_for_order(pop, archive, pnum, i)
        if self.problem.objective == "min":
            order = np.argsort(np.asarray(fits, dtype=float))
        else:
            order = np.argsort(np.asarray(fits, dtype=float))[::-1]
        x_best = vecs[int(order[0])]
        x_median = vecs[int(order[1])]
        x_worst = vecs[int(order[2])]
        donor = pop[i, :-1] + float(F) * (x_best - pop[i, :-1]) + float(F) * (x_median - x_worst)
        return donor, r2_from_archive

    def _crossover_with_cauchy_perturbation(self, parent: np.ndarray, donor: np.ndarray, cr: float) -> tuple[np.ndarray, int]:
        dim = int(self.problem.dimension)
        mask = np.random.rand(dim) < float(cr)
        mask[np.random.randint(dim)] = True
        trial = parent.copy()
        trial[mask] = donor[mask]
        not_from_donor = ~mask
        pr = float(self._params.get("cauchy_perturbation_rate", 0.2))
        perturb_mask = not_from_donor & (np.random.rand(dim) < pr)
        if np.any(perturb_mask):
            scale = float(self._params.get("cauchy_perturbation_scale", 0.1))
            trial[perturb_mask] = _cauchy_around(parent[perturb_mask], scale=scale)
        return trial, int(np.count_nonzero(perturb_mask))

    def _random_bound_repair(self, trial: np.ndarray) -> tuple[np.ndarray, int]:
        trial = np.asarray(trial, dtype=float).copy()
        bad = (trial < self._lo) | (trial > self._hi)
        count = int(np.count_nonzero(bad))
        if count:
            trial[bad] = np.random.uniform(self._lo[bad], self._hi[bad])
        return trial, count

    def _archive_insert(self, archive: np.ndarray, row: np.ndarray, capacity: int) -> np.ndarray:
        capacity = max(1, int(capacity))
        row = np.asarray(row, dtype=float).reshape(1, -1)
        if archive.shape[0] < capacity:
            return np.vstack((archive, row)) if archive.size else row.copy()
        archive = archive.copy()
        archive[int(np.random.randint(archive.shape[0])), :] = row[0]
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
        h = int(M_F.shape[0])
        k_mem = int(state.payload.get("k", 0)) % h
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim + 1))), dtype=float).reshape(-1, dim + 1).copy()
        initial_n = int(state.payload.get("initial_n", n))
        gamma_order = float(np.clip(state.payload.get("gamma_order", 0.5), 0.0, 1.0))

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state.evaluations, state.step)
        p_value = self._p_value(progress)
        pnum = self._pbest_count(n, progress)

        # RDE paper uses a generation memory index h=mod(k/H), with the last
        # memory cell reserved at 0.9.  F is sampled from Cauchy(MF,0.1) and
        # CR from Normal(MCR,0.1), consistent with the jSO/iL-SHADE rule that
        # the paper says it adopts.
        mem_idx = k_mem
        reserved_idx = h - 1
        reserved_value = float(self._params.get("memory_reserved_value", 0.9))
        M_F[reserved_idx] = reserved_value
        M_CR[reserved_idx] = reserved_value
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        for i in range(n):
            F = _positive_cauchy(M_F[mem_idx], float(self._params.get("f_scale", 0.1)))
            CR = _normal_cr(M_CR[mem_idx], float(self._params.get("cr_sigma", 0.1)))
            if progress < 0.60 and F > 0.70:
                F = 0.70
            if progress < 0.25 and CR < 0.70:
                CR = 0.70
            elif progress < 0.50 and CR < 0.60:
                CR = 0.60
            Fs[i] = F
            CRs[i] = CR

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        use_order = np.random.rand(n) < gamma_order
        perturb_count = 0
        repair_count = 0
        archive_r2_count = 0

        self._add(contrib, counts, "strategy_resource_allocation", 0.0, n)
        # Extended RSP participates in pbest/r1/r2 for both strategies and also
        # in archive sampling, so every mutation attempt is counted here.
        self._add(contrib, counts, "extended_rank_selective_pressure", 0.0, n)

        for i in range(n):
            if use_order[i]:
                donor, r2_from_archive = self._current_to_order_pbest_donor(pop, archive, i, pnum, float(Fs[i]))
            else:
                donor, r2_from_archive = self._current_to_pbest_donor(pop, archive, i, pnum, float(Fs[i]))
            archive_r2_count += int(r2_from_archive)
            trial, n_perturb = self._crossover_with_cauchy_perturbation(pop[i, :-1], donor, float(CRs[i]))
            perturb_count += n_perturb
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

        order_count = int(np.count_nonzero(use_order))
        current_count = int(n - order_count)
        gain_order = float(np.sum(positive_gains[use_order]))
        gain_current = float(np.sum(positive_gains[~use_order]))

        # Direct-improvement telemetry. Attribution is approximate but counts
        # expose all paper mechanisms.
        self._add(contrib, counts, "mutation_current_to_order_pbest", gain_order / 3.0, order_count)
        self._add(contrib, counts, "mutation_current_to_pbest", gain_current / 3.0, current_count)
        self._add(contrib, counts, "crossover", accepted_gain / 3.0, n)
        self._add(contrib, counts, "selection", accepted_gain / 3.0, int(np.count_nonzero(mask)))
        self._add(contrib, counts, "cauchy_target_perturbation", 0.0, perturb_count)
        self._add(contrib, counts, "bound_repair", 0.0, repair_count)

        if np.any(mask):
            capacity = self._archive_capacity(n)
            for old_x, old_f in zip(parent_before[mask], parent_fit_before[mask]):
                archive = self._archive_insert(archive, np.r_[old_x, old_f], capacity)
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            sf = Fs[mask]
            scr = CRs[mask]
            df = np.abs(parent_fit_before[mask] - trial_fit[mask])
            weights = df / (float(np.sum(df)) + _EPS)
            new_f = _weighted_lehmer(sf, weights, fallback=float(M_F[mem_idx]))
            new_cr = _weighted_lehmer(scr, weights, fallback=float(M_CR[mem_idx]))
            if mem_idx != reserved_idx:
                M_F[mem_idx] = float(np.clip(new_f, 0.0, 1.0))
                M_CR[mem_idx] = float(np.clip(new_cr, 0.0, 1.0))
                self._add(contrib, counts, "success_history_update", 0.0, 1)
            M_F[reserved_idx] = reserved_value
            M_CR[reserved_idx] = reserved_value

            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        # Eq. (6)-(8): allocate next-generation resources according to average
        # fitness progress of each mutation strategy, fallback to 0.5/0.5.
        avg_order = gain_order / max(1, order_count)
        avg_current = gain_current / max(1, current_count)
        if avg_order + avg_current > _EPS:
            gamma_order = avg_order / (avg_order + avg_current)
        else:
            gamma_order = 0.5
        gamma_order = float(np.clip(gamma_order, 0.0, 1.0))

        k_mem = (k_mem + 1) % h

        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0])
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "linear_population_reduction", 0.0, removed)

        archive = self._truncate_archive(archive, self._archive_capacity(pop.shape[0]))

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "initial_n": initial_n,
            "gamma_order": gamma_order,
            "gamma_current": 1.0 - gamma_order,
            "p_value": p_value,
            "archive_r2_count": archive_r2_count,
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
        obs["gamma_order"] = float(state.payload.get("gamma_order", self._params.get("gamma_order_init", 0.5)))
        obs["gamma_current"] = float(state.payload.get("gamma_current", 1.0 - obs["gamma_order"]))
        obs["p_value"] = float(state.payload.get("p_value", self._params.get("pmax", 0.25)))
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        obs["archive_r2_count"] = int(state.payload.get("archive_r2_count", 0))
        return obs
