"""pyMetaheuristic src — RDEx-SOP Engine.

RDEx-SOP is an exploitation-biased success-history Differential Evolution
variant for fixed-budget bound-constrained single-objective optimization.  This
implementation follows Tao et al. (arXiv:2603.27089, 2026): success-rate guided
pbest pressure, two mutation branches, adaptive exploitation-biased branch
allocation, success-history F/CR memories, Cauchy local perturbation, greedy
selection, and linear population size reduction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray, fallback: float = 0.5) -> float:
    """Weighted Lehmer mean used by SHADE-family parameter memories."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float(fallback)
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


def _positive_cauchy(center: float, scale: float = 0.1, *, fallback: float = 0.5, upper: float = 1.0) -> float:
    """Sample a positive Cauchy-distributed scaling factor and cap it at upper."""
    if not np.isfinite(center) or center <= 0.0:
        center = float(fallback)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * float(scale) + float(center))
        if value > 0.0 and np.isfinite(value):
            return min(value, float(upper))
    return float(np.clip(center, 1.0e-12, upper))


def _truncated_normal(center: float, sigma: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Gaussian sample clipped to the requested bounded interval."""
    if not np.isfinite(center):
        center = 0.5 * (float(lower) + float(upper))
    value = float(np.random.normal(float(center), float(sigma)))
    return float(np.clip(value, float(lower), float(upper)))


class RDEXSOPEngine(PortedPopulationEngine):
    """RDEx-SOP: exploitation-biased reconstructed Differential Evolution."""

    algorithm_id = "rdex_sop"
    algorithm_name = "RDEx-SOP"
    family = "evolutionary"
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _REFERENCE = {
        "doi": "10.48550/arXiv.2603.27089",
        "title": "RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution for Fixed-Budget Bound-Constrained Single-Objective Optimization",
        "authors": "Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, Shangce Gao",
        "year": 2026,
    }
    _DEFAULTS = dict(
        population_size=600,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        initial_hybrid_rate=0.7,
        local_perturbation_rate=0.1,
        local_perturbation_scale=0.1,
        standard_f_sigma=0.02,
        eb_f_scale=0.1,
        cr_sigma=0.1,
        xi=0.7,
        k_success=7.0,
        eb_fallback=0.5,
        early_cr_fraction=0.25,
        early_cr_lower_bound=0.7,
        budget_aware_population=True,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        requested_n = max(4, int(self._params.get("population_size", 600)))
        if bool(self._params.get("budget_aware_population", True)) and config.max_evaluations is not None and config.max_evaluations > 0:
            # Fixed-budget competitions count the initial front in MaxFE.  When a
            # tiny budget is supplied by a smoke test or tutorial, shrink N0 so
            # at least one native generation can be executed.  With the paper
            # budget this leaves the default N0=600 unchanged.
            max_initial = max(4, int(config.max_evaluations) // 2)
            requested_n = min(requested_n, max_initial)
        self._n = requested_n
        self._params["population_size"] = self._n
        min_n = int(self._params.get("min_population_size", 4))
        if min_n < 4:
            raise ValueError("min_population_size must be >= 4 for RDEx-SOP.")
        if min_n > self._n:
            raise ValueError("min_population_size cannot exceed population_size for RDEx-SOP.")
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}

    # ------------------------------------------------------------------
    # EvoMapX telemetry helpers
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        return [
            f"{self.algorithm_id}.standard_branch_mutation",
            f"{self.algorithm_id}.exploitation_biased_mutation",
            f"{self.algorithm_id}.binomial_crossover",
            f"{self.algorithm_id}.cauchy_local_perturbation",
            f"{self.algorithm_id}.greedy_selection",
            f"{self.algorithm_id}.dynamic_pbest_selection",
            f"{self.algorithm_id}.hybrid_rate_update",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.linear_population_reduction",
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
    # Native schedules and state
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(1, int(self._params.get("hist_mem_size", 5)))
        rho = float(np.clip(float(self._params.get("initial_hybrid_rate", 0.7)), 0.0, 1.0))
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5)), dtype=float),
            "k": 0,
            "rho_eb": rho,
            "success_rate": 0.0,
            "initial_n": int(pop.shape[0]),
            "last_pbest_count": max(2, min(int(pop.shape[0]), int(round(0.7 * pop.shape[0])))),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step)
        target = int(np.floor(initial_n + (min_n - initial_n) * progress))
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    def _pbest_count(self, n: int, success_rate: float) -> int:
        xi = float(self._params.get("xi", 0.7))
        k = float(self._params.get("k_success", 7.0))
        count = int(np.floor(float(n) * xi * np.exp(-k * float(np.clip(success_rate, 0.0, 1.0)))))
        return int(np.clip(max(2, count), 2, n))

    def _choose_pbest_index(self, order: np.ndarray, pnum: int, i: int) -> int:
        top = np.asarray(order[:max(2, int(pnum))], dtype=int)
        choices = top[top != int(i)]
        if choices.size == 0:
            choices = np.asarray([idx for idx in range(len(order)) if idx != int(i)], dtype=int)
        if choices.size == 0:
            return int(i)
        return int(np.random.choice(choices))

    def _choice_population_index(self, n: int, exclude: set[int]) -> int:
        choices = np.asarray([idx for idx in range(n) if idx not in exclude], dtype=int)
        if choices.size == 0:
            choices = np.arange(n, dtype=int)
        return int(np.random.choice(choices))

    def _resample_bounds(self, vector: np.ndarray) -> tuple[np.ndarray, int]:
        repaired = np.asarray(vector, dtype=float).copy()
        bad = (repaired < self._lo) | (repaired > self._hi) | ~np.isfinite(repaired)
        count = int(np.count_nonzero(bad))
        if count:
            repaired[bad] = np.random.uniform(self._lo[bad], self._hi[bad])
        return self.problem.apply_variable_types(repaired), count

    def _acceptance_mask(self, trial_fit: np.ndarray, parent_fit: np.ndarray) -> np.ndarray:
        if self.problem.objective == "min":
            return np.asarray(trial_fit, dtype=float) <= np.asarray(parent_fit, dtype=float)
        return np.asarray(trial_fit, dtype=float) >= np.asarray(parent_fit, dtype=float)

    # ------------------------------------------------------------------
    # Main native step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        if n < 4:
            raise ValueError("RDEx-SOP requires at least four population members.")
        active_n = n
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            remaining = int(self.config.max_evaluations) - int(state.evaluations)
            active_n = max(0, min(n, remaining))
        if active_n <= 0:
            return pop, 0, {}

        M_F = np.asarray(state.payload.get("M_F"), dtype=float).copy()
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float).copy()
        h = int(M_F.shape[0])
        k_mem = int(state.payload.get("k", 0)) % h
        rho_eb = float(np.clip(float(state.payload.get("rho_eb", self._params.get("initial_hybrid_rate", 0.7))), 0.0, 1.0))
        success_rate = float(np.clip(float(state.payload.get("success_rate", 0.0)), 0.0, 1.0))
        initial_n = int(state.payload.get("initial_n", n))
        progress = self._progress(state.evaluations, state.step)
        order = self._order(pop[:, -1])
        pnum = self._pbest_count(n, success_rate)

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        self._add(contrib, counts, "dynamic_pbest_selection", 0.0, active_n)

        trials = np.empty((active_n, dim), dtype=float)
        Fs = np.empty(active_n, dtype=float)
        CRs = np.empty(active_n, dtype=float)
        use_eb = np.random.rand(active_n) < rho_eb
        mem_indices = np.random.randint(0, h, size=active_n)
        branch_names: list[str] = []
        local_perturbation_count = 0
        bound_resample_count = 0

        mu_f_std = float(0.4 + 0.25 * np.tanh(5.0 * success_rate))
        cr_sigma = float(self._params.get("cr_sigma", 0.1))
        early_cr_fraction = float(np.clip(float(self._params.get("early_cr_fraction", 0.25)), 0.0, 1.0))
        early_cr_lb = float(np.clip(float(self._params.get("early_cr_lower_bound", 0.7)), 0.0, 1.0))

        for i in range(active_n):
            parent = pop[i, :-1]
            mem = int(mem_indices[i])
            pbest_idx = self._choose_pbest_index(order, pnum, i)
            r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)})
            r2_idx = self._choice_population_index(n, {int(i), int(pbest_idx), int(r1_idx)})

            if bool(use_eb[i]):
                F = _positive_cauchy(
                    float(M_F[mem]),
                    float(self._params.get("eb_f_scale", 0.1)),
                    fallback=float(self._params.get("eb_fallback", 0.5)),
                )
                CR = _truncated_normal(float(M_CR[mem]), cr_sigma)
                if progress <= early_cr_fraction:
                    CR = max(CR, early_cr_lb)
                donor_ids = np.asarray([pbest_idx, r1_idx, r2_idx], dtype=int)
                donor_order = donor_ids[np.argsort(pop[donor_ids, -1])]
                if self.problem.objective == "max":
                    donor_order = donor_order[::-1]
                best = pop[donor_order[0], :-1]
                mid = pop[donor_order[1], :-1]
                worst = pop[donor_order[2], :-1]
                donor = parent + F * (best - parent) + F * (mid - worst)
                branch = "exploitation_biased_mutation"
            else:
                F = _truncated_normal(mu_f_std, float(self._params.get("standard_f_sigma", 0.02)))
                CR = _truncated_normal(float(M_CR[mem]), cr_sigma)
                donor = parent + F * (pop[pbest_idx, :-1] - parent) + F * (pop[r1_idx, :-1] - pop[r2_idx, :-1])
                branch = "standard_branch_mutation"

            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, donor, parent)

            # Eq. (10): perturb only dimensions not inherited from the donor.
            pr = float(np.clip(float(self._params.get("local_perturbation_rate", 0.1)), 0.0, 1.0))
            if pr > 0.0:
                perturb_mask = (~cross) & (np.random.rand(dim) < pr)
                if np.any(perturb_mask):
                    scale = float(self._params.get("local_perturbation_scale", 0.1))
                    trial[perturb_mask] = parent[perturb_mask] + np.random.standard_cauchy(np.count_nonzero(perturb_mask)) * scale
                    local_perturbation_count += int(np.count_nonzero(perturb_mask))

            trial, n_repaired = self._resample_bounds(trial)
            bound_resample_count += int(n_repaired)
            trials[i] = trial
            Fs[i] = float(F)
            CRs[i] = float(CR)
            branch_names.append(branch)

        trial_fit = self._evaluate_population(trials)
        evals = int(active_n)
        active_indices = np.arange(active_n, dtype=int)
        parent_fit = pop[active_indices, -1].copy()
        accepted = self._acceptance_mask(trial_fit, parent_fit)
        if self.problem.objective == "min":
            gains = np.where(accepted, parent_fit - trial_fit, 0.0)
        else:
            gains = np.where(accepted, trial_fit - parent_fit, 0.0)
        gains = np.maximum(gains.astype(float), 0.0)
        accepted_gain = float(np.sum(gains))

        branch_array = np.asarray(branch_names, dtype=object)
        std_gain = float(np.sum(gains[branch_array == "standard_branch_mutation"]))
        eb_gain = float(np.sum(gains[branch_array == "exploitation_biased_mutation"]))
        direct_gain_share = accepted_gain / 4.0 if accepted_gain > 0.0 else 0.0
        self._add(contrib, counts, "standard_branch_mutation", std_gain, int(np.count_nonzero(branch_array == "standard_branch_mutation")))
        self._add(contrib, counts, "exploitation_biased_mutation", eb_gain, int(np.count_nonzero(branch_array == "exploitation_biased_mutation")))
        self._add(contrib, counts, "binomial_crossover", direct_gain_share, active_n)
        self._add(contrib, counts, "cauchy_local_perturbation", direct_gain_share if local_perturbation_count else 0.0, local_perturbation_count)
        self._add(contrib, counts, "greedy_selection", direct_gain_share, int(np.count_nonzero(accepted)))
        self._add(contrib, counts, "bound_resampling", 0.0, bound_resample_count)

        if np.any(accepted):
            accepted_indices = active_indices[accepted]
            pop[accepted_indices, :-1] = trials[accepted]
            pop[accepted_indices, -1] = trial_fit[accepted]

        positive = accepted & (gains > 0.0)
        if np.any(positive):
            weights = gains[positive] / (float(np.sum(gains[positive])) + _EPS)
            sf = Fs[positive]
            scr = CRs[positive]
            M_F[k_mem] = _weighted_lehmer(sf, weights, fallback=float(M_F[k_mem]))
            cr_lehmer = _weighted_lehmer(scr, weights, fallback=float(M_CR[k_mem]))
            M_CR[k_mem] = 0.5 * (float(M_CR[k_mem]) + cr_lehmer)
            k_mem = (k_mem + 1) % h
            self._add(contrib, counts, "success_history_update", 0.0, 1)

        denom = eb_gain + std_gain
        if denom > _EPS:
            rho_eb = float(np.clip(eb_gain / denom, 0.0, 1.0))
            self._add(contrib, counts, "hybrid_rate_update", 0.0, 1)

        next_success_rate = float(np.count_nonzero(accepted) / max(1, active_n))
        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0])
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "linear_population_reduction", 0.0, removed)

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "rho_eb": rho_eb,
            "success_rate": next_success_rate,
            "initial_n": initial_n,
            "last_pbest_count": int(pnum),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "direct_improvement"
        obs["evomapx_fidelity"] = "native"
        obs["rho_eb"] = float(state.payload.get("rho_eb", self._params.get("initial_hybrid_rate", 0.7)))
        obs["success_rate"] = float(state.payload.get("success_rate", 0.0))
        obs["pbest_count"] = int(state.payload.get("last_pbest_count", 0))
        obs["current_population_size"] = int(state.payload["population"].shape[0])
        if "M_F" in state.payload and "M_CR" in state.payload:
            obs["mean_memory_f"] = float(np.nanmean(np.asarray(state.payload["M_F"], dtype=float)))
            obs["mean_memory_cr"] = float(np.nanmean(np.asarray(state.payload["M_CR"], dtype=float)))
        return obs
