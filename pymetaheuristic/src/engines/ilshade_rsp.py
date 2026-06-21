"""pyMetaheuristic src — faithful iLSHADE-RSP Engine.

This module implements the iLSHADE-RSP variant proposed by Choi and Ahn
(arXiv:2006.02591): LSHADE-RSP/current-to-pbest-r mutation with rank-based
selective pressure, jSO-style weighted pbest term, increasing pbest pool,
linear population-size reduction, success-history F/CR adaptation with a
reserved 0.9 memory cell, and the paper's Cauchy target-vector perturbation
selected by the jumping rate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """SHADE-style positive Cauchy draw, resampled while F <= 0 and capped at upper."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(center, 0.5, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian CR draw clipped to [0, 1]."""
    if not np.isfinite(center) or center < 0.0:
        return 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used by SHADE-family success-history updates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.average(values, weights=weights))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


class ILSHADERspEngine(PortedPopulationEngine):
    """iLSHADE-RSP with paper-faithful Cauchy perturbation of target vectors."""

    algorithm_id = "ilshade_rsp"
    algorithm_name = "iLSHADE-RSP"
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
        "doi": "10.48550/arXiv.2006.02591",
        "title": "An Improved LSHADE-RSP Algorithm with the Cauchy Perturbation: iLSHADE-RSP",
        "authors": "Tae Jong Choi and Chang Wook Ahn",
        "year": 2020,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults are dimension-dependent where stated explicitly.
        # None/"paper"/0 means NPinit = round(sqrt(D) * log(D) * 25).
        population_size=None,
        min_population_size=4,
        # The paper states a historical memory of capacity H but does not state
        # H numerically in the provided manuscript; LSHADE-RSP package convention
        # is retained and remains user-configurable.
        hist_mem_size=5,
        memory_f_init=0.3,
        memory_cr_init=0.8,
        reserved_memory_value=0.9,
        archive_rate=2.0,
        rank_greediness=3.0,
        pbest_base=0.085,
        jumping_rate=0.2,
        cauchy_scale=0.1,
        f_scale=0.1,
        cr_sigma=0.1,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            d = max(2, int(self.problem.dimension))
            self._n = max(4, int(round(np.sqrt(d) * np.log(d) * 25.0)))
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
            f"{self.algorithm_id}.crossover",
            f"{self.algorithm_id}.selection",
            f"{self.algorithm_id}.archive_update",
            f"{self.algorithm_id}.success_history_update",
            f"{self.algorithm_id}.population_reduction",
            f"{self.algorithm_id}.rank_selective_pressure",
            f"{self.algorithm_id}.weighted_pbest_scaling",
            f"{self.algorithm_id}.cauchy_target_perturbation",
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
        # Algorithm 2: if r_i = H, both memories are 0.9; last entry remains 0.9.
        m_f[-1] = reserved
        m_cr[-1] = reserved
        return {
            "M_F": m_f,
            "M_CR": m_cr,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _pbest_count(self, n: int, progress: float) -> int:
        # LSHADE-RSP p setting used by iLSHADE-RSP: p = 0.085 * (1 + NFE/NFEmax).
        p = float(self._params.get("pbest_base", 0.085)) * (1.0 + float(progress))
        return int(np.clip(round(p * n), 2, max(2, n)))

    def _weighted_pbest_factor(self, F: float, progress: float) -> float:
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
    def _rank_probabilities(self, n: int, order: np.ndarray) -> np.ndarray:
        # order is best-to-worst. Paper: Rank_i = k*(NP_g - i) + 1.
        # With zero-based sorted rank s, this is k*(n - 1 - s) + 1.
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
        """Select r2 from the population with rank pressure or uniformly from archive.

        L-SHADE-style union sampling is used: when the archive is non-empty,
        archive selection probability is |A|/(NP + |A|); population-selected r2
        uses the rank probabilities from the LSHADE-RSP paper.
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
        # Algorithm 2 caps F at 0.7 before 60% of the evaluation budget.
        if progress < 0.6 and F > 0.7:
            F = 0.7
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        # Algorithm 2 applies early CR floors after clipping.
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
        use_cauchy_operator: bool,
    ) -> tuple[np.ndarray, int, int]:
        """Construct Eq. (16) or Eq. (17) from the iLSHADE-RSP paper.

        Original operator: non-crossover components remain x_i.
        Modified operator: non-crossover components are Cauchy(x_i^j, 0.1).
        """
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        pbest_idx = self._choose_pbest_index(order, pnum, i)
        r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)}, order, rank_based=True)
        r2, archive_used = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx, order)
        Fw = self._weighted_pbest_factor(F, progress)
        donor = parent + Fw * (pop[pbest_idx, :-1] - parent) + float(F) * (pop[r1_idx, :-1] - r2)
        cross = np.random.rand(dim) < float(CR)
        cross[np.random.randint(dim)] = True
        if use_cauchy_operator:
            # Eq. (17): the *target vector* is perturbed for components not
            # inherited from the donor expression.
            cauchy_target = parent + np.random.standard_cauchy(dim) * float(self._params.get("cauchy_scale", 0.1))
            trial = np.where(cross, donor, cauchy_target)
        else:
            # Eq. (16): standard LSHADE-RSP recombination.
            trial = np.where(cross, donor, parent)
        return self._midpoint_repair(trial, parent), int(archive_used), int(np.count_nonzero(~cross) if use_cauchy_operator else 0)

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
            raise ValueError("hist_mem_size must be at least 2 because iLSHADE-RSP reserves the last memory cell at 0.9.")
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

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        use_cauchy = np.random.rand(n) <= float(self._params.get("jumping_rate", 0.2))
        archive_r2_count = 0
        perturbed_components = 0

        for i in range(n):
            mem_idx = int(np.random.randint(h))
            F, CR = self._sample_f_cr(M_F, M_CR, mem_idx, progress)
            trial, used_archive, n_cauchy_components = self._make_trial(
                parent=pop[i, :-1],
                pop=pop,
                archive=archive,
                order=order,
                i=i,
                F=F,
                CR=CR,
                progress=progress,
                pnum=pnum,
                use_cauchy_operator=bool(use_cauchy[i]),
            )
            trials[i] = trial
            Fs[i] = F
            CRs[i] = CR
            archive_r2_count += int(used_archive)
            perturbed_components += int(n_cauchy_components)

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        # The paper accepts equality. For continuous problems equality is rare,
        # but using <= preserves the pseudo-code semantics for minimization.
        if self.problem.objective == "min":
            mask = trial_fit <= pop[:, -1]
            strict_gain = np.maximum(pop[:, -1] - trial_fit, 0.0)
        else:
            mask = trial_fit >= pop[:, -1]
            strict_gain = np.maximum(trial_fit - pop[:, -1], 0.0)
        accepted_gain = float(np.sum(strict_gain[mask]))

        # Direct-improvement telemetry. State/control operators are counted even
        # when they should not receive immediate Δf credit.
        cauchy_gain = float(np.sum(strict_gain[mask & use_cauchy]))
        original_gain = float(np.sum(strict_gain[mask & ~use_cauchy]))
        self._add(contrib, counts, "mutation", original_gain / 3.0 if original_gain > 0.0 else 0.0, int(n))
        self._add(contrib, counts, "crossover", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(n))
        self._add(contrib, counts, "selection", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(np.count_nonzero(mask)))
        if cauchy_gain > 0.0:
            self._add(contrib, counts, "mutation", cauchy_gain / 4.0, 0)
            self._add(contrib, counts, "cauchy_target_perturbation", cauchy_gain / 4.0, 0)
        self._add(contrib, counts, "rank_selective_pressure", 0.0, int(2 * n - archive_r2_count))
        self._add(contrib, counts, "weighted_pbest_scaling", 0.0, int(n))
        self._add(contrib, counts, "cauchy_target_perturbation", 0.0, int(np.count_nonzero(use_cauchy)))

        if np.any(mask):
            old_success = parent_before[mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(round(float(self._params.get("archive_rate", 2.0)) * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            sf = Fs[mask]
            scr = CRs[mask]
            df = np.abs(parent_fit_before[mask] - trial_fit[mask])
            weights = df / (float(np.sum(df)) + _EPS)
            M_F[k_mem] = _weighted_lehmer(sf, weights)
            M_CR[k_mem] = _weighted_lehmer(scr, weights)
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

        max_arc = max(1, int(round(float(self._params.get("archive_rate", 2.0)) * pop.shape[0])))
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
            "last_cauchy_perturbed_components": int(perturbed_components),
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
        obs["last_cauchy_perturbed_components"] = int(state.payload.get("last_cauchy_perturbed_components", 0))
        return obs
