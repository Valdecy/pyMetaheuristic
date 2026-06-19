"""Shared native Differential-Evolution engines for SHADE/L-SHADE paper variants.

The classes in this module keep the package protocol used by pyMetaheuristic and
implement the common SHADE machinery: success-history memories for F/CR,
current-to-pbest differential mutation, binomial crossover, greedy replacement,
external archive, optional rank-based selective pressure, optional nonlinear
population reduction, and variant hooks used by jSO, LSHADE-EpSin,
LSHADE-SPACMA, LSHADE-RSP, iLSHADE-RSP, NL-SHADE-RSP, NL-SHADE-LBC,
NL-SHADE-RSP-Midpoint, and RDE.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


def _safe_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0, fallback: float = 0.5) -> float:
    """Sample a positive Cauchy value truncated to (0, upper]."""
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(fallback, 1.0e-12), upper)


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights * values))
    if abs(denom) <= 1.0e-30:
        return float(np.mean(values)) if values.size else 0.5
    return float(np.sum(weights * values * values) / denom)


def _levy_step(dim: int, beta: float = 1.5, scale: float = 0.01) -> np.ndarray:
    beta = float(beta)
    sigma = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    v1 = np.random.normal(0.0, sigma, dim)
    v2 = np.random.normal(0.0, 1.0, dim)
    return float(scale) * v1 / (np.abs(v2) ** (1.0 / beta) + 1.0e-30)


class ShadeVariantEngine(PortedPopulationEngine):
    """Configurable base class for paper-level L-SHADE descendants."""

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
    _DEFAULTS = dict(
        population_size=80,
        min_population_size=4,
        hist_mem_size=6,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        archive_rate=2.0,
        pbest_factor=0.11,
        pbest_min=0.05,
        pbest_max=0.25,
        f_scale=0.1,
        cr_sigma=0.1,
    )
    _VARIANT = dict(
        weighted_pbest=False,
        rank_pressure=False,
        nonlinear_reduction=False,
        midpoint=False,
        midpoint_cluster=False,
        restart=False,
        epsin=False,
        spacma=False,
        cauchy_perturb=False,
        lbc=False,
        rde=False,
        use_order_pbest=False,
        resample_bounds=False,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        if self._n < 4:
            raise ValueError(f"population_size must be >= 4 for {self.algorithm_id}.")
        self._variant: dict[str, Any] = {**self._VARIANT, **getattr(self, "_VARIANT_OVERRIDES", {})}
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}

    # ------------------------------ labels ------------------------------
    def _make_operator_labels(self) -> list[str]:
        labels = [
            "mutation",
            "crossover",
            "selection",
            "archive_update",
            "success_history_update",
            "population_reduction",
        ]
        if self._variant.get("rank_pressure"):
            labels.append("rank_selective_pressure")
        if self._variant.get("epsin"):
            labels.extend(["ensemble_sinusoidal_adaptation", "gaussian_walk_local_search"])
        if self._variant.get("spacma"):
            labels.append("cma_es_sampling")
        if self._variant.get("cauchy_perturb"):
            labels.append("cauchy_target_perturbation")
        if self._variant.get("lbc"):
            labels.append("linear_bias_change")
        if self._variant.get("rde"):
            labels.extend(["order_pbest_mutation", "strategy_ratio_update"])
        if self._variant.get("midpoint"):
            labels.extend(["midpoint_evaluation", "midpoint_restart"])
        return [f"{self.algorithm_id}.{label}" for label in labels]

    def _blank_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._operator_labels}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._operator_labels}

    def _add(self, contrib: dict[str, float], counts: dict[str, int], suffix: str, value: float = 0.0, count: int = 0) -> None:
        label = f"{self.algorithm_id}.{suffix}"
        if label in contrib:
            contrib[label] += float(value)
            counts[label] += int(count)

    # ------------------------------ helpers -----------------------------
    def _progress(self, state) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(state.evaluations / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        return float(np.clip((state.step + 1) / float(horizon), 0.0, 1.0))

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = int(self._params.get("hist_mem_size", 6))
        if h <= 0:
            raise ValueError("hist_mem_size must be positive.")
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5)), dtype=float),
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
            "gamma": np.asarray(self._params.get("strategy_ratios", [0.5, 0.5]), dtype=float),
            "strategy_success": np.zeros(2, dtype=float),
            "last_midpoint": None,
            "midpoint_stagnation": 0,
            "restart_count": 0,
        }

    def _rank_weights(self, n: int, order: np.ndarray | None = None) -> np.ndarray:
        kr = float(self._params.get("rank_greediness", 3.0))
        ranks = np.arange(n, 0, -1, dtype=float)  # high mass for good ranks after order mapping
        weights_sorted = ranks ** max(0.0, kr)
        weights_sorted /= float(np.sum(weights_sorted)) + 1.0e-30
        if order is None:
            return weights_sorted
        weights = np.empty(n, dtype=float)
        weights[order] = weights_sorted
        return weights

    def _choice_population_index(self, n: int, exclude: int | None, order: np.ndarray, *, rank_based: bool = False) -> int:
        candidates = np.arange(n, dtype=int)
        if exclude is not None and n > 1:
            candidates = candidates[candidates != int(exclude)]
        if candidates.size == 0:
            return 0
        if rank_based:
            weights = self._rank_weights(n, order)[candidates]
            weights = weights / (float(np.sum(weights)) + 1.0e-30)
            return int(np.random.choice(candidates, p=weights))
        return int(np.random.choice(candidates))

    def _choice_union_vector(self, pop: np.ndarray, archive: np.ndarray, exclude: int, order: np.ndarray, *, rank_based: bool = False) -> np.ndarray:
        n, dim = pop.shape[0], self.problem.dimension
        use_archive = archive.size and np.random.random() < float(self._params.get("archive_use_probability", 0.5))
        if use_archive:
            if rank_based and archive.shape[0] > 1:
                # Archive has no fitness; prefer newer entries weakly by sampling from the tail.
                idx = int(np.random.randint(max(1, archive.shape[0] // 2), archive.shape[0]))
            else:
                idx = int(np.random.randint(archive.shape[0]))
            return archive[idx].copy()
        idx = self._choice_population_index(n, exclude, order, rank_based=rank_based)
        return pop[idx, :-1].copy()

    def _sample_memory_index(self, h: int, progress: float) -> int:
        if self._variant.get("lbc"):
            # NL-SHADE-LBC biases parameter memory access linearly over time.
            weights = np.linspace(1.0 - 0.75 * progress, 1.0 + 0.75 * progress, h)
            weights = weights / (float(np.sum(weights)) + 1.0e-30)
            return int(np.random.choice(np.arange(h), p=weights))
        if self._variant.get("rde"):
            return int(np.random.randint(h))
        return int(np.random.randint(h))

    def _sample_f_cr(self, M_F: np.ndarray, M_CR: np.ndarray, state, idx: int, i: int) -> tuple[float, float]:
        progress = self._progress(state)
        mf = float(M_F[idx])
        mcr = float(M_CR[idx]) if np.isfinite(M_CR[idx]) else 0.0
        if self._variant.get("epsin"):
            # LSHADE-EpSin: ensemble sinusoidal F; history-based fallback is retained.
            freq = float(self._params.get("epsin_frequency", 0.5))
            if np.random.random() < 0.5:
                raw = 0.5 * (math.sin(2.0 * math.pi * freq * (state.step + 1)) + 1.0)
                F = 0.05 + 0.95 * raw * (1.0 - 0.35 * progress)
            else:
                raw = 0.5 * (math.sin(2.0 * math.pi * (freq + 0.25 * mf) * (state.step + 1)) + 1.0)
                F = 0.1 + 0.9 * raw
        else:
            F = _safe_cauchy(mf, float(self._params.get("f_scale", 0.1)), fallback=mf if mf > 0 else 0.5)
        CR = 0.0 if not np.isfinite(mcr) else float(np.clip(np.random.normal(mcr, float(self._params.get("cr_sigma", 0.1))), 0.0, 1.0))
        if self.algorithm_id == "jso_de":
            # jSO paper: high CR floor in the first half and cap large F early.
            if progress < 0.25:
                CR = max(CR, 0.7)
            elif progress < 0.5:
                CR = max(CR, 0.6)
            if progress < 0.6 and F > 0.7:
                F = 0.7
        if self._variant.get("lbc"):
            # Linear-bias change: early exploration accepts larger F, late stages push CR up.
            F = float(np.clip((1.0 - 0.35 * progress) * F + 0.35 * progress * min(F, 0.7), 1.0e-12, 1.0))
            CR = float(np.clip((1.0 - progress) * CR + progress * max(CR, 0.8), 0.0, 1.0))
        return float(F), float(CR)

    def _p_value(self, n: int, state) -> float:
        progress = self._progress(state)
        pmax = float(self._params.get("pbest_max", self._params.get("pbest_factor", 0.11)))
        pmin = float(self._params.get("pbest_min", 2.0 / max(n, 2)))
        if self._variant.get("rde"):
            pmax = float(self._params.get("pmax", 0.25))
            p = pmax * (1.0 - progress) + max(2.0 / max(n, 2), pmin) * progress
        elif self._variant.get("nlshade") or self._variant.get("rank_pressure"):
            p = pmax + (pmin - pmax) * progress
        else:
            p = float(self._params.get("pbest_factor", 0.11))
        return float(np.clip(p, 2.0 / max(2, n), 1.0))

    def _target_population_size(self, state, initial_n: int, current_n: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(state)
        if self._variant.get("nonlinear_reduction"):
            exponent = float(self._params.get("nlpsr_exponent", 1.5))
            target = min_n + (initial_n - min_n) * ((1.0 - progress) ** exponent)
        else:
            target = initial_n + (min_n - initial_n) * progress
        return int(np.clip(round(target), min_n, max(min_n, initial_n, current_n)))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> np.ndarray:
        donor = np.asarray(donor, dtype=float).copy()
        parent = np.asarray(parent, dtype=float)
        if self._variant.get("resample_bounds"):
            mask = (donor < self._lo) | (donor > self._hi)
            if np.any(mask):
                donor[mask] = np.random.uniform(self._lo[mask], self._hi[mask])
        else:
            below = donor < self._lo
            above = donor > self._hi
            donor[below] = 0.5 * (self._lo[below] + parent[below])
            donor[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(donor, self._lo, self._hi)

    def _weighted_pbest_factor(self, F: float, state) -> float:
        if not self._variant.get("weighted_pbest"):
            return F
        progress = self._progress(state)
        if progress < 0.2:
            return 0.7 * F
        if progress < 0.4:
            return 0.8 * F
        return 1.2 * F

    def _order_pbest_donor(self, parent: np.ndarray, pop: np.ndarray, archive: np.ndarray, order: np.ndarray, F: float, pbest: np.ndarray, i: int) -> np.ndarray:
        n = pop.shape[0]
        r1i = self._choice_population_index(n, i, order, rank_based=True)
        r2 = self._choice_union_vector(pop, archive, i, order, rank_based=True)
        x1 = pop[r1i, :-1]
        # Order the three anchors by current objective quality when possible.
        anchors = [pbest, x1, r2]
        approx_f = []
        for a in anchors:
            # Archive members do not have stored fitness; use distance to best as a cheap rank proxy.
            d = float(np.linalg.norm(a - pbest))
            approx_f.append(d)
        sorted_anchors = [anchors[j] for j in np.argsort(approx_f)]
        best_a, mid_a, worst_a = sorted_anchors
        return parent + F * (best_a - parent) + F * (mid_a - worst_a)

    def _donor(self, parent: np.ndarray, pop: np.ndarray, archive: np.ndarray, order: np.ndarray, F: float, pbest: np.ndarray, i: int, strategy: int, state) -> tuple[np.ndarray, str]:
        n = pop.shape[0]
        rank = bool(self._variant.get("rank_pressure"))
        if self._variant.get("rde") and strategy == 0:
            return self._order_pbest_donor(parent, pop, archive, order, F, pbest, i), "order_pbest_mutation"
        r1i = self._choice_population_index(n, i, order, rank_based=rank)
        r2 = self._choice_union_vector(pop, archive, i, order, rank_based=rank)
        fp = self._weighted_pbest_factor(F, state)
        return parent + fp * (pbest - parent) + F * (pop[r1i, :-1] - r2), "mutation"

    # --------------------------- variant add-ons --------------------------
    def _evaluate_midpoints(self, state, pop: np.ndarray, contrib: dict[str, float], counts: dict[str, int]) -> tuple[np.ndarray, int]:
        if not self._variant.get("midpoint") or pop.shape[0] < 4:
            return pop, 0
        pos = pop[:, :-1]
        candidates = [np.mean(pos, axis=0)]
        if self._variant.get("midpoint_cluster") and pop.shape[0] >= 8:
            # Lightweight deterministic two-means split to avoid sklearn dependency.
            a = pos[self._best_index(pop[:, -1])]
            b = pos[self._worst_index(pop[:, -1])]
            for _ in range(3):
                da = np.linalg.norm(pos - a, axis=1)
                db = np.linalg.norm(pos - b, axis=1)
                mask = da <= db
                if np.any(mask):
                    a = np.mean(pos[mask], axis=0)
                if np.any(~mask):
                    b = np.mean(pos[~mask], axis=0)
            candidates.extend([a, b])
        cand = np.clip(np.asarray(candidates, dtype=float), self._lo, self._hi)
        fit = self._evaluate_population(cand)
        evals = int(cand.shape[0])
        self._add(contrib, counts, "midpoint_evaluation", 0.0, evals)
        for x, f in zip(cand, fit):
            wi = self._worst_index(pop[:, -1])
            if self._is_better(float(f), float(pop[wi, -1])):
                if self.problem.objective == "min":
                    gain = max(0.0, float(pop[wi, -1] - f))
                else:
                    gain = max(0.0, float(f - pop[wi, -1]))
                pop[wi, :-1] = x
                pop[wi, -1] = f
                self._add(contrib, counts, "midpoint_evaluation", gain, 0)
        # Restart trigger: if midpoint does not move for several iterations, reseed worst half.
        if self._variant.get("restart"):
            midpoint = cand[0]
            last = state.payload.get("last_midpoint")
            moved = np.inf if last is None else float(np.linalg.norm(midpoint - np.asarray(last, dtype=float)))
            tol = float(self._params.get("midpoint_restart_tol", 1.0e-8)) * float(np.linalg.norm(self._span))
            stagnation = int(state.payload.get("midpoint_stagnation", 0))
            stagnation = stagnation + 1 if moved <= tol else 0
            state.payload["last_midpoint"] = midpoint.copy()
            state.payload["midpoint_stagnation"] = stagnation
            patience = int(self._params.get("midpoint_restart_patience", 9))
            if stagnation >= patience:
                order = self._order(pop[:, -1])
                worst = order[pop.shape[0] // 2:]
                new_pos = self._new_positions(len(worst))
                new_fit = self._evaluate_population(new_pos)
                pop[worst, :-1] = new_pos
                pop[worst, -1] = new_fit
                evals += int(len(worst))
                state.payload["midpoint_stagnation"] = 0
                state.payload["restart_count"] = int(state.payload.get("restart_count", 0)) + 1
                self._add(contrib, counts, "midpoint_restart", 0.0, len(worst))
        return pop, evals

    def _cma_sampling(self, pop: np.ndarray, state, contrib: dict[str, float], counts: dict[str, int]) -> tuple[np.ndarray, int]:
        if not self._variant.get("spacma") or pop.shape[0] < 6:
            return pop, 0
        progress = self._progress(state)
        n, dim = pop.shape[0], self.problem.dimension
        m = max(1, int(round((0.25 * (1.0 - progress) + 0.08) * n)))
        order = self._order(pop[:, -1])
        elites = pop[order[: max(3, min(n, n // 3))], :-1]
        mean = np.mean(elites, axis=0)
        cov = np.cov(elites.T) if elites.shape[0] > 1 else np.diag((0.05 * self._span) ** 2)
        if np.ndim(cov) == 0:
            cov = np.eye(dim) * float(cov)
        cov = np.asarray(cov, dtype=float) + np.eye(dim) * (1.0e-12 + (0.05 * (1.0 - progress)) ** 2)
        try:
            samples = np.random.multivariate_normal(mean, cov, size=m)
        except np.linalg.LinAlgError:
            samples = mean + np.random.normal(0.0, 0.1 * self._span, size=(m, dim))
        samples = np.clip(samples, self._lo, self._hi)
        fit = self._evaluate_population(samples)
        evals = int(m)
        for x, f in zip(samples, fit):
            wi = self._worst_index(pop[:, -1])
            if self._is_better(float(f), float(pop[wi, -1])):
                gain = max(0.0, float(pop[wi, -1] - f if self.problem.objective == "min" else f - pop[wi, -1]))
                pop[wi, :-1] = x
                pop[wi, -1] = f
                self._add(contrib, counts, "cma_es_sampling", gain, 0)
        self._add(contrib, counts, "cma_es_sampling", 0.0, evals)
        return pop, evals

    def _epsin_local_search(self, pop: np.ndarray, state, contrib: dict[str, float], counts: dict[str, int]) -> tuple[np.ndarray, int]:
        if not self._variant.get("epsin") or self._progress(state) < float(self._params.get("local_search_start", 0.85)):
            return pop, 0
        order = self._order(pop[:, -1])
        k = max(1, int(self._params.get("local_search_elites", 2)))
        sigma = float(self._params.get("local_search_sigma", 0.01)) * self._span * max(0.05, 1.0 - self._progress(state))
        trials = []
        source_idx = []
        for idx in order[:k]:
            trials.append(np.clip(pop[idx, :-1] + np.random.normal(0.0, sigma, self.problem.dimension), self._lo, self._hi))
            source_idx.append(int(idx))
        trials = np.asarray(trials, dtype=float)
        fit = self._evaluate_population(trials)
        for row_i, f in zip(source_idx, fit):
            if self._is_better(float(f), float(pop[row_i, -1])):
                gain = max(0.0, float(pop[row_i, -1] - f if self.problem.objective == "min" else f - pop[row_i, -1]))
                pop[row_i, :-1] = trials[source_idx.index(row_i)]
                pop[row_i, -1] = f
                self._add(contrib, counts, "gaussian_walk_local_search", gain, 0)
        self._add(contrib, counts, "gaussian_walk_local_search", 0.0, len(trials))
        return pop, int(len(trials))

    # ------------------------------ main step -----------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        if n < 4:
            raise ValueError(f"{self.algorithm_id} requires at least 4 population members.")
        M_F = np.asarray(state.payload.get("M_F"), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float)
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        h = len(M_F)
        k_mem = int(state.payload.get("k", 0)) % h
        initial_n = int(state.payload.get("initial_n", n))
        gamma = np.asarray(state.payload.get("gamma", np.array([0.5, 0.5])), dtype=float)
        if gamma.size != 2 or np.sum(gamma) <= 0:
            gamma = np.array([0.5, 0.5], dtype=float)
        gamma = gamma / np.sum(gamma)

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state)
        order = self._order(pop[:, -1])
        p = self._p_value(n, state)
        pnum = max(2, int(round(p * n)))

        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        mem_idx = np.empty(n, dtype=int)
        strategies = np.ones(n, dtype=int)
        mutation_suffixes: list[str] = []
        parent_before = pop[:, :-1].copy()

        for i in range(n):
            r = self._sample_memory_index(h, progress)
            F, CR = self._sample_f_cr(M_F, M_CR, state, r, i)
            if self._variant.get("rde"):
                strategies[i] = 0 if np.random.random() < gamma[0] else 1
            pbest = pop[np.random.choice(order[:pnum]), :-1]
            parent = pop[i, :-1]
            donor, suffix = self._donor(parent, pop, archive, order, F, pbest, i, int(strategies[i]), state)
            if self._variant.get("cauchy_perturb") and np.random.random() < float(self._params.get("jumping_rate", self._params.get("cauchy_rate", 0.2))):
                donor = donor + np.random.standard_cauchy(dim) * float(self._params.get("cauchy_scale", 0.1)) * self._span
                suffix = "cauchy_target_perturbation"
            donor = self._repair_bound(donor, parent)
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trials[i] = np.where(cross, donor, parent)
            Fs[i] = F
            CRs[i] = CR
            mem_idx[i] = r
            mutation_suffixes.append(suffix)

        trial_fit = self._evaluate_population(trials)
        evals = n
        mask = self._better_mask(trial_fit, pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - trial_fit[mask]
        else:
            gains[mask] = trial_fit[mask] - pop[mask, -1]
        accepted_gain = float(np.sum(np.maximum(gains, 0.0)))
        self._add(contrib, counts, "crossover", accepted_gain / 3.0 if accepted_gain > 0 else 0.0, n)
        self._add(contrib, counts, "selection", accepted_gain / 3.0 if accepted_gain > 0 else 0.0, int(np.count_nonzero(mask)))
        if self._variant.get("rank_pressure"):
            self._add(contrib, counts, "rank_selective_pressure", 0.0, n)
        if self._variant.get("epsin"):
            self._add(contrib, counts, "ensemble_sinusoidal_adaptation", 0.0, n)
        if self._variant.get("lbc"):
            self._add(contrib, counts, "linear_bias_change", 0.0, n)

        # Attribute accepted gain to the actual mutation mechanism used.
        if np.any(mask):
            for suffix in set(mutation_suffixes):
                idxs = np.array([j for j, s in enumerate(mutation_suffixes) if s == suffix], dtype=int)
                gain = float(np.sum(np.maximum(gains[idxs], 0.0)))
                self._add(contrib, counts, suffix, gain / 3.0 if gain > 0 else 0.0, int(idxs.size))

            old_success = parent_before[mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(float(self._params.get("archive_rate", 2.0)) * n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            sf = Fs[mask]
            scr = CRs[mask]
            df = np.abs(pop[mask, -1] - trial_fit[mask])
            weights = df / (float(np.sum(df)) + 1.0e-30)
            M_F[k_mem] = _weighted_lehmer(sf, weights)
            if np.max(scr) <= 0.0 or not np.isfinite(M_CR[k_mem]):
                M_CR[k_mem] = np.nan
            else:
                M_CR[k_mem] = float(np.sum(weights * scr * scr) / (np.sum(weights * scr) + 1.0e-30))
            k_mem = (k_mem + 1) % h
            self._add(contrib, counts, "success_history_update", 0.0, 1)

            if self._variant.get("rde"):
                succ0 = float(np.sum(gains[mask & (strategies == 0)]))
                succ1 = float(np.sum(gains[mask & (strategies == 1)]))
                prev = np.asarray(state.payload.get("strategy_success", np.zeros(2)), dtype=float)
                succ = 0.7 * prev + 0.3 * np.array([succ0, succ1], dtype=float)
                if np.sum(succ) > 0:
                    gamma = 0.1 + 0.8 * succ / np.sum(succ)
                    gamma = gamma / np.sum(gamma)
                state.payload["strategy_success"] = succ
                self._add(contrib, counts, "strategy_ratio_update", 0.0, 1)

            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        pop, mid_evals = self._evaluate_midpoints(state, pop, contrib, counts)
        evals += mid_evals
        pop, cma_evals = self._cma_sampling(pop, state, contrib, counts)
        evals += cma_evals
        pop, ls_evals = self._epsin_local_search(pop, state, contrib, counts)
        evals += ls_evals

        target_n = self._target_population_size(state, initial_n, pop.shape[0])
        if pop.shape[0] > target_n:
            order = self._order(pop[:, -1])[:target_n]
            removed = pop.shape[0] - target_n
            pop = pop[order]
            max_arc = max(1, int(float(self._params.get("archive_rate", 2.0)) * target_n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "population_reduction", 0.0, removed)

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "initial_n": initial_n,
            "gamma": gamma,
            "strategy_success": state.payload.get("strategy_success", np.zeros(2, dtype=float)),
            "last_midpoint": state.payload.get("last_midpoint"),
            "midpoint_stagnation": state.payload.get("midpoint_stagnation", 0),
            "restart_count": state.payload.get("restart_count", 0),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        if "gamma" in state.payload:
            obs["strategy_ratios"] = np.asarray(state.payload["gamma"], dtype=float).tolist()
        if "restart_count" in state.payload:
            obs["restart_count"] = int(state.payload.get("restart_count", 0))
        return obs
