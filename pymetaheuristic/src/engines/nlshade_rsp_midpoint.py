"""pyMetaheuristic src — faithful NL-SHADE-RSP-Midpoint Engine.

Native NumPy implementation of the algorithm proposed by Biedrzycki, Arabas,
and Warchulski (CEC 2022).  The engine starts from the NL-SHADE-RSP mechanics
(NLPSR, adaptive archive usage, selective pressure on population-selected r2,
and mixed binomial/exponential crossover) and adds the paper's midpoint-based
estimation, restart triggers, resampling bound handling, and optional two-cluster
midpoint evaluation.
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
    """Weighted Lehmer mean used by SHADE/L-SHADE success-history updates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


def _is_better_value(objective: str, candidate: float, incumbent: float) -> bool:
    return float(candidate) < float(incumbent) if str(objective).lower() == "min" else float(candidate) > float(incumbent)


class NLSHADERspMidpointEngine(PortedPopulationEngine):
    """NL-SHADE-RSP-MID with midpoint, restart, resampling, and k-means midpoint."""

    algorithm_id = "nlshade_rsp_midpoint"
    algorithm_name = "NL-SHADE-RSP-Midpoint"
    family = "evolutionary"
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _REFERENCE = {
        "doi": "10.1109/CEC55065.2022.9870220",
        "title": "A Version of NL-SHADE-RSP Algorithm with Midpoint for CEC 2022 Single Objective Bound Constrained Problems",
        "authors": "Rafal Biedrzycki, Jaroslaw Arabas, Eryk Warchulski",
        "year": 2022,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Final paper version is named NL-SHADE-RSP-MID for an initial population of 100.
        # Passing population_size=None/0/"5D" activates the paper's earlier 5D setting.
        population_size=100,
        min_population_size=20,
        hist_mem_size=None,          # paper default inherited from NL-SHADE-RSP: H = 20D
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
        midpoint_restart_patience=9,
        midpoint_restart_tol=1.0e-9,
        bound_restart_patience=10,
        restart_population_size=400,
        restart_min_population_size=20,
        resample_same_parameter_attempts=10,
        resample_max_attempts=100,
        kmeans_min_population_size=20,
        kmeans_max_iterations=12,
        kmeans_silhouette_denominator=4.0,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", 100)
        if pop_param in (None, 0, "5D", "5d", "paper_5d"):
            self._n = max(4, 5 * int(self.problem.dimension))
        elif pop_param in ("paper", "auto", "mid", "midpoint"):
            self._n = 100
        else:
            self._n = max(4, int(pop_param))
        self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}
        self._last_restart_reason: str | None = None

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
            f"{self.algorithm_id}.bound_resampling",
            f"{self.algorithm_id}.bound_random_repair_fallback",
            f"{self.algorithm_id}.midpoint_evaluation",
            f"{self.algorithm_id}.midpoint_replacement",
            f"{self.algorithm_id}.kmeans_midpoint",
            f"{self.algorithm_id}.midpoint_restart",
            f"{self.algorithm_id}.bounds_restart",
            f"{self.algorithm_id}.restart",
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
            "base_initial_n": int(pop.shape[0]),
            "p_archive": float(self._params.get("archive_use_probability", 0.5)),
            "pbest_value": float(self._params.get("pbest_max", 0.4)),
            "binomial_cr": 0.0,
            "midpoint_history": [],
            "bound_counters": np.zeros(int(pop.shape[0]), dtype=int),
            "restart_count": 0,
            "restart_start_evaluations": 0,
            "restart_budget": None,
            "last_restart_reason": None,
        }

    def _reset_memory(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(h, float(self._params.get("memory_f_init", 0.2)), dtype=float),
            np.full(h, float(self._params.get("memory_cr_init", 0.2)), dtype=float),
        )

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
        if progress < 0.5:
            return 0.0
        return float(np.clip(2.0 * (float(progress) - 0.5), 0.0, 1.0))

    def _homographic_target(self, elapsed: int, start_n: int, end_n: int, budget: int) -> int:
        """Paper restart reduction: a1 + 1/(b*1e-6 + 2.57*a2), matching endpoints."""
        start_n = max(4, int(start_n))
        end_n = max(4, int(end_n))
        if start_n <= end_n or budget <= 0:
            return end_n
        c = max(1.0e-12, float(budget) * 1.0e-6)
        d = float(start_n - end_n)
        t = (-c + np.sqrt(c * c + 4.0 * c / d)) / 2.0
        if not np.isfinite(t) or t <= 0.0:
            return end_n
        a1 = float(start_n) - 1.0 / t
        target = np.floor(a1 + 1.0 / (float(elapsed) * 1.0e-6 + t))
        return int(np.clip(target, end_n, start_n))

    def _target_population_size(self, evaluations_after_step: int, payload: dict[str, Any], current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 20)))
        restart_count = int(payload.get("restart_count", 0))
        if restart_count > 0:
            start_n = int(payload.get("initial_n", current_n))
            end_n = max(4, int(self._params.get("restart_min_population_size", 20)))
            start_eval = int(payload.get("restart_start_evaluations", 0))
            if payload.get("restart_budget") is not None:
                budget = int(payload.get("restart_budget"))
            elif self.config.max_evaluations is not None:
                budget = max(1, int(self.config.max_evaluations) - start_eval)
            else:
                budget = max(1, start_n * max(1, int(self.config.max_steps or 100)))
            elapsed = max(0, int(evaluations_after_step) - start_eval)
            return int(np.clip(self._homographic_target(elapsed, start_n, end_n, budget), end_n, max(start_n, current_n)))

        initial_n = int(payload.get("initial_n", current_n))
        progress = self._progress(evaluations_after_step, step)
        if progress <= 0.0:
            target = initial_n
        else:
            target = round((min_n - initial_n) * (progress ** (1.0 - progress)) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    def _archive_capacity(self, n: int) -> int:
        return max(1, int(round(float(self._params.get("archive_rate", 2.1)) * int(n))))

    # ------------------------------------------------------------------
    # Selection primitives
    # ------------------------------------------------------------------
    def _rank_probabilities(self, n: int, order: np.ndarray) -> np.ndarray:
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
    # Crossover, bounds, midpoint, and k-means helpers
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

    def _sample_donor_resampling(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        i: int,
        order: np.ndarray,
        pnum: int,
        F: float,
        cr_value: float,
        p_archive: float,
        M_F: np.ndarray,
        M_CR: np.ndarray,
    ) -> tuple[np.ndarray, float, float, bool, int, bool]:
        """Generate a donor; if it violates bounds, repeat mutation before fallback repair."""
        max_attempts = max(1, int(self._params.get("resample_max_attempts", 100)))
        same_param = max(0, int(self._params.get("resample_same_parameter_attempts", 10)))
        current_F = float(F)
        current_cr = float(cr_value)
        archive_flag = False
        for attempt in range(max_attempts):
            if attempt > 0 and attempt >= same_param:
                idx = int(np.random.randint(len(M_F)))
                current_F = _positive_cauchy(M_F[idx], float(self._params.get("f_scale", 0.1)))
                current_cr = _normal_cr(M_CR[idx], float(self._params.get("cr_sigma", 0.1)))
            pbest_idx = self._choose_pbest_index(order, pnum, i)
            r1_idx = self._choice_population_index(pop.shape[0], {int(i), int(pbest_idx)}, order, rank_based=False)
            r2_vec, archive_flag = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx, order, p_archive)
            donor = pop[i, :-1] + current_F * (pop[pbest_idx, :-1] - pop[i, :-1]) + current_F * (pop[r1_idx, :-1] - r2_vec)
            if np.all((donor >= self._lo) & (donor <= self._hi)):
                return donor, current_F, current_cr, archive_flag, attempt, False
        donor, _ = self._random_bound_repair(donor)
        return donor, current_F, current_cr, archive_flag, max_attempts, True

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

    def _evaluate_and_maybe_insert_point(
        self,
        pop: np.ndarray,
        point: np.ndarray,
        contrib: dict[str, float],
        counts: dict[str, int],
        eval_label: str,
        replace_label: str,
    ) -> tuple[np.ndarray, int, float]:
        point = np.clip(np.asarray(point, dtype=float), self._lo, self._hi)
        fit = float(self._evaluate_population(point.reshape(1, -1))[0])
        self._add(contrib, counts, eval_label, 0.0, 1)
        distances = np.linalg.norm(pop[:, :-1] - point.reshape(1, -1), axis=1)
        idx = int(np.argmin(distances))
        if _is_better_value(self.problem.objective, fit, float(pop[idx, -1])):
            gain = abs(float(pop[idx, -1]) - fit)
            pop = pop.copy()
            pop[idx, :-1] = point
            pop[idx, -1] = fit
            self._add(contrib, counts, replace_label, gain, 1)
            return pop, 1, gain
        return pop, 1, 0.0

    def _midpoint_pass(self, pop: np.ndarray, contrib: dict[str, float], counts: dict[str, int]) -> tuple[np.ndarray, int, np.ndarray, float]:
        midpoint = np.mean(pop[:, :-1], axis=0)
        pop, evals, gain = self._evaluate_and_maybe_insert_point(pop, midpoint, contrib, counts, "midpoint_evaluation", "midpoint_replacement")
        return pop, evals, midpoint, gain

    def _simple_kmeans2(self, X: np.ndarray, max_iter: int = 12) -> tuple[np.ndarray, np.ndarray] | None:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n < 4:
            return None
        # Deterministic farthest-pair style initialization for stability.
        mean = np.mean(X, axis=0)
        i0 = int(np.argmax(np.linalg.norm(X - mean, axis=1)))
        i1 = int(np.argmax(np.linalg.norm(X - X[i0], axis=1)))
        centers = np.vstack((X[i0], X[i1])).astype(float)
        labels = np.zeros(n, dtype=int)
        for _ in range(max(1, int(max_iter))):
            dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dist, axis=1)
            if len(np.unique(new_labels)) < 2:
                return None
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centers = np.vstack([X[labels == c].mean(axis=0) for c in (0, 1)])
        return labels, centers

    def _silhouette2(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if X.shape[0] < 4 or len(np.unique(labels)) < 2:
            return -1.0
        values: list[float] = []
        for i in range(X.shape[0]):
            same = labels == labels[i]
            other = ~same
            if np.count_nonzero(same) <= 1 or np.count_nonzero(other) == 0:
                continue
            a = float(np.mean(np.linalg.norm(X[same & (np.arange(X.shape[0]) != i)] - X[i], axis=1)))
            b = float(np.mean(np.linalg.norm(X[other] - X[i], axis=1)))
            denom = max(a, b, _EPS)
            values.append((b - a) / denom)
        if not values:
            return -1.0
        return float(np.mean(values))

    def _kmeans_midpoint_pass(self, pop: np.ndarray, contrib: dict[str, float], counts: dict[str, int]) -> tuple[np.ndarray, int, float]:
        n = int(pop.shape[0])
        dim = int(self.problem.dimension)
        if n < max(4, int(self._params.get("kmeans_min_population_size", 20))):
            return pop, 0, -1.0
        result = self._simple_kmeans2(pop[:, :-1], int(self._params.get("kmeans_max_iterations", 12)))
        if result is None:
            return pop, 0, -1.0
        labels, _centers = result
        silhouette = self._silhouette2(pop[:, :-1], labels)
        threshold = 1.0 / (float(self._params.get("kmeans_silhouette_denominator", 4.0)) * np.sqrt(max(1, dim)))
        if silhouette <= threshold:
            return pop, 0, silhouette
        evals = 0
        gain = 0.0
        for c in (0, 1):
            cluster = pop[labels == c, :-1]
            if cluster.shape[0] == 0:
                continue
            midpoint = np.mean(cluster, axis=0)
            pop, one_eval, one_gain = self._evaluate_and_maybe_insert_point(pop, midpoint, contrib, counts, "kmeans_midpoint", "midpoint_replacement")
            evals += one_eval
            gain += one_gain
        return pop, evals, silhouette

    def _update_bound_counters(self, pop: np.ndarray, counters: np.ndarray) -> tuple[np.ndarray, bool]:
        n = int(pop.shape[0])
        if counters.shape[0] != n:
            new_counters = np.zeros(n, dtype=int)
            m = min(n, counters.shape[0])
            if m:
                new_counters[:m] = counters[:m]
            counters = new_counters
        on_bounds = np.any((np.isclose(pop[:, :-1], self._lo, atol=0.0, rtol=0.0)) | (np.isclose(pop[:, :-1], self._hi, atol=0.0, rtol=0.0)), axis=1)
        counters[on_bounds] += 1
        counters[~on_bounds] = 0
        trigger = bool(np.any(counters >= int(self._params.get("bound_restart_patience", 10))))
        return counters, trigger

    def _midpoint_restart_trigger(self, history: list[np.ndarray], midpoint: np.ndarray) -> bool:
        patience = max(1, int(self._params.get("midpoint_restart_patience", 9)))
        if len(history) < patience:
            return False
        old = np.asarray(history[-patience], dtype=float)
        tol = float(self._params.get("midpoint_restart_tol", 1.0e-9))
        return bool(np.linalg.norm(np.asarray(midpoint, dtype=float) - old) < tol)

    def _restart_payload_updates(self, state, reason: str, evaluations_so_far: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        restart_n = max(4, int(self._params.get("restart_population_size", 400)))
        h = max(1, int(self._params.get("hist_mem_size", 20 * int(self.problem.dimension))))
        M_F, M_CR = self._reset_memory(h)
        positions = self._new_positions(restart_n)
        pop = self._pop_from_positions(positions)
        if self.config.max_evaluations is not None:
            remaining = max(1, int(self.config.max_evaluations) - int(evaluations_so_far))
        else:
            remaining = max(1, restart_n * max(1, int(self.config.max_steps or 100)))
        self._last_restart_reason = reason
        updates = {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": restart_n,
            "p_archive": float(self._params.get("archive_use_probability", 0.5)),
            "pbest_value": float(self._params.get("pbest_max", 0.4)),
            "binomial_cr": 0.0,
            "midpoint_history": [],
            "bound_counters": np.zeros(restart_n, dtype=int),
            "restart_count": int(state.payload.get("restart_count", 0)) + 1,
            "restart_start_evaluations": int(evaluations_so_far),
            "restart_budget": int(remaining),
            "last_restart_reason": reason,
        }
        return pop, int(restart_n), updates

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

        # Midpoint before mutation, as specified by the paper.
        evals_extra = 0
        pop, midpoint_evals, midpoint, _mid_gain = self._midpoint_pass(pop, contrib, counts)
        evals_extra += midpoint_evals
        kmeans_silhouette = -1.0
        pop, k_evals, kmeans_silhouette = self._kmeans_midpoint_pass(pop, contrib, counts)
        evals_extra += k_evals

        midpoint_history = list(state.payload.get("midpoint_history", []))
        midpoint_trigger = self._midpoint_restart_trigger(midpoint_history, midpoint)
        midpoint_history.append(np.asarray(midpoint, dtype=float))
        # Keep enough history for the restart trigger but avoid unbounded growth.
        keep_history = max(2, int(self._params.get("midpoint_restart_patience", 9)) + 1)
        midpoint_history = midpoint_history[-keep_history:]

        bound_counters = np.asarray(state.payload.get("bound_counters", np.zeros(n, dtype=int)), dtype=int).copy()
        bound_counters, bounds_trigger = self._update_bound_counters(pop, bound_counters)

        evaluations_before_restart_check = int(state.evaluations + evals_extra)
        if midpoint_trigger or bounds_trigger:
            reason = "midpoint" if midpoint_trigger else "bounds"
            self._add(contrib, counts, "midpoint_restart" if midpoint_trigger else "bounds_restart", 0.0, 1)
            self._add(contrib, counts, "restart", 0.0, 1)
            new_pop, restart_evals, updates = self._restart_payload_updates(state, reason, evaluations_before_restart_check)
            self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
            self._last_operator_counts = {key: int(value) for key, value in counts.items()}
            return new_pop, evals_extra + restart_evals, updates

        # Recompute order after possible midpoint insertions.
        n = int(pop.shape[0])
        order = self._order(pop[:, -1])
        pnum = self._pbest_count(n, progress)

        mem_indices = np.random.randint(0, h, size=n)
        Fs = np.array([_positive_cauchy(M_F[idx], float(self._params.get("f_scale", 0.1))) for idx in mem_indices], dtype=float)
        raw_cr = np.array([_normal_cr(M_CR[idx], float(self._params.get("cr_sigma", 0.1))) for idx in mem_indices], dtype=float)
        CRs = raw_cr.copy()
        CRs[order] = np.sort(raw_cr)
        self._add(contrib, counts, "crossover_rate_sorting", 0.0, n)

        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        used_archive = np.zeros(n, dtype=bool)
        used_binomial = np.zeros(n, dtype=bool)
        used_exponential = np.zeros(n, dtype=bool)
        fallback_repair_count = 0
        resampling_attempts = 0

        binomial_probability = float(self._params.get("binomial_crossover_probability", 0.5))
        for i in range(n):
            donor, F_i, CR_i, archive_flag, attempts, fallback = self._sample_donor_resampling(
                pop, archive, i, order, pnum, float(Fs[i]), float(CRs[i]), p_archive, M_F, M_CR
            )
            Fs[i] = F_i
            CRs[i] = CR_i
            used_archive[i] = bool(archive_flag)
            resampling_attempts += int(attempts)
            fallback_repair_count += int(fallback)

            if np.random.random() < binomial_probability:
                trial = self._binomial_crossover(pop[i, :-1], donor, cr_bin)
                used_binomial[i] = True
            else:
                trial = self._exponential_crossover(pop[i, :-1], donor, float(CRs[i]))
                used_exponential[i] = True
            # A trial can still become infeasible through crossover only when donor was repaired by fallback.
            trial, fixed = self._random_bound_repair(trial)
            fallback_repair_count += int(fixed)
            trials[i] = trial

        trial_fit = self._evaluate_population(trials)
        evals = int(n) + int(evals_extra)
        mask = self._better_mask(trial_fit, pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - trial_fit[mask]
        else:
            gains[mask] = trial_fit[mask] - pop[mask, -1]
        positive_gains = np.maximum(gains, 0.0)
        accepted_gain = float(np.sum(positive_gains))

        mutation_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        crossover_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        selection_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        self._add(contrib, counts, "mutation", mutation_gain, n)
        self._add(contrib, counts, "crossover_binomial", crossover_gain * (np.count_nonzero(used_binomial) / max(1, n)), int(np.count_nonzero(used_binomial)))
        self._add(contrib, counts, "crossover_exponential", crossover_gain * (np.count_nonzero(used_exponential) / max(1, n)), int(np.count_nonzero(used_exponential)))
        self._add(contrib, counts, "selection", selection_gain, int(np.count_nonzero(mask)))
        self._add(contrib, counts, "rank_selective_pressure", 0.0, int(np.count_nonzero(~used_archive)))
        self._add(contrib, counts, "bound_resampling", 0.0, resampling_attempts)
        self._add(contrib, counts, "bound_random_repair_fallback", 0.0, fallback_repair_count)

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
        target_n = self._target_population_size(evaluations_after_step, state.payload, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            bound_counters = bound_counters[keep] if bound_counters.shape[0] == keep.shape[0] else np.zeros(target_n, dtype=int)
            self._add(contrib, counts, "nonlinear_population_reduction", 0.0, removed)

        archive = self._truncate_archive(archive, self._archive_capacity(pop.shape[0]))

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "initial_n": int(state.payload.get("initial_n", n)),
            "base_initial_n": int(state.payload.get("base_initial_n", self._n)),
            "p_archive": p_archive,
            "pbest_value": pbest_value,
            "binomial_cr": cr_bin,
            "midpoint_history": midpoint_history,
            "bound_counters": bound_counters,
            "restart_count": int(state.payload.get("restart_count", 0)),
            "restart_start_evaluations": int(state.payload.get("restart_start_evaluations", 0)),
            "restart_budget": state.payload.get("restart_budget", None),
            "last_restart_reason": state.payload.get("last_restart_reason", self._last_restart_reason),
            "kmeans_silhouette": float(kmeans_silhouette),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "direct_improvement_plus_midpoint_insertions"
        obs["evomapx_fidelity"] = "native"
        if "M_F" in state.payload and "M_CR" in state.payload:
            obs["mean_memory_f"] = float(np.nanmean(np.asarray(state.payload["M_F"], dtype=float)))
            obs["mean_memory_cr"] = float(np.nanmean(np.asarray(state.payload["M_CR"], dtype=float)))
        obs["p_archive"] = float(state.payload.get("p_archive", self._params.get("archive_use_probability", 0.5)))
        obs["pbest_value"] = float(state.payload.get("pbest_value", self._params.get("pbest_max", 0.4)))
        obs["binomial_cr"] = float(state.payload.get("binomial_cr", 0.0))
        obs["restart_count"] = int(state.payload.get("restart_count", 0))
        obs["last_restart_reason"] = state.payload.get("last_restart_reason", None)
        obs["kmeans_silhouette"] = float(state.payload.get("kmeans_silhouette", -1.0))
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        return obs
