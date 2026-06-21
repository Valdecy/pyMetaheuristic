"""pyMetaheuristic src — faithful LSHADE-SPACMA Engine.

This module implements Mohamed et al.'s LSHADE-SPACMA idea as a native
pyMetaheuristic engine: LSHADE-SPA semi-parameter adaptation, a shared
population split between LSHADE and CMA-ES-style offspring generation,
FCP memory controlling the split probability, external archive support,
linear population-size reduction, and CMA sampling followed by binomial
crossover.

The original paper hybridized LSHADE-SPA with Hansen's MATLAB CMA-ES code.
Here the CMA side is implemented internally as a compact covariance-adaptive
sampler so the engine remains self-contained in Python while preserving the
published LSHADE-SPACMA data flow and parameter schedules.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """Cauchy F sample: resample while non-positive and cap at ``upper``."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(center, 0.5, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian crossover-rate sample clipped to [0, 1]."""
    if not np.isfinite(center) or center < 0.0:
        return 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted Lehmer mean used for successful F values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.5
    if weights is None:
        weights = np.ones(values.size, dtype=float) / float(values.size)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / (float(np.sum(weights)) + _EPS)
    denom = float(np.sum(weights * values))
    if abs(denom) <= _EPS or not np.isfinite(denom):
        return float(np.clip(np.mean(values), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


def _safe_spd(cov: np.ndarray, dim: int) -> np.ndarray:
    """Return a symmetric positive-definite covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (dim, dim) or not np.all(np.isfinite(cov)):
        return np.eye(dim, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1.0e-12, 1.0e12)
    return (eigvecs * eigvals) @ eigvecs.T


class LSHADESpacmaEngine(PortedPopulationEngine):
    """LSHADE-SPACMA: semi-parameter-adaptive LSHADE hybridized with CMA-ES."""

    algorithm_id = "lshade_spacma"
    algorithm_name = "LSHADE-SPACMA"
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
        "doi": "10.1109/CEC.2017.7969307",
        "title": "LSHADE with Semi-Parameter Adaptation Hybrid with CMA-ES for Solving CEC 2017 Benchmark Problems",
        "authors": "Ali W. Mohamed, Anas A. Hadi, Anas M. Fattouh, Kamal M. Jambi",
        "year": 2017,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults: NP = 18D, Nmin = 4, Pbest = 0.11, H = 5,
        # archive rate = 1.4, FCP initial value = 0.5, learning rate c = 0.8.
        population_size=None,
        min_population_size=4,
        pbest_factor=0.11,
        hist_mem_size=5,
        archive_rate=1.4,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        memory_fcp_init=0.5,
        fcp_learning_rate=0.8,
        fcp_min=0.2,
        fcp_max=0.8,
        cauchy_scale=0.1,
        cr_sigma=0.1,
        spa_switch_fraction=0.5,
        cma_cov_learning_rate=0.20,
        cma_sigma_learning_rate=0.20,
        cma_initial_sigma_fraction=0.30,
        cma_min_sigma_fraction=1.0e-6,
        cma_max_sigma_fraction=2.0,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            d = max(1, int(self.problem.dimension))
            self._n = max(4, int(round(18.0 * float(d))))
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
            f"{self.algorithm_id}.cma_es_sampling",
            f"{self.algorithm_id}.cma_es_update",
            f"{self.algorithm_id}.semi_parameter_adaptation",
            f"{self.algorithm_id}.fcp_assignment",
            f"{self.algorithm_id}.fcp_memory_update",
            f"{self.algorithm_id}.lshade_branch",
            f"{self.algorithm_id}.cma_branch",
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
        h = max(1, int(self._params.get("hist_mem_size", 5)))
        self._params["hist_mem_size"] = h
        dim = int(self.problem.dimension)
        order = self._order(pop[:, -1])
        mu = max(1, min(pop.shape[0], pop.shape[0] // 2))
        elite = pop[order[:mu], :-1]
        mean = np.mean(elite, axis=0) if elite.size else np.mean(pop[:, :-1], axis=0)
        sigma = float(self._params.get("cma_initial_sigma_fraction", 0.30)) * float(np.mean(self._span))
        sigma = max(float(sigma), float(np.mean(self._span)) * 1.0e-12)
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5)), dtype=float),
            "M_FCP": np.full(h, float(self._params.get("memory_fcp_init", 0.5)), dtype=float),
            "k": 0,
            "archive": np.empty((0, dim), dtype=float),
            "initial_n": int(pop.shape[0]),
            "cma_mean": np.asarray(mean, dtype=float),
            "cma_cov": np.eye(dim, dtype=float),
            "cma_sigma": sigma,
            "first_half_f_means": [],
            "mf_initialized_from_first_half": False,
            "last_fcp_delta": 0.5,
            "last_lshade_count": 0,
            "last_cma_count": 0,
            "last_spa_phase": "first",
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _in_second_spa_phase(self, progress: float) -> bool:
        return bool(progress >= float(self._params.get("spa_switch_fraction", 0.5)))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        target = round(((min_n - initial_n) * progress) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    # ------------------------------------------------------------------
    # LSHADE and CMA primitives
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

    def _choose_archive_or_population_r2(self, pop: np.ndarray, archive: np.ndarray, exclude: set[int]) -> np.ndarray:
        n = int(pop.shape[0])
        if archive.shape[0] > 0:
            # LSHADE selects r2 from P union A. This probability is equivalent to
            # sampling uniformly from the concatenated set.
            archive_probability = archive.shape[0] / float(n + archive.shape[0])
            if np.random.random() < archive_probability:
                return archive[int(np.random.randint(archive.shape[0]))].copy()
        idx = self._choice_population_index(n, exclude)
        return pop[idx, :-1].copy()

    def _lshade_donor(self, parent: np.ndarray, pop: np.ndarray, archive: np.ndarray, order: np.ndarray, i: int, F: float) -> np.ndarray:
        n = int(pop.shape[0])
        p_rate = float(self._params.get("pbest_factor", 0.11))
        pnum = int(np.clip(round(p_rate * n), 2, max(2, n)))
        pbest_idx = self._choose_pbest_index(order, pnum, i)
        r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)})
        r2 = self._choose_archive_or_population_r2(pop, archive, {int(i), int(pbest_idx), int(r1_idx)})
        return parent + float(F) * (pop[pbest_idx, :-1] - parent) + float(F) * (pop[r1_idx, :-1] - r2)

    def _cma_sample(self, mean: np.ndarray, cov: np.ndarray, sigma: float, dim: int) -> np.ndarray:
        cov = _safe_spd(cov, dim)
        z = np.random.multivariate_normal(np.zeros(dim, dtype=float), cov)
        return np.asarray(mean, dtype=float) + float(sigma) * z

    def _crossover_and_repair(self, parent: np.ndarray, donor: np.ndarray, CR: float) -> tuple[np.ndarray, int]:
        dim = int(self.problem.dimension)
        cross = np.random.rand(dim) < float(CR)
        cross[np.random.randint(dim)] = True
        trial = np.where(cross, donor, parent)
        out = (trial < self._lo) | (trial > self._hi)
        repaired_count = int(np.count_nonzero(out))
        if repaired_count > 0:
            # LSHADE-family midpoint target repair for bound violations.
            below = trial < self._lo
            above = trial > self._hi
            trial = trial.copy()
            trial[below] = 0.5 * (self._lo[below] + parent[below])
            trial[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(trial, self._lo, self._hi), repaired_count

    def _sample_f_cr_fcp(self, M_F: np.ndarray, M_CR: np.ndarray, M_FCP: np.ndarray, mem_idx: int, second_phase: bool) -> tuple[float, float, float]:
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        if second_phase:
            F = _positive_cauchy(float(M_F[mem_idx]), float(self._params.get("cauchy_scale", 0.1)))
        else:
            F = float(0.45 + 0.1 * np.random.random())
        fcp = float(np.clip(M_FCP[mem_idx], float(self._params.get("fcp_min", 0.2)), float(self._params.get("fcp_max", 0.8))))
        return F, CR, fcp

    def _maybe_initialize_mf_from_first_half(self, M_F: np.ndarray, payload: dict[str, Any]) -> tuple[np.ndarray, bool]:
        if bool(payload.get("mf_initialized_from_first_half", False)):
            return M_F, False
        values = list(payload.get("first_half_f_means", []))[-int(M_F.shape[0]) :]
        if not values:
            return M_F, False
        arr = np.asarray(values, dtype=float)
        if arr.size < M_F.shape[0]:
            arr = np.pad(arr, (M_F.shape[0] - arr.size, 0), mode="edge")
        M_F[:] = np.clip(arr[-M_F.shape[0] :], 0.0, 1.0)
        payload["mf_initialized_from_first_half"] = True
        return M_F, True

    def _update_cma(self, pop: np.ndarray, old_mean: np.ndarray, old_cov: np.ndarray, old_sigma: float, cma_success_rate: float) -> tuple[np.ndarray, np.ndarray, float]:
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        order = self._order(pop[:, -1])
        mu = max(2, min(n, n // 2))
        elite = pop[order[:mu], :-1]
        # Logarithmic positive CMA weights.
        ranks = np.arange(1, mu + 1, dtype=float)
        weights = np.log(mu + 0.5) - np.log(ranks)
        weights = np.maximum(weights, 0.0)
        if float(np.sum(weights)) <= _EPS:
            weights = np.ones(mu, dtype=float)
        weights = weights / float(np.sum(weights))
        new_mean = np.sum(elite * weights[:, None], axis=0)
        centered = elite - new_mean
        denom = max(float(old_sigma) ** 2, 1.0e-30)
        sample_cov = (centered.T @ (centered * weights[:, None])) / denom
        c_cov = float(np.clip(self._params.get("cma_cov_learning_rate", 0.20), 0.0, 1.0))
        new_cov = (1.0 - c_cov) * _safe_spd(old_cov, dim) + c_cov * _safe_spd(sample_cov, dim)
        new_cov = _safe_spd(new_cov, dim)
        # Lightweight one-fifth-style step-size feedback using only the CMA branch
        # success rate; this keeps the self-contained port stable without importing
        # Hansen's MATLAB implementation.
        c_sig = float(np.clip(self._params.get("cma_sigma_learning_rate", 0.20), 0.0, 1.0))
        sigma = float(old_sigma) * float(np.exp(c_sig * (float(cma_success_rate) - 0.2)))
        mean_span = float(np.mean(self._span))
        sigma = float(np.clip(sigma, mean_span * float(self._params.get("cma_min_sigma_fraction", 1.0e-6)), mean_span * float(self._params.get("cma_max_sigma_fraction", 2.0))))
        if not np.all(np.isfinite(new_mean)):
            new_mean = np.asarray(old_mean, dtype=float)
        return np.clip(new_mean, self._lo, self._hi), new_cov, sigma

    # ------------------------------------------------------------------
    # Main native step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        if n < 4:
            raise ValueError(f"{self.algorithm_id} requires at least four population members.")

        M_F = np.asarray(state.payload.get("M_F"), dtype=float).copy()
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float).copy()
        M_FCP = np.asarray(state.payload.get("M_FCP"), dtype=float).copy()
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim).copy()
        cma_mean = np.asarray(state.payload.get("cma_mean", np.mean(pop[:, :-1], axis=0)), dtype=float).copy()
        cma_cov = _safe_spd(np.asarray(state.payload.get("cma_cov", np.eye(dim))), dim)
        cma_sigma = float(state.payload.get("cma_sigma", float(np.mean(self._span)) * 0.3))
        initial_n = int(state.payload.get("initial_n", n))
        h = int(M_F.shape[0])
        k_mem = int(state.payload.get("k", 0)) % max(1, h)
        first_half_f_means = list(state.payload.get("first_half_f_means", []))

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state.evaluations, state.step)
        second_phase = self._in_second_spa_phase(progress)
        if second_phase:
            payload_proxy = {
                "first_half_f_means": first_half_f_means,
                "mf_initialized_from_first_half": bool(state.payload.get("mf_initialized_from_first_half", False)),
            }
            M_F, initialized_mf = self._maybe_initialize_mf_from_first_half(M_F, payload_proxy)
            mf_initialized_from_first_half = bool(payload_proxy["mf_initialized_from_first_half"])
        else:
            initialized_mf = False
            mf_initialized_from_first_half = bool(state.payload.get("mf_initialized_from_first_half", False))

        order = self._order(pop[:, -1])
        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()

        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        FCPs = np.empty(n, dtype=float)
        branch_is_lshade = np.empty(n, dtype=bool)
        donors = np.empty((n, dim), dtype=float)
        trials = np.empty((n, dim), dtype=float)
        repair_count = 0

        for i in range(n):
            mem_idx = int(np.random.randint(h))
            F, CR, FCP = self._sample_f_cr_fcp(M_F, M_CR, M_FCP, mem_idx, second_phase)
            Fs[i], CRs[i], FCPs[i] = F, CR, FCP
            # FCP is the class probability of assigning the individual to the
            # LSHADE-SPA branch; 1-FCP is assigned to the CMA branch.
            use_lshade = bool(np.random.random() < FCP)
            branch_is_lshade[i] = use_lshade
            if use_lshade:
                donors[i] = self._lshade_donor(pop[i, :-1], pop, archive, order, i, F)
            else:
                donors[i] = self._cma_sample(cma_mean, cma_cov, cma_sigma, dim)
            trials[i], repaired = self._crossover_and_repair(pop[i, :-1], donors[i], CR)
            repair_count += repaired

        lshade_count = int(np.count_nonzero(branch_is_lshade))
        cma_count = int(n - lshade_count)
        self._add(contrib, counts, "fcp_assignment", 0.0, int(n))
        self._add(contrib, counts, "lshade_branch", 0.0, lshade_count)
        self._add(contrib, counts, "cma_branch", 0.0, cma_count)
        self._add(contrib, counts, "mutation", 0.0, lshade_count)
        self._add(contrib, counts, "cma_es_sampling", 0.0, cma_count)
        self._add(contrib, counts, "semi_parameter_adaptation", 0.0, int(n))
        self._add(contrib, counts, "crossover", 0.0, int(n))
        self._add(contrib, counts, "bound_repair", 0.0, repair_count)
        if initialized_mf:
            self._add(contrib, counts, "semi_parameter_adaptation", 0.0, 1)

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        if self.problem.objective == "min":
            mask = trial_fit <= pop[:, -1]
            strict_gain = np.maximum(pop[:, -1] - trial_fit, 0.0)
        else:
            mask = trial_fit >= pop[:, -1]
            strict_gain = np.maximum(trial_fit - pop[:, -1], 0.0)
        accepted_gain = float(np.sum(strict_gain[mask]))

        # Attribute direct accepted improvement to candidate generation,
        # recombination, and greedy selection. Control operators are counted
        # separately because their effect is delayed.
        gen_gain = accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0
        contrib[f"{self.algorithm_id}.mutation"] += gen_gain * (lshade_count / max(1, n))
        contrib[f"{self.algorithm_id}.cma_es_sampling"] += gen_gain * (cma_count / max(1, n))
        contrib[f"{self.algorithm_id}.crossover"] += gen_gain
        contrib[f"{self.algorithm_id}.selection"] += gen_gain
        counts[f"{self.algorithm_id}.selection"] += int(np.count_nonzero(mask))

        strict_mask = mask & (strict_gain > 0.0)
        if np.any(mask):
            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        if np.any(strict_mask):
            old_success = parent_before[strict_mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.4)) * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(strict_mask)))

        # Store first-half F values for initializing MF at the SPA transition.
        if not second_phase:
            if np.any(strict_mask):
                first_half_f_means.append(float(np.mean(Fs[strict_mask])))
            else:
                first_half_f_means.append(float(np.mean(Fs)))
            first_half_f_means = first_half_f_means[-5:]

        # Success-history memory update: M_CR and M_FCP every generation with
        # successes; M_F only in the second SPA part.
        if np.any(strict_mask):
            gain_success = strict_gain[strict_mask]
            weights = gain_success / (float(np.sum(gain_success)) + _EPS)
            S_CR = CRs[strict_mask]
            M_CR[k_mem] = float(np.clip(np.mean(S_CR), 0.0, 1.0))
            if second_phase:
                S_F = Fs[strict_mask]
                M_F[k_mem] = _weighted_lehmer(S_F, weights)
            # FCP memory is based on relative branch improvement, clipped to
            # [0.2, 0.8] so both LSHADE and CMA remain active.
            l_gain = float(np.sum(strict_gain[strict_mask & branch_is_lshade]))
            c_gain = float(np.sum(strict_gain[strict_mask & (~branch_is_lshade)]))
            if (l_gain + c_gain) > _EPS:
                delta_lshade = float(np.clip(l_gain / (l_gain + c_gain), float(self._params.get("fcp_min", 0.2)), float(self._params.get("fcp_max", 0.8))))
                learn = float(np.clip(self._params.get("fcp_learning_rate", 0.8), 0.0, 1.0))
                M_FCP[k_mem] = float(np.clip((1.0 - learn) * M_FCP[k_mem] + learn * delta_lshade, float(self._params.get("fcp_min", 0.2)), float(self._params.get("fcp_max", 0.8))))
            else:
                delta_lshade = float(state.payload.get("last_fcp_delta", 0.5))
            k_mem = (k_mem + 1) % max(1, h)
            self._add(contrib, counts, "success_history_update", 0.0, 1)
            self._add(contrib, counts, "fcp_memory_update", 0.0, 1)
        else:
            delta_lshade = float(state.payload.get("last_fcp_delta", 0.5))

        cma_attempts = max(1, int(cma_count))
        cma_successes = int(np.count_nonzero(strict_mask & (~branch_is_lshade)))
        cma_success_rate = cma_successes / float(cma_attempts)
        cma_mean, cma_cov, cma_sigma = self._update_cma(pop, cma_mean, cma_cov, cma_sigma, cma_success_rate)
        self._add(contrib, counts, "cma_es_update", 0.0, 1)

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
            "M_FCP": M_FCP,
            "k": k_mem,
            "archive": archive,
            "initial_n": initial_n,
            "cma_mean": cma_mean,
            "cma_cov": cma_cov,
            "cma_sigma": float(cma_sigma),
            "first_half_f_means": first_half_f_means,
            "mf_initialized_from_first_half": bool(mf_initialized_from_first_half),
            "last_fcp_delta": float(delta_lshade),
            "last_lshade_count": lshade_count,
            "last_cma_count": cma_count,
            "last_spa_phase": "second" if second_phase else "first",
            "last_bound_repair_count": int(repair_count),
            "last_cma_success_rate": float(cma_success_rate),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "direct_improvement"
        obs["evomapx_fidelity"] = "native"
        if "M_F" in state.payload:
            obs["mean_memory_f"] = float(np.nanmean(np.asarray(state.payload["M_F"], dtype=float)))
        if "M_CR" in state.payload:
            obs["mean_memory_cr"] = float(np.nanmean(np.asarray(state.payload["M_CR"], dtype=float)))
        if "M_FCP" in state.payload:
            obs["mean_memory_fcp"] = float(np.nanmean(np.asarray(state.payload["M_FCP"], dtype=float)))
            obs["memory_fcp"] = [float(v) for v in np.asarray(state.payload["M_FCP"], dtype=float)]
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        obs["cma_sigma"] = float(state.payload.get("cma_sigma", 0.0))
        obs["last_fcp_delta"] = float(state.payload.get("last_fcp_delta", 0.5))
        obs["last_lshade_count"] = int(state.payload.get("last_lshade_count", 0))
        obs["last_cma_count"] = int(state.payload.get("last_cma_count", 0))
        obs["last_spa_phase"] = str(state.payload.get("last_spa_phase", "unknown"))
        obs["last_bound_repair_count"] = int(state.payload.get("last_bound_repair_count", 0))
        obs["last_cma_success_rate"] = float(state.payload.get("last_cma_success_rate", 0.0))
        obs["mf_initialized_from_first_half"] = bool(state.payload.get("mf_initialized_from_first_half", False))
        return obs
