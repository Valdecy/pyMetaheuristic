"""pyMetaheuristic src — faithful NL-SHADE-LBC Engine.

This module implements the NL-SHADE-LBC variant described by Stanovov,
Akhmedova, and Semenkin (CEC 2022): current-to-pbest/1 binomial DE,
exponential rank-based selective pressure for the population-selected r2,
fixed archive-use probability, nonlinear population-size reduction, crossover
rate sorting, repeated out-of-bounds regeneration, fitness-aware archive
replacement, and linear-bias-change generalized Lehmer memory updates.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """Draw a SHADE-style positive Cauchy value and truncate it to (0, upper]."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    # Extremely unlikely, but avoids returning F <= 0 after 100 failed draws.
    return min(max(center, 0.5, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    if not np.isfinite(center):
        center = 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _generalized_weighted_lehmer(values: np.ndarray, weights: np.ndarray, p: float, m: float = 1.5) -> float:
    """Generalized weighted Lehmer mean from the NL-SHADE-LBC paper.

    mean_{w,p,m}(S) = sum_j w_j S_j^p / sum_j w_j S_j^(p-m)

    Small eps protection is used only to avoid numerical singularities when the
    exponent p-m is negative and a successful CR happens to be zero.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    safe_values = np.maximum(values, 1.0e-12)
    numerator = float(np.sum(weights * np.power(safe_values, float(p))))
    denominator = float(np.sum(weights * np.power(safe_values, float(p) - float(m))))
    if abs(denominator) <= _EPS or not np.isfinite(denominator):
        return float(np.average(values, weights=weights))
    value = numerator / denominator
    return float(np.clip(value, 0.0, 1.0))


class NLSHADELbcEngine(PortedPopulationEngine):
    """NL-SHADE-LBC with paper-faithful linear parameter adaptation bias change."""

    algorithm_id = "nlshade_lbc"
    algorithm_name = "NL-SHADE-LBC"
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
        "doi": "10.1109/CEC55065.2022.9870295",
        "title": "NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2022,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults are dimension-dependent. None/"paper"/0 means: NP = 23D, H = 20D.
        population_size=None,
        min_population_size=4,
        hist_mem_size=None,
        memory_f_init=0.5,
        memory_cr_init=0.9,
        archive_rate=1.0,
        archive_use_probability=0.5,
        pbest_initial=0.2,
        pbest_final=0.3,
        rank_pressure_coefficient=4.0,
        f_scale=0.1,
        cr_sigma=0.1,
        lbc_m=1.5,
        lbc_p_f_initial=3.5,
        lbc_p_cr_initial=1.0,
        lbc_p_final=1.5,
        resampling_attempts=100,
        crossover_rate_sorting=True,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            self._n = max(4, 23 * int(self.problem.dimension))
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
            f"{self.algorithm_id}.linear_bias_change",
            f"{self.algorithm_id}.bound_resampling",
            f"{self.algorithm_id}.crossover_rate_sorting",
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
        dim = int(self.problem.dimension)
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.9)), dtype=float),
            "k": 0,
            "archive": np.empty((0, dim), dtype=float),
            "archive_fitness": np.empty(0, dtype=float),
            "initial_n": int(pop.shape[0]),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _pbest_count(self, n: int, progress: float) -> int:
        p0 = float(self._params.get("pbest_initial", 0.2))
        p1 = float(self._params.get("pbest_final", 0.3))
        p = p0 + (p1 - p0) * float(progress)
        return int(np.clip(round(n * p), 2, max(2, n)))

    def _lbc_p_values(self, progress: float) -> tuple[float, float]:
        p_final = float(self._params.get("lbc_p_final", 1.5))
        p_f0 = float(self._params.get("lbc_p_f_initial", 3.5))
        p_cr0 = float(self._params.get("lbc_p_cr_initial", 1.0))
        p_f = p_f0 + (p_final - p_f0) * float(progress)
        p_cr = p_cr0 + (p_final - p_cr0) * float(progress)
        return float(p_f), float(p_cr)

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        if progress <= 0.0:
            target = initial_n
        else:
            # AGSK/NLPSR: NP_{g+1}=round[(NPmin-NPinit)*(NFE/NFEmax)^(1-NFE/NFEmax)+NPinit]
            target = round((min_n - initial_n) * (progress ** (1.0 - progress)) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    # ------------------------------------------------------------------
    # Selection primitives
    # ------------------------------------------------------------------
    def _rank_probabilities(self, n: int, order: np.ndarray) -> np.ndarray:
        # order is best-to-worst. Rank i starts at 1 for the best individual.
        coeff = float(self._params.get("rank_pressure_coefficient", 4.0))
        rank_positions = np.arange(1, n + 1, dtype=float)
        sorted_weights = np.exp(-coeff * rank_positions / float(max(1, n)))
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
    ) -> tuple[np.ndarray, bool]:
        n = int(pop.shape[0])
        archive_probability = float(self._params.get("archive_use_probability", 0.5))
        if archive.shape[0] > 0 and np.random.random() < archive_probability:
            aidx = int(np.random.randint(archive.shape[0]))
            return archive[aidx].copy(), True
        r2_idx = self._choice_population_index(n, {int(i), int(pbest_idx), int(r1_idx)}, order, rank_based=True)
        return pop[r2_idx, :-1].copy(), False

    # ------------------------------------------------------------------
    # Variation and archive logic
    # ------------------------------------------------------------------
    def _make_trial(
        self,
        parent: np.ndarray,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        mem_index: int,
        cr: float,
        progress: float,
        pnum: int,
    ) -> tuple[np.ndarray, float, int, int]:
        """Generate one trial using the paper's repeated bound-regeneration BCHM.

        Returns: trial, final F, number of regeneration attempts beyond the first,
        and whether r2 came from the archive at least once.
        """
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        max_attempts = max(1, int(self._params.get("resampling_attempts", 100)))
        mf = float(pop.shape[0])  # dummy overwritten below; keeps linters quiet
        del mf
        archive_used_any = 0
        last_trial: np.ndarray | None = None
        last_f = 0.5

        for attempt in range(max_attempts):
            F = _positive_cauchy(float(self._current_M_F[mem_index]), float(self._params.get("f_scale", 0.1)))
            pbest_idx = self._choose_pbest_index(order, pnum, i)
            r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)}, order, rank_based=False)
            r2, archive_used = self._choose_r2_vector(pop, archive, i, pbest_idx, r1_idx, order)
            archive_used_any += int(archive_used)
            donor = parent + F * (pop[pbest_idx, :-1] - parent) + F * (pop[r1_idx, :-1] - r2)
            cross = np.random.rand(dim) < float(cr)
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, donor, parent)
            last_trial = trial
            last_f = F
            if np.all((trial >= self._lo) & (trial <= self._hi)):
                return trial.astype(float), float(F), int(attempt), int(archive_used_any > 0)

        # Midpoint target fallback after 100 failed regenerations.
        assert last_trial is not None
        repaired = last_trial.astype(float).copy()
        below = repaired < self._lo
        above = repaired > self._hi
        repaired[below] = 0.5 * (self._lo[below] + parent[below])
        repaired[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(repaired, self._lo, self._hi), float(last_f), int(max_attempts), int(archive_used_any > 0)

    def _archive_insert(
        self,
        archive: np.ndarray,
        archive_fitness: np.ndarray,
        vector: np.ndarray,
        fitness: float,
        capacity: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        capacity = max(1, int(capacity))
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        fitness = float(fitness)
        if archive.shape[0] < capacity:
            archive = np.vstack((archive, vector)) if archive.size else vector.copy()
            archive_fitness = np.append(archive_fitness, fitness)
            return archive, archive_fitness

        # Fitness-aware replacement: try random archive positions until the new
        # point is better; otherwise replace one random archive point.
        size = int(archive.shape[0])
        tried = np.random.permutation(size)
        replace_idx: int | None = None
        for idx in tried:
            if self._is_better(fitness, float(archive_fitness[idx])):
                replace_idx = int(idx)
                break
        if replace_idx is None:
            replace_idx = int(np.random.randint(size))
        archive[replace_idx, :] = vector[0]
        archive_fitness[replace_idx] = fitness
        return archive, archive_fitness

    def _truncate_archive(self, archive: np.ndarray, archive_fitness: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
        capacity = max(1, int(capacity))
        if archive.shape[0] <= capacity:
            return archive, archive_fitness
        keep = np.random.choice(archive.shape[0], capacity, replace=False)
        return archive[keep].copy(), archive_fitness[keep].copy()

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
        archive_fitness = np.asarray(state.payload.get("archive_fitness", np.empty(0)), dtype=float).copy()
        if archive_fitness.shape[0] != archive.shape[0]:
            archive_fitness = np.full(archive.shape[0], self.problem.worst_fitness(), dtype=float)
        h = int(M_F.shape[0])
        k_mem = int(state.payload.get("k", 0)) % h
        initial_n = int(state.payload.get("initial_n", n))

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        progress = self._progress(state.evaluations, state.step)
        order = self._order(pop[:, -1])
        pnum = self._pbest_count(n, progress)

        mem_indices = np.random.randint(0, h, size=n)
        raw_cr = np.array([_normal_cr(M_CR[idx], float(self._params.get("cr_sigma", 0.1))) for idx in mem_indices], dtype=float)
        CRs = raw_cr.copy()
        if bool(self._params.get("crossover_rate_sorting", True)):
            # Crossover-rate sorting: smaller CR for better-ranked individuals,
            # larger CR for worse-ranked individuals.
            CRs[order] = np.sort(raw_cr)
            self._add(contrib, counts, "crossover_rate_sorting", 0.0, n)

        self._current_M_F = M_F  # used inside _make_trial to resample F from the selected memory cell
        parent_before = pop[:, :-1].copy()
        parent_fit_before = pop[:, -1].copy()
        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        resampling_count = 0
        archive_r2_count = 0
        for i in range(n):
            trial, F, n_resampled, used_archive = self._make_trial(
                parent=pop[i, :-1],
                pop=pop,
                archive=archive,
                order=order,
                i=i,
                mem_index=int(mem_indices[i]),
                cr=float(CRs[i]),
                progress=progress,
                pnum=pnum,
            )
            trials[i] = trial
            Fs[i] = F
            resampling_count += int(n_resampled)
            archive_r2_count += int(used_archive)

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        mask = self._better_mask(trial_fit, pop[:, -1])
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = pop[mask, -1] - trial_fit[mask]
        else:
            gains[mask] = trial_fit[mask] - pop[mask, -1]
        accepted_gain = float(np.sum(np.maximum(gains, 0.0)))

        # Direct-improvement telemetry. Control operators are counted separately
        # because their causal effect is delayed and should not be misread as zero use.
        self._add(contrib, counts, "mutation", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, n)
        self._add(contrib, counts, "crossover", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, n)
        self._add(contrib, counts, "selection", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(np.count_nonzero(mask)))
        self._add(contrib, counts, "rank_selective_pressure", 0.0, n - archive_r2_count)
        self._add(contrib, counts, "bound_resampling", 0.0, resampling_count)

        if np.any(mask):
            current_archive_capacity = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * n)))
            for old_x, old_f in zip(parent_before[mask], parent_fit_before[mask]):
                archive, archive_fitness = self._archive_insert(archive, archive_fitness, old_x, float(old_f), current_archive_capacity)
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(mask)))

            sf = Fs[mask]
            scr = CRs[mask]
            df = np.abs(parent_fit_before[mask] - trial_fit[mask])
            weights = df / (float(np.sum(df)) + _EPS)
            p_f, p_cr = self._lbc_p_values(progress)
            m = float(self._params.get("lbc_m", 1.5))
            M_F[k_mem] = _generalized_weighted_lehmer(sf, weights, p_f, m)
            if np.max(scr) <= 0.0:
                M_CR[k_mem] = 0.0
            else:
                M_CR[k_mem] = _generalized_weighted_lehmer(scr, weights, p_cr, m)
            k_mem = (k_mem + 1) % h
            self._add(contrib, counts, "success_history_update", 0.0, 1)
            self._add(contrib, counts, "linear_bias_change", 0.0, 1)

            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        evaluations_after_step = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_step, initial_n, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "population_reduction", 0.0, removed)

        archive_capacity = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * pop.shape[0])))
        archive, archive_fitness = self._truncate_archive(archive, archive_fitness, archive_capacity)

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k_mem,
            "archive": archive,
            "archive_fitness": archive_fitness,
            "initial_n": initial_n,
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
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        return obs
