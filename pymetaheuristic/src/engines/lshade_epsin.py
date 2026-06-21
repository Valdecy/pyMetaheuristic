"""pyMetaheuristic src — faithful LSHADE-EpSin Engine.

This module implements Awad et al.'s LSHADE-EpSin as a native
pyMetaheuristic engine: L-SHADE current-to-pbest/1 mutation with external
archive, ensemble sinusoidal scaling-factor adaptation in the first half of
the search, L-SHADE success-history F/CR adaptation in the second half, linear
population-size reduction, and the late Gaussian-Walk local search triggered
when the population size reaches 20.

The paper uses CEC-style generation/budget notation. This implementation uses
function-evaluation progress when max_evaluations is provided and falls back to
step progress otherwise; this preserves the published phase schedules inside
pyMetaheuristic's macro-step protocol.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, *, upper: float = 1.0) -> float:
    """SHADE-style Cauchy sample for F: resample while non-positive; cap at upper."""
    center = float(center)
    scale = float(scale)
    for _ in range(100):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, upper)
    return min(max(center, 0.5, 1.0e-12), upper)


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Gaussian CR sample clipped to [0, 1]."""
    if not np.isfinite(center) or center < 0.0:
        return 0.0
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted Lehmer mean used by SHADE-family memories."""
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
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denom, 0.0, 1.0))


class LSHADEEpSinEngine(PortedPopulationEngine):
    """LSHADE-EpSin: ensemble sinusoidal parameter adaptation with late local search."""

    algorithm_id = "lshade_epsin"
    algorithm_name = "LSHADE-EpSin"
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
        "doi": "10.1109/CEC.2016.7744313",
        "title": "An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems",
        "authors": "Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Robert G. Reynolds",
        "year": 2016,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper defaults: NPmax = 18D, NPmin = 4, H = 5, fixed sinusoidal
        # frequency = 0.5, and Gaussian-walk local search length GLS = 250.
        population_size=None,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        memory_freq_init=0.5,
        pbest_fraction=0.10,
        archive_rate=1.0,
        fixed_frequency=0.5,
        cauchy_scale=0.1,
        cr_sigma=0.1,
        phase_switch_fraction=0.5,
        local_search_trigger_n=20,
        local_search_group_size=10,
        local_search_generations=250,
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
            f"{self.algorithm_id}.sinusoidal_decreasing_f",
            f"{self.algorithm_id}.sinusoidal_increasing_f",
            f"{self.algorithm_id}.adaptive_frequency_update",
            f"{self.algorithm_id}.lshade_second_phase_adaptation",
            f"{self.algorithm_id}.gaussian_walk_local_search",
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
        # The paper states that the archive is initially filled with P0.
        archive = np.asarray(pop[:, :-1], dtype=float).copy()
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5)), dtype=float),
            "M_FREQ": np.full(h, float(self._params.get("memory_freq_init", 0.5)), dtype=float),
            "k": 0,
            "k_freq": 0,
            "archive": archive,
            "initial_n": int(pop.shape[0]),
            "local_search_done": False,
            "last_phase": "sinusoidal_first_half",
            "last_local_search_evals": 0,
            "last_bound_repairs": 0,
            "last_frequency_mean": float(self._params.get("memory_freq_init", 0.5)),
        }

    def _progress(self, evaluations: int | float, step: int | None = None) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(float(evaluations) / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step_value = 0 if step is None else int(step)
        return float(np.clip(step_value / float(horizon), 0.0, 1.0))

    def _in_second_phase(self, progress: float) -> bool:
        return bool(progress >= float(self._params.get("phase_switch_fraction", 0.5)))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step)
        target = round(((min_n - initial_n) * progress) + initial_n)
        return int(np.clip(target, min_n, max(min_n, initial_n, current_n)))

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------
    def _choice_population_index(self, n: int, exclude: set[int]) -> int:
        candidates = np.array([idx for idx in range(n) if idx not in exclude], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        return int(np.random.choice(candidates))

    def _choose_pbest_index(self, order: np.ndarray, n: int, i: int) -> int:
        # Fig. 3 line 22: p is random in the top NP*0.1 individuals. We implement
        # this as a random pbest pool size in [2, ceil(0.1*NP)].
        frac = float(self._params.get("pbest_fraction", 0.10))
        max_top = max(2, int(np.ceil(frac * float(n))))
        pnum = int(np.random.randint(2, max_top + 1)) if max_top >= 2 else 2
        pnum = int(np.clip(pnum, 1, n))
        top = np.asarray(order[:pnum], dtype=int)
        top = top[top != int(i)]
        if top.size == 0:
            top = np.asarray(order[:pnum], dtype=int)
        return int(np.random.choice(top))

    def _sample_r2_vector(self, pop: np.ndarray, archive: np.ndarray, exclude: set[int]) -> np.ndarray:
        n = int(pop.shape[0])
        joint_size = n + int(archive.shape[0])
        if joint_size <= 0:
            return pop[self._choice_population_index(n, exclude), :-1]
        for _ in range(100):
            idx = int(np.random.randint(joint_size))
            if idx < n:
                if idx not in exclude:
                    return np.asarray(pop[idx, :-1], dtype=float)
            else:
                return np.asarray(archive[idx - n], dtype=float)
        return np.asarray(pop[self._choice_population_index(n, exclude), :-1], dtype=float)

    def _nonadaptive_decreasing_f(self, progress: float) -> float:
        freq = float(self._params.get("fixed_frequency", 0.5))
        # Eq. (3): 1/2 * (sin(2*pi*freq*g + pi) * ((Gmax-g)/Gmax) + 1).
        return float(np.clip(0.5 * (np.sin(2.0 * np.pi * freq * progress + np.pi) * (1.0 - progress) + 1.0), 0.0, 1.0))

    def _adaptive_increasing_f(self, progress: float, M_FREQ: np.ndarray) -> tuple[float, float]:
        h = int(M_FREQ.size)
        mem_idx = int(np.random.randint(h))
        freq = _positive_cauchy(float(M_FREQ[mem_idx]), float(self._params.get("cauchy_scale", 0.1)), upper=1.0)
        # Eq. (4): 1/2 * (sin(2*pi*freq*g) * g/Gmax + 1).
        F = 0.5 * (np.sin(2.0 * np.pi * freq * progress) * progress + 1.0)
        return float(np.clip(F, 0.0, 1.0)), float(freq)

    def _sample_first_phase_parameters(self, M_CR: np.ndarray, M_FREQ: np.ndarray, progress: float) -> tuple[float, float, str, float | None]:
        # First half: ensemble sinusoidal F. CR is sampled by the success-history
        # machinery as in L-SHADE.
        if np.random.rand() < 0.5:
            F = self._nonadaptive_decreasing_f(progress)
            source = "sinusoidal_decreasing_f"
            freq = None
        else:
            F, freq = self._adaptive_increasing_f(progress, M_FREQ)
            source = "sinusoidal_increasing_f"
        mem_idx = int(np.random.randint(int(M_CR.size)))
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        return F, CR, source, freq

    def _sample_second_phase_parameters(self, M_F: np.ndarray, M_CR: np.ndarray) -> tuple[float, float]:
        h = int(M_F.size)
        mem_idx = int(np.random.randint(h))
        F = _positive_cauchy(float(M_F[mem_idx]), float(self._params.get("cauchy_scale", 0.1)), upper=1.0)
        CR = _normal_cr(float(M_CR[mem_idx]), float(self._params.get("cr_sigma", 0.1)))
        return F, CR

    def _make_trial(
        self,
        parent: np.ndarray,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        F: float,
        CR: float,
    ) -> tuple[np.ndarray, int]:
        n, dim = pop.shape[0], int(self.problem.dimension)
        x_i = np.asarray(parent, dtype=float)
        pbest_idx = self._choose_pbest_index(order, n, i)
        r1_idx = self._choice_population_index(n, {int(i), int(pbest_idx)})
        r2_vec = self._sample_r2_vector(pop, archive, {int(i), int(pbest_idx), int(r1_idx)})
        donor = x_i + float(F) * (pop[pbest_idx, :-1] - x_i) + float(F) * (pop[r1_idx, :-1] - r2_vec)
        jrand = int(np.random.randint(dim))
        mask = np.random.rand(dim) <= float(CR)
        mask[jrand] = True
        trial = np.where(mask, donor, x_i)
        repaired = int(np.count_nonzero((trial < self._lo) | (trial > self._hi)))
        trial = np.clip(trial, self._lo, self._hi)
        return trial, repaired

    # ------------------------------------------------------------------
    # Memory and local search
    # ------------------------------------------------------------------
    def _update_success_memory(
        self,
        M_F: np.ndarray,
        M_CR: np.ndarray,
        M_FREQ: np.ndarray,
        k_mem: int,
        k_freq: int,
        sf: list[float],
        scr: list[float],
        sfreq: list[float],
        df: list[float],
        *,
        second_phase: bool,
    ) -> tuple[int, int, bool, bool]:
        if not df:
            return k_mem, k_freq, False, False
        weights = np.asarray(df, dtype=float)
        weights = weights / (float(np.sum(weights)) + _EPS)
        updated_main = False
        updated_freq = False
        h = int(M_F.size)
        if scr:
            M_CR[k_mem] = _weighted_lehmer(np.asarray(scr, dtype=float), weights[: len(scr)])
            updated_main = True
        if second_phase and sf:
            M_F[k_mem] = _weighted_lehmer(np.asarray(sf, dtype=float), weights[: len(sf)])
            updated_main = True
        if updated_main:
            k_mem = (int(k_mem) + 1) % h
        if sfreq:
            fweights = weights[: len(sfreq)]
            M_FREQ[k_freq] = _weighted_lehmer(np.asarray(sfreq, dtype=float), fweights)
            k_freq = (int(k_freq) + 1) % int(M_FREQ.size)
            updated_freq = True
        return k_mem, k_freq, updated_main, updated_freq

    def _remaining_budget(self, evals_used: int) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(0, int(self.config.max_evaluations) - int(evals_used))

    def _gaussian_walk_local_search(
        self,
        pop: np.ndarray,
        state_evals_after_generation: int,
        contrib: dict[str, float],
        counts: dict[str, int],
    ) -> tuple[np.ndarray, int]:
        group_size = max(2, int(self._params.get("local_search_group_size", 10)))
        gls = max(1, int(self._params.get("local_search_generations", 250)))
        dim = int(self.problem.dimension)
        walkers = self._new_positions(group_size)
        walker_fit = self._evaluate_population(walkers)
        evals = group_size
        remaining = self._remaining_budget(state_evals_after_generation + evals)
        if remaining is not None:
            max_extra_generations = max(0, remaining // group_size)
            gls = min(gls, max_extra_generations)
        for g in range(1, gls + 1):
            bidx = self._best_index(walker_fit)
            x_best = walkers[bidx].copy()
            sigma_vec = np.abs((np.log(float(g) + 1.0) / (float(g) + 1.0)) * (walkers - x_best))
            sigma_vec = np.maximum(sigma_vec, 1.0e-12 * self._span)
            eps = np.random.rand(group_size, dim)
            eps_hat = np.random.rand(group_size, dim)
            gaussian = np.random.normal(loc=x_best, scale=sigma_vec)
            candidates = gaussian + (eps * x_best - eps_hat * walkers)
            candidates = np.clip(candidates, self._lo, self._hi)
            cand_fit = self._evaluate_population(candidates)
            evals += group_size
            mask = self._better_mask(cand_fit, walker_fit)
            if np.any(mask):
                walkers[mask] = candidates[mask]
                walker_fit[mask] = cand_fit[mask]
        # Try to replace the worst group_size individuals in the main population.
        pop = pop.copy()
        worst_order = self._order(pop[:, -1])[::-1]
        replace_k = min(group_size, pop.shape[0], walkers.shape[0])
        worder = self._order(walker_fit)[:replace_k]
        for source_idx, target_idx in zip(worder, worst_order[:replace_k]):
            if self._is_better(float(walker_fit[source_idx]), float(pop[target_idx, -1])):
                gain = abs(float(pop[target_idx, -1] - walker_fit[source_idx]))
                pop[target_idx, :-1] = walkers[source_idx]
                pop[target_idx, -1] = walker_fit[source_idx]
                self._add(contrib, counts, "gaussian_walk_local_search", gain, 1)
        # Count all local-search objective calls as operator applications.
        self._add(contrib, counts, "gaussian_walk_local_search", 0.0, evals)
        return pop, int(evals)

    # ------------------------------------------------------------------
    # Main macro-step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        M_F = np.asarray(state.payload.get("M_F"), dtype=float).copy()
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float).copy()
        M_FREQ = np.asarray(state.payload.get("M_FREQ"), dtype=float).copy()
        k_mem = int(state.payload.get("k", 0)) % int(M_F.size)
        k_freq = int(state.payload.get("k_freq", 0)) % int(M_FREQ.size)
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        initial_n = int(state.payload.get("initial_n", n))
        progress = self._progress(state.evaluations, state.step)
        second_phase = self._in_second_phase(progress)

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        order = self._order(pop[:, -1])
        parent_positions = pop[:, :-1].copy()
        parent_fit = pop[:, -1].copy()

        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        f_sources: list[str] = []
        sampled_freqs: list[float | None] = []
        repairs = 0

        for i in range(n):
            if second_phase:
                F, CR = self._sample_second_phase_parameters(M_F, M_CR)
                f_source = "lshade_second_phase_adaptation"
                freq = None
            else:
                F, CR, f_source, freq = self._sample_first_phase_parameters(M_CR, M_FREQ, progress)
            trial, repaired = self._make_trial(pop[i, :-1], pop, archive, order, i, F, CR)
            trials[i] = trial
            Fs[i] = F
            CRs[i] = CR
            f_sources.append(f_source)
            sampled_freqs.append(freq)
            repairs += int(repaired)
            self._add(contrib, counts, f_source, 0.0, 1)

        trial_fit = self._evaluate_population(trials)
        evals = int(n)
        if self.problem.objective == "min":
            mask = trial_fit <= pop[:, -1]
            strict_gain = np.maximum(pop[:, -1] - trial_fit, 0.0)
        else:
            mask = trial_fit >= pop[:, -1]
            strict_gain = np.maximum(trial_fit - pop[:, -1], 0.0)
        accepted_gain = float(np.sum(strict_gain[mask]))

        self._add(contrib, counts, "mutation", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, n)
        self._add(contrib, counts, "crossover", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, n)
        self._add(contrib, counts, "selection", accepted_gain / 3.0 if accepted_gain > 0.0 else 0.0, int(np.count_nonzero(mask)))
        self._add(contrib, counts, "bound_repair", 0.0, int(repairs))

        strict_mask = mask & (strict_gain > 0.0)
        if np.any(mask):
            pop[mask, :-1] = trials[mask]
            pop[mask, -1] = trial_fit[mask]

        sf: list[float] = []
        scr: list[float] = []
        sfreq: list[float] = []
        df: list[float] = []
        if np.any(strict_mask):
            old_success = parent_positions[strict_mask]
            archive = np.vstack((archive, old_success)) if archive.size else old_success.copy()
            max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * n)))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            self._add(contrib, counts, "archive_update", 0.0, int(np.count_nonzero(strict_mask)))

            success_indices = np.flatnonzero(strict_mask)
            for idx in success_indices:
                sf.append(float(Fs[idx]))
                scr.append(float(CRs[idx]))
                df.append(float(abs(parent_fit[idx] - trial_fit[idx])))
                if sampled_freqs[idx] is not None and f_sources[idx] == "sinusoidal_increasing_f":
                    sfreq.append(float(sampled_freqs[idx]))

        k_mem, k_freq, updated_main, updated_freq = self._update_success_memory(
            M_F,
            M_CR,
            M_FREQ,
            k_mem,
            k_freq,
            sf,
            scr,
            sfreq,
            df,
            second_phase=second_phase,
        )
        if updated_main:
            self._add(contrib, counts, "success_history_update", 0.0, 1)
        if updated_freq:
            self._add(contrib, counts, "adaptive_frequency_update", 0.0, 1)

        evaluations_after_generation = int(state.evaluations + evals)
        target_n = self._target_population_size(evaluations_after_generation, initial_n, pop.shape[0], state.step + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._add(contrib, counts, "population_reduction", 0.0, removed)

        # Keep archive capacity equal to current NP, as stated for LSHADE-EpSin.
        max_arc = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * pop.shape[0])))
        if archive.shape[0] > max_arc:
            archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]

        local_evals = 0
        local_trigger_n = int(self._params.get("local_search_trigger_n", 20))
        local_done = bool(state.payload.get("local_search_done", False))
        if (not local_done) and pop.shape[0] <= local_trigger_n:
            pop, local_evals = self._gaussian_walk_local_search(pop, evaluations_after_generation, contrib, counts)
            evals += int(local_evals)
            local_done = True

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, int(evals), {
            "M_F": M_F,
            "M_CR": M_CR,
            "M_FREQ": M_FREQ,
            "k": int(k_mem),
            "k_freq": int(k_freq),
            "archive": archive,
            "initial_n": initial_n,
            "local_search_done": bool(local_done),
            "last_phase": "lshade_second_half" if second_phase else "sinusoidal_first_half",
            "last_local_search_evals": int(local_evals),
            "last_bound_repairs": int(repairs),
            "last_frequency_mean": float(np.nanmean(M_FREQ)),
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
        if "M_FREQ" in state.payload:
            obs["mean_memory_frequency"] = float(np.nanmean(np.asarray(state.payload["M_FREQ"], dtype=float)))
        if "archive" in state.payload:
            obs["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        obs["last_phase"] = str(state.payload.get("last_phase", "unknown"))
        obs["local_search_done"] = bool(state.payload.get("local_search_done", False))
        obs["last_local_search_evals"] = int(state.payload.get("last_local_search_evals", 0))
        obs["last_bound_repairs"] = int(state.payload.get("last_bound_repairs", 0))
        return obs
