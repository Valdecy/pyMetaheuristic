"""pyMetaheuristic src — paper-faithful LSHADE-cnEpSin engine.

Implements Awad, Ali, and Suganthan's LSHADE-cnEpSin: current-to-pbest/1
mutation with an external archive, performance-adapted sinusoidal scaling in
the first half of the run, SHADE success-history adaptation in the second
half, covariance-eigenvector crossover using a Euclidean neighbourhood around
the current best, and linear population-size reduction.

Two details are not numerically fixed by the paper: the learning-period length
LP and the p-best fraction.  The defaults below use conventional values (20
and 0.10) and expose both as parameters.  The published sinusoidal equations
write the generation index inside the sine, while also setting the fixed
frequency to 0.5; a literal integer generation would make the decreasing wave
constant.  Therefore the default uses normalized run progress in the phase
argument, matching the intended sinusoidal schedule.  A literal mode remains
available through ``sinusoid_argument_mode='generation'``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(center: float, scale: float = 0.1, attempts: int = 100) -> float:
    """Sample SHADE's positive, upper-truncated Cauchy scaling factor."""
    for _ in range(max(1, int(attempts))):
        value = float(center) + float(scale) * float(np.random.standard_cauchy())
        if value > 0.0:
            return min(value, 1.0)
    return float(np.clip(center, 1.0e-12, 1.0))


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Sample CR from the paper's normal distribution and clip to [0, 1]."""
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used in Eqs. (11)–(15)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    weights = weights / (float(np.sum(weights)) + _EPS)
    denominator = float(np.sum(weights * values))
    if abs(denominator) <= _EPS or not np.isfinite(denominator):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denominator, 0.0, 1.0))


class LSHADECnEpSinEngine(PortedPopulationEngine):
    """LSHADE-cnEpSin with native paper-specific operators and telemetry."""

    algorithm_id = "lshade_cnepsin"
    algorithm_name = "LSHADE-cnEpSin"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2017.7969336",
        "title": "Ensemble Sinusoidal Differential Covariance Matrix Adaptation with Euclidean Neighborhood for Solving CEC2017 Benchmark Problems",
        "authors": "Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan",
        "year": 2017,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        population_size=None,              # paper: NPmax = 18 * D
        initial_population_multiplier=18.0,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        memory_freq_init=0.5,
        pbest_fraction=0.10,               # paper leaves p unspecified
        archive_rate=1.0,                  # archive capacity equals current NP
        fixed_frequency=0.5,
        cauchy_scale=0.1,
        cr_sigma=0.1,
        phase_switch_fraction=0.5,
        learning_period=20,                # LP is named but not numerically set
        success_epsilon=0.01,
        strategy_selection="roulette",     # text/equations; "argmax" follows Fig. 1 literally
        covariance_probability=0.4,        # pc
        neighborhood_fraction=0.5,         # ps
        sinusoid_argument_mode="progress",
        resampling_attempts=100,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        requested = self._params.get("population_size")
        if requested in (None, 0, "paper", "auto"):
            multiplier = float(self._params.get("initial_population_multiplier", 18.0))
            self._n = max(4, int(round(multiplier * max(1, self.problem.dimension))))
        else:
            self._n = max(4, int(requested))
        self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}

    # ------------------------------------------------------------------
    # EvoMapX telemetry
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        prefix = self.algorithm_id
        return [
            f"{prefix}.sinusoidal_performance_adaptation",
            f"{prefix}.sinusoidal_decreasing_f",
            f"{prefix}.sinusoidal_increasing_f",
            f"{prefix}.adaptive_frequency_update",
            f"{prefix}.lshade_second_phase_adaptation",
            f"{prefix}.current_to_pbest_mutation",
            f"{prefix}.covariance_eigen_crossover",
            f"{prefix}.binomial_crossover",
            f"{prefix}.midpoint_bound_repair",
            f"{prefix}.greedy_selection",
            f"{prefix}.external_archive_update",
            f"{prefix}.success_history_update",
            f"{prefix}.linear_population_size_reduction",
        ]

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._operator_labels}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._operator_labels}

    def _record(
        self,
        contributions: dict[str, float],
        counts: dict[str, int],
        suffix: str,
        contribution: float = 0.0,
        count: int = 0,
    ) -> None:
        label = f"{self.algorithm_id}.{suffix}"
        contributions[label] += float(contribution)
        counts[label] += int(count)

    # ------------------------------------------------------------------
    # Initialization and schedules
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(1, int(self._params.get("hist_mem_size", 5)))
        self._params["hist_mem_size"] = h
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5))),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5))),
            "M_FREQ": np.full(h, float(self._params.get("memory_freq_init", 0.5))),
            "k": 0,
            # The paper explicitly initializes A with P0.
            "archive": np.asarray(pop[:, :-1], dtype=float).copy(),
            "initial_n": int(pop.shape[0]),
            "strategy_success_history": np.empty((0, 2), dtype=float),
            "strategy_failure_history": np.empty((0, 2), dtype=float),
            "strategy_probabilities": np.array([0.5, 0.5], dtype=float),
            "last_phase": "sinusoidal_first_half",
            "last_covariance_neighborhood_size": 0,
            "last_covariance_crossovers": 0,
            "last_bound_repairs": 0,
        }

    def _progress(self, evaluations: int, step: int) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(evaluations / float(self.config.max_evaluations), 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        return float(np.clip(step / float(horizon), 0.0, 1.0))

    def _phase_argument(self, progress: float, generation: int) -> float:
        mode = str(self._params.get("sinusoid_argument_mode", "progress")).strip().lower()
        return float(generation if mode in {"generation", "literal", "paper_literal"} else progress)

    def _decreasing_f(self, progress: float, generation: int) -> float:
        freq = float(self._params.get("fixed_frequency", 0.5))
        argument = self._phase_argument(progress, generation)
        value = 0.5 * (
            np.sin(2.0 * np.pi * freq * argument + np.pi) * (1.0 - progress) + 1.0
        )
        return float(np.clip(value, 0.0, 1.0))

    def _increasing_f(
        self,
        progress: float,
        generation: int,
        frequency_center: float,
    ) -> tuple[float, float]:
        frequency = _positive_cauchy(
            frequency_center,
            float(self._params.get("cauchy_scale", 0.1)),
            int(self._params.get("resampling_attempts", 100)),
        )
        argument = self._phase_argument(progress, generation)
        value = 0.5 * (np.sin(2.0 * np.pi * frequency * argument) * progress + 1.0)
        return float(np.clip(value, 0.0, 1.0)), float(frequency)

    def _strategy_probabilities(
        self,
        success_history: np.ndarray,
        failure_history: np.ndarray,
    ) -> np.ndarray:
        lp = max(1, int(self._params.get("learning_period", 20)))
        if success_history.shape[0] < lp:
            return np.array([0.5, 0.5], dtype=float)
        successes = np.sum(success_history[-lp:], axis=0)
        failures = np.sum(failure_history[-lp:], axis=0)
        epsilon = float(self._params.get("success_epsilon", 0.01))
        scores = successes / (successes + failures + _EPS) + epsilon
        return np.asarray(scores / (float(np.sum(scores)) + _EPS), dtype=float)

    def _choose_strategy(self, probabilities: np.ndarray) -> int:
        mode = str(self._params.get("strategy_selection", "roulette")).strip().lower()
        if mode in {"argmax", "best", "larger_probability"} and not np.allclose(probabilities, probabilities[0]):
            return int(np.argmax(probabilities))
        return int(np.random.choice(2, p=np.asarray(probabilities, dtype=float)))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, step_after: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress = self._progress(evaluations_after_step, step_after)
        return max(min_n, int(round(initial_n + (min_n - initial_n) * progress)))

    # ------------------------------------------------------------------
    # DE mutation, covariance crossover, and bound handling
    # ------------------------------------------------------------------
    def _choice_population_index(self, n: int, excluded: set[int]) -> int:
        candidates = np.array([idx for idx in range(n) if idx not in excluded], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        return int(np.random.choice(candidates))

    def _choose_pbest_index(self, order: np.ndarray, n: int, i: int) -> int:
        p = float(np.clip(self._params.get("pbest_fraction", 0.10), _EPS, 1.0))
        pnum = min(n, max(2, int(np.ceil(p * n))))
        candidates = np.asarray(order[:pnum], dtype=int)
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            candidates = np.asarray(order[:pnum], dtype=int)
        return int(np.random.choice(candidates))

    def _sample_r2(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        excluded_population_indices: set[int],
    ) -> np.ndarray:
        n = int(pop.shape[0])
        total = n + int(archive.shape[0])
        for _ in range(max(1, int(self._params.get("resampling_attempts", 100)))):
            idx = int(np.random.randint(total))
            if idx < n:
                if idx not in excluded_population_indices:
                    return np.asarray(pop[idx, :-1], dtype=float)
            else:
                return np.asarray(archive[idx - n], dtype=float)
        return np.asarray(
            pop[self._choice_population_index(n, excluded_population_indices), :-1],
            dtype=float,
        )

    def _midpoint_repair(self, vector: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, int]:
        repaired = np.asarray(vector, dtype=float).copy()
        below = repaired < self._lo
        above = repaired > self._hi
        count = int(np.count_nonzero(below) + np.count_nonzero(above))
        repaired[below] = 0.5 * (self._lo[below] + parent[below])
        repaired[above] = 0.5 * (self._hi[above] + parent[above])
        return np.clip(repaired, self._lo, self._hi), count

    def _covariance_basis(self, pop: np.ndarray) -> tuple[np.ndarray, int, bool]:
        positions = np.asarray(pop[:, :-1], dtype=float)
        n, dim = positions.shape
        best_idx = self._best_index(pop[:, -1])
        distances = np.linalg.norm(positions - positions[best_idx], axis=1)
        fraction = float(np.clip(self._params.get("neighborhood_fraction", 0.5), _EPS, 1.0))
        neighborhood_size = min(n, max(2, int(round(fraction * n))))
        neighborhood = positions[np.argsort(distances)[:neighborhood_size]]
        centered = neighborhood - np.mean(neighborhood, axis=0, keepdims=True)
        if neighborhood_size < 2 or not np.any(np.abs(centered) > 0.0):
            return np.eye(dim), neighborhood_size, False
        covariance = (centered.T @ centered) / float(max(1, neighborhood_size - 1))
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, basis = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return np.eye(dim), neighborhood_size, False
        valid = bool(np.all(np.isfinite(basis)) and np.any(eigenvalues > 1.0e-20))
        return (basis if valid else np.eye(dim)), neighborhood_size, valid

    @staticmethod
    def _binomial_crossover(parent: np.ndarray, donor: np.ndarray, cr: float) -> np.ndarray:
        dim = int(parent.size)
        mask = np.random.rand(dim) <= float(cr)
        mask[int(np.random.randint(dim))] = True
        return np.where(mask, donor, parent)

    def _make_trial(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        f_value: float,
        cr_value: float,
        basis: np.ndarray,
        covariance_valid: bool,
    ) -> tuple[np.ndarray, str, int]:
        parent = np.asarray(pop[i, :-1], dtype=float)
        pbest_idx = self._choose_pbest_index(order, pop.shape[0], i)
        r1_idx = self._choice_population_index(pop.shape[0], {i, pbest_idx})
        r2 = self._sample_r2(pop, archive, {i, pbest_idx, r1_idx})
        donor = parent + f_value * (pop[pbest_idx, :-1] - parent) + f_value * (pop[r1_idx, :-1] - r2)
        donor, repair_count = self._midpoint_repair(donor, parent)

        use_covariance = covariance_valid and (
            np.random.rand() < float(self._params.get("covariance_probability", 0.4))
        )
        if use_covariance:
            parent_rotated = basis.T @ parent
            donor_rotated = basis.T @ donor
            trial = basis @ self._binomial_crossover(parent_rotated, donor_rotated, cr_value)
            crossover = "covariance_eigen_crossover"
        else:
            trial = self._binomial_crossover(parent, donor, cr_value)
            crossover = "binomial_crossover"
        trial, final_repairs = self._midpoint_repair(trial, parent)
        return trial, crossover, repair_count + final_repairs

    def _survival_mask(self, trial_fit: np.ndarray, parent_fit: np.ndarray) -> np.ndarray:
        return trial_fit <= parent_fit if self.problem.objective == "min" else trial_fit >= parent_fit

    def _strict_mask(self, trial_fit: np.ndarray, parent_fit: np.ndarray) -> np.ndarray:
        return trial_fit < parent_fit if self.problem.objective == "min" else trial_fit > parent_fit

    # ------------------------------------------------------------------
    # Main native macro-step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), int(self.problem.dimension)
        if n < 4:
            raise ValueError("LSHADE-cnEpSin requires at least four population members.")

        M_F = np.asarray(state.payload["M_F"], dtype=float).copy()
        M_CR = np.asarray(state.payload["M_CR"], dtype=float).copy()
        M_FREQ = np.asarray(state.payload["M_FREQ"], dtype=float).copy()
        memory_index = int(state.payload.get("k", 0)) % int(M_F.size)
        archive = np.asarray(state.payload.get("archive"), dtype=float).reshape(-1, dim)
        initial_n = int(state.payload.get("initial_n", n))
        success_history = np.asarray(
            state.payload.get("strategy_success_history", np.empty((0, 2))), dtype=float
        ).reshape(-1, 2)
        failure_history = np.asarray(
            state.payload.get("strategy_failure_history", np.empty((0, 2))), dtype=float
        ).reshape(-1, 2)

        progress = self._progress(int(state.evaluations), int(state.step))
        second_phase = progress > float(self._params.get("phase_switch_fraction", 0.5))
        probabilities = self._strategy_probabilities(success_history, failure_history)
        basis, neighborhood_size, covariance_valid = self._covariance_basis(pop)
        order = self._order(pop[:, -1])
        parent_positions = pop[:, :-1].copy()
        parent_fitness = pop[:, -1].copy()

        contributions = self._blank_contributions()
        counts = self._blank_counts()
        trials = np.empty((n, dim), dtype=float)
        f_values = np.empty(n, dtype=float)
        cr_values = np.empty(n, dtype=float)
        frequency_values = np.full(n, np.nan, dtype=float)
        strategies = np.full(n, -1, dtype=int)
        crossover_labels: list[str] = []
        repairs = 0

        for i in range(n):
            memory_sample = int(np.random.randint(M_F.size))
            cr_value = _normal_cr(M_CR[memory_sample], float(self._params.get("cr_sigma", 0.1)))
            if second_phase:
                f_value = _positive_cauchy(
                    M_F[memory_sample],
                    float(self._params.get("cauchy_scale", 0.1)),
                    int(self._params.get("resampling_attempts", 100)),
                )
                self._record(contributions, counts, "lshade_second_phase_adaptation", count=1)
            else:
                strategy = self._choose_strategy(probabilities)
                strategies[i] = strategy
                self._record(contributions, counts, "sinusoidal_performance_adaptation", count=1)
                if strategy == 0:
                    f_value = self._decreasing_f(progress, int(state.step) + 1)
                    self._record(contributions, counts, "sinusoidal_decreasing_f", count=1)
                else:
                    f_value, frequency = self._increasing_f(
                        progress,
                        int(state.step) + 1,
                        M_FREQ[memory_sample],
                    )
                    frequency_values[i] = frequency
                    self._record(contributions, counts, "sinusoidal_increasing_f", count=1)

            trial, crossover, repaired = self._make_trial(
                pop, archive, order, i, f_value, cr_value, basis, covariance_valid
            )
            trials[i] = trial
            f_values[i] = f_value
            cr_values[i] = cr_value
            crossover_labels.append(crossover)
            repairs += repaired
            self._record(contributions, counts, "current_to_pbest_mutation", count=1)
            self._record(contributions, counts, crossover, count=1)

        trial_fitness = self._evaluate_population(trials)
        survival = self._survival_mask(trial_fitness, parent_fitness)
        strict = self._strict_mask(trial_fitness, parent_fitness)
        if self.problem.objective == "min":
            gains = np.maximum(parent_fitness - trial_fitness, 0.0)
        else:
            gains = np.maximum(trial_fitness - parent_fitness, 0.0)

        for idx in np.flatnonzero(strict):
            gain_share = float(gains[idx]) / 3.0
            contributions[f"{self.algorithm_id}.current_to_pbest_mutation"] += gain_share
            contributions[f"{self.algorithm_id}.{crossover_labels[idx]}"] += gain_share
            contributions[f"{self.algorithm_id}.greedy_selection"] += gain_share
        self._record(contributions, counts, "greedy_selection", count=int(np.count_nonzero(survival)))
        self._record(contributions, counts, "midpoint_bound_repair", count=repairs)

        if not second_phase:
            generation_successes = np.zeros(2, dtype=float)
            generation_failures = np.zeros(2, dtype=float)
            for strategy in (0, 1):
                selected = strategies == strategy
                generation_successes[strategy] = np.count_nonzero(selected & survival)
                generation_failures[strategy] = np.count_nonzero(selected & ~survival)
            success_history = np.vstack((success_history, generation_successes))
            failure_history = np.vstack((failure_history, generation_failures))
            lp = max(1, int(self._params.get("learning_period", 20)))
            success_history = success_history[-lp:]
            failure_history = failure_history[-lp:]

        if np.any(strict):
            replaced_parents = parent_positions[strict]
            archive = np.vstack((archive, replaced_parents)) if archive.size else replaced_parents.copy()
            self._record(
                contributions,
                counts,
                "external_archive_update",
                count=int(np.count_nonzero(strict)),
            )

            successful_indices = np.flatnonzero(strict)
            improvements = gains[successful_indices]
            weights = improvements / (float(np.sum(improvements)) + _EPS)
            M_CR[memory_index] = _weighted_lehmer(cr_values[successful_indices], weights)
            if second_phase:
                M_F[memory_index] = _weighted_lehmer(f_values[successful_indices], weights)
            else:
                adaptive_success = successful_indices[
                    (strategies[successful_indices] == 1)
                    & np.isfinite(frequency_values[successful_indices])
                ]
                if adaptive_success.size:
                    frequency_weights = gains[adaptive_success]
                    M_FREQ[memory_index] = _weighted_lehmer(
                        frequency_values[adaptive_success], frequency_weights
                    )
                    self._record(contributions, counts, "adaptive_frequency_update", count=1)
            memory_index = (memory_index + 1) % int(M_F.size)
            self._record(contributions, counts, "success_history_update", count=1)

        if np.any(survival):
            pop[survival, :-1] = trials[survival]
            pop[survival, -1] = trial_fitness[survival]

        evaluations_after = int(state.evaluations + n)
        target_n = self._target_population_size(evaluations_after, initial_n, int(state.step) + 1)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._record(contributions, counts, "linear_population_size_reduction", count=removed)

        # Paper capacity: |A| <= current NP, with random deletion.
        archive_capacity = max(1, int(round(float(self._params.get("archive_rate", 1.0)) * pop.shape[0])))
        if archive.shape[0] > archive_capacity:
            archive = archive[np.random.choice(archive.shape[0], archive_capacity, replace=False)]

        next_probabilities = self._strategy_probabilities(success_history, failure_history)
        self._last_operator_contributions = {
            label: float(value) for label, value in contributions.items()
        }
        self._last_operator_counts = {label: int(value) for label, value in counts.items()}
        return pop, n, {
            "M_F": M_F,
            "M_CR": M_CR,
            "M_FREQ": M_FREQ,
            "k": int(memory_index),
            "archive": archive,
            "initial_n": initial_n,
            "strategy_success_history": success_history,
            "strategy_failure_history": failure_history,
            "strategy_probabilities": next_probabilities,
            "last_phase": "lshade_second_half" if second_phase else "sinusoidal_first_half",
            "last_covariance_neighborhood_size": int(neighborhood_size),
            "last_covariance_crossovers": int(
                counts[f"{self.algorithm_id}.covariance_eigen_crossover"]
            ),
            "last_bound_repairs": int(repairs),
        }

    def observe(self, state):
        observation = super().observe(state)
        observation["operator_contributions"] = dict(self._last_operator_contributions)
        observation["operator_counts"] = dict(self._last_operator_counts)
        observation["evomapx_delta_f"] = "direct_improvement"
        observation["evomapx_fidelity"] = "native"
        observation["mean_memory_f"] = float(np.mean(np.asarray(state.payload["M_F"])))
        observation["mean_memory_cr"] = float(np.mean(np.asarray(state.payload["M_CR"])))
        observation["mean_memory_frequency"] = float(np.mean(np.asarray(state.payload["M_FREQ"])))
        observation["archive_size"] = int(np.asarray(state.payload["archive"]).shape[0])
        probabilities = np.asarray(state.payload.get("strategy_probabilities", [0.5, 0.5]))
        observation["sinusoidal_strategy_probabilities"] = probabilities.tolist()
        observation["last_phase"] = str(state.payload.get("last_phase", "unknown"))
        observation["last_covariance_neighborhood_size"] = int(
            state.payload.get("last_covariance_neighborhood_size", 0)
        )
        observation["last_covariance_crossovers"] = int(
            state.payload.get("last_covariance_crossovers", 0)
        )
        observation["last_bound_repairs"] = int(state.payload.get("last_bound_repairs", 0))
        return observation
