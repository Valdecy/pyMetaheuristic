"""Native LSHADE-EpSin engine for pyMetaheuristic.

The implementation follows Awad et al. (CEC 2016): L-SHADE's
current-to-pbest/1 mutation with an external archive, a 50/50 ensemble of two
sinusoidal scaling-factor schedules in the first half of the search,
success-history F/CR adaptation in the second half, linear population-size
reduction, and the one-shot Gaussian-Walk local search activated when the
population first reaches 20 members.

The paper contains two internal inconsistencies.  Its prose places the
sinusoidal ensemble in the first half and L-SHADE adaptation in the second,
whereas parts of Fig. 3 reverse those labels; this engine follows the prose,
abstract, and equation discussion.  Also, the published fixed frequency 0.5
combined with an integer generation argument makes Eq. (3) numerically
constant.  The paper-literal generation argument is therefore the default,
while ``sinusoid_argument_mode='progress'`` is exposed for experiments with
the intended smooth schedule suggested by Figs. 1-2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


def _positive_cauchy(
    center: float,
    scale: float = 0.1,
    *,
    upper: float = 1.0,
    attempts: int = 100,
) -> float:
    """Sample a positive Cauchy variate and cap it at ``upper``."""
    center = float(center)
    scale = float(scale)
    for _ in range(max(1, int(attempts))):
        value = float(np.random.standard_cauchy() * scale + center)
        if value > 0.0:
            return min(value, float(upper))
    return min(max(center, 0.5, 1.0e-12), float(upper))


def _normal_cr(center: float, sigma: float = 0.1) -> float:
    """Sample CR from the paper's normal law and clip it to [0, 1]."""
    if not np.isfinite(center):
        center = 0.5
    return float(np.clip(np.random.normal(float(center), float(sigma)), 0.0, 1.0))


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Lehmer mean used by the success-history memories."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    if weights.size != values.size:
        raise ValueError("Lehmer values and weights must have equal length.")
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= _EPS:
        weights = np.full(values.size, 1.0 / float(values.size), dtype=float)
    else:
        weights = weights / weight_sum
    denominator = float(np.sum(weights * values))
    if not np.isfinite(denominator) or abs(denominator) <= _EPS:
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    numerator = float(np.sum(weights * values * values))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


class LSHADEEpSinEngine(PortedPopulationEngine):
    """LSHADE-EpSin with native sinusoidal adaptation and Gaussian Walks."""

    algorithm_id = "lshade_epsin"
    algorithm_name = "LSHADE-EpSin"
    family = "evolutionary"
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
    _REFERENCE = {
        "doi": "10.1109/CEC.2016.7744163",
        "title": "An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems",
        "authors": "Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Robert G. Reynolds",
        "year": 2016,
    }
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        population_size=None,
        initial_population_multiplier=18.0,
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
        resampling_attempts=100,
        phase_switch_fraction=0.5,
        generation_horizon=None,
        sinusoid_argument_mode="generation",
        local_search_trigger_n=20,
        local_search_group_size=10,
        local_search_generations=250,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        requested = self._params.get("population_size")
        if requested in (None, 0, "paper", "auto"):
            multiplier = float(self._params.get("initial_population_multiplier", 18.0))
            self._n = max(4, int(round(multiplier * max(1, int(self.problem.dimension)))))
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
            f"{prefix}.mutation",
            f"{prefix}.crossover",
            f"{prefix}.selection",
            f"{prefix}.archive_update",
            f"{prefix}.success_history_update",
            f"{prefix}.population_reduction",
            f"{prefix}.sinusoidal_decreasing_f",
            f"{prefix}.sinusoidal_increasing_f",
            f"{prefix}.adaptive_frequency_update",
            f"{prefix}.lshade_second_phase_adaptation",
            f"{prefix}.gaussian_walk_local_search",
            f"{prefix}.bound_repair",
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
    def _infer_generation_horizon(self, initial_n: int) -> int:
        configured = self._params.get("generation_horizon")
        if configured not in (None, 0, "auto", "paper"):
            return max(1, int(configured))
        if self.config.max_steps is not None and int(self.config.max_steps) > 0:
            return max(1, int(self.config.max_steps))
        if self.config.max_evaluations is None or int(self.config.max_evaluations) <= 0:
            return 100

        # Infer the number of complete DE generations possible under Eq. (13).
        # The one-shot local search is excluded because Gmax is the generation
        # horizon of the evolutionary loop, while its objective calls still
        # count toward the actual FE budget.
        max_evaluations = max(int(initial_n), int(self.config.max_evaluations))
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        evaluations = int(initial_n)
        population_size = int(initial_n)
        generations = 0
        safety_limit = max(1, max_evaluations)
        while evaluations + population_size <= max_evaluations and generations < safety_limit:
            evaluations += population_size
            generations += 1
            target = int(
                round(
                    float(initial_n)
                    + float(min_n - initial_n) * (float(evaluations) / float(max_evaluations))
                )
            )
            population_size = max(min_n, min(population_size, target))
        return max(1, generations)

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(1, int(self._params.get("hist_mem_size", 5)))
        self._params["hist_mem_size"] = h
        initial_n = int(pop.shape[0])
        return {
            "M_F": np.full(h, float(self._params.get("memory_f_init", 0.5)), dtype=float),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 0.5)), dtype=float),
            "M_FREQ": np.full(h, float(self._params.get("memory_freq_init", 0.5)), dtype=float),
            "k": 0,
            # Section II.B explicitly initializes A with P0.
            "archive": np.asarray(pop[:, :-1], dtype=float).copy(),
            "initial_n": initial_n,
            "generation_horizon": self._infer_generation_horizon(initial_n),
            "local_search_done": False,
            "last_phase": "sinusoidal_first_half",
            "last_generation_ratio": 0.0,
            "last_local_search_evals": 0,
            "last_local_search_generations": 0,
            "last_local_search_replacements": 0,
            "last_local_search_truncated": False,
            "last_bound_repairs": 0,
            "last_frequency_mean": float(self._params.get("memory_freq_init", 0.5)),
            "last_accepted_trials": 0,
            "last_strict_improvements": 0,
        }

    @staticmethod
    def _generation_ratio(generation: int, generation_horizon: int) -> float:
        return float(np.clip(float(generation) / float(max(1, generation_horizon)), 0.0, 1.0))

    def _is_second_phase(self, generation_ratio: float) -> bool:
        return bool(generation_ratio > float(self._params.get("phase_switch_fraction", 0.5)))

    def _sinusoid_argument(self, generation: int, generation_ratio: float) -> float:
        mode = str(self._params.get("sinusoid_argument_mode", "generation")).strip().lower()
        if mode in {"progress", "normalized", "normalized_progress", "smooth"}:
            return float(generation_ratio)
        return float(generation)

    def _target_population_size(
        self,
        evaluations_after_generation: int,
        initial_n: int,
        current_n: int,
        generation_ratio: float,
    ) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        if self.config.max_evaluations is not None and int(self.config.max_evaluations) > 0:
            progress = float(
                np.clip(
                    float(evaluations_after_generation) / float(self.config.max_evaluations),
                    0.0,
                    1.0,
                )
            )
        else:
            progress = float(np.clip(generation_ratio, 0.0, 1.0))
        target = int(round(float(initial_n) + float(min_n - initial_n) * progress))
        return max(min_n, min(int(current_n), target))

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------
    def _decreasing_f(self, generation: int, generation_ratio: float) -> float:
        frequency = float(self._params.get("fixed_frequency", 0.5))
        argument = self._sinusoid_argument(generation, generation_ratio)
        value = 0.5 * (
            np.sin(2.0 * np.pi * frequency * argument + np.pi)
            * (1.0 - generation_ratio)
            + 1.0
        )
        return float(np.clip(value, 0.0, 1.0))

    def _increasing_f(
        self,
        generation: int,
        generation_ratio: float,
        frequency_center: float,
    ) -> tuple[float, float]:
        frequency = _positive_cauchy(
            frequency_center,
            float(self._params.get("cauchy_scale", 0.1)),
            upper=1.0,
            attempts=int(self._params.get("resampling_attempts", 100)),
        )
        argument = self._sinusoid_argument(generation, generation_ratio)
        value = 0.5 * (
            np.sin(2.0 * np.pi * frequency * argument) * generation_ratio + 1.0
        )
        return float(np.clip(value, 0.0, 1.0)), float(frequency)

    def _sample_parameters(
        self,
        M_F: np.ndarray,
        M_CR: np.ndarray,
        M_FREQ: np.ndarray,
        generation: int,
        generation_ratio: float,
        second_phase: bool,
    ) -> tuple[float, float, str, float | None]:
        memory_index = int(np.random.randint(M_CR.size))
        cr_value = _normal_cr(
            float(M_CR[memory_index]), float(self._params.get("cr_sigma", 0.1))
        )
        if second_phase:
            f_value = _positive_cauchy(
                float(M_F[memory_index]),
                float(self._params.get("cauchy_scale", 0.1)),
                upper=1.0,
                attempts=int(self._params.get("resampling_attempts", 100)),
            )
            return f_value, cr_value, "lshade_second_phase_adaptation", None
        if np.random.rand() < 0.5:
            return (
                self._decreasing_f(generation, generation_ratio),
                cr_value,
                "sinusoidal_decreasing_f",
                None,
            )
        f_value, frequency = self._increasing_f(
            generation,
            generation_ratio,
            float(M_FREQ[memory_index]),
        )
        return f_value, cr_value, "sinusoidal_increasing_f", frequency

    # ------------------------------------------------------------------
    # DE/current-to-pbest/1 and bounds
    # ------------------------------------------------------------------
    @staticmethod
    def _choice_excluding(n: int, excluded: set[int]) -> int:
        candidates = np.asarray([idx for idx in range(n) if idx not in excluded], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        return int(np.random.choice(candidates))

    def _choose_pbest_index(self, order: np.ndarray, n: int) -> int:
        max_fraction = float(np.clip(self._params.get("pbest_fraction", 0.10), _EPS, 1.0))
        max_pool = min(n, max(2, int(round(max_fraction * float(n)))))
        pool_size = 2 if max_pool <= 2 else int(np.random.randint(2, max_pool + 1))
        return int(np.random.choice(np.asarray(order[:pool_size], dtype=int)))

    def _select_r2(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        i: int,
        r1_index: int,
    ) -> np.ndarray:
        n = int(pop.shape[0])
        archive = np.asarray(archive, dtype=float).reshape(-1, int(self.problem.dimension))
        total = n + int(archive.shape[0])
        for _ in range(100):
            index = int(np.random.randint(total))
            if index < n:
                if index in {int(i), int(r1_index)}:
                    continue
                return np.asarray(pop[index, :-1], dtype=float)
            return np.asarray(archive[index - n], dtype=float)
        valid_population = [j for j in range(n) if j not in {int(i), int(r1_index)}]
        if valid_population:
            return np.asarray(pop[int(np.random.choice(valid_population)), :-1], dtype=float)
        if archive.size:
            return np.asarray(archive[int(np.random.randint(archive.shape[0]))], dtype=float)
        return np.asarray(pop[r1_index, :-1], dtype=float)

    def _midpoint_bound_repair(
        self,
        donor: np.ndarray,
        parent: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        # The article does not specify bound handling; midpoint repair is the
        # native L-SHADE convention and avoids the distributional distortion of
        # direct clipping.
        donor = np.asarray(donor, dtype=float).copy()
        parent = np.asarray(parent, dtype=float)
        below = donor < self._lo
        above = donor > self._hi
        repaired = int(np.count_nonzero(below) + np.count_nonzero(above))
        donor[below] = (self._lo[below] + parent[below]) / 2.0
        donor[above] = (self._hi[above] + parent[above]) / 2.0
        return np.clip(donor, self._lo, self._hi), repaired

    def _make_trial(
        self,
        pop: np.ndarray,
        archive: np.ndarray,
        order: np.ndarray,
        i: int,
        f_value: float,
        cr_value: float,
    ) -> tuple[np.ndarray, int]:
        n = int(pop.shape[0])
        dim = int(self.problem.dimension)
        parent = np.asarray(pop[i, :-1], dtype=float)
        pbest_index = self._choose_pbest_index(order, n)
        r1_index = self._choice_excluding(n, {int(i)})
        r2 = self._select_r2(pop, archive, i, r1_index)
        donor = (
            parent
            + float(f_value) * (pop[pbest_index, :-1] - parent)
            + float(f_value) * (pop[r1_index, :-1] - r2)
        )
        donor, repaired = self._midpoint_bound_repair(donor, parent)
        crossover = np.random.rand(dim) < float(cr_value)
        crossover[int(np.random.randint(dim))] = True
        return np.where(crossover, donor, parent), repaired

    # ------------------------------------------------------------------
    # Gaussian-Walk local search
    # ------------------------------------------------------------------
    def _remaining_budget(self, evaluations_used: int) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(0, int(self.config.max_evaluations) - int(evaluations_used))

    def _gaussian_walk_local_search(
        self,
        pop: np.ndarray,
        evaluations_after_generation: int,
        contributions: dict[str, float],
        counts: dict[str, int],
    ) -> tuple[np.ndarray, int, int, int, bool, int]:
        group_size = max(1, int(self._params.get("local_search_group_size", 10)))
        requested_generations = max(1, int(self._params.get("local_search_generations", 250)))
        remaining = self._remaining_budget(evaluations_after_generation)
        if remaining is not None and remaining < group_size:
            return pop, 0, 0, 0, True, 0

        actual_generations = requested_generations
        if remaining is not None:
            actual_generations = min(
                requested_generations,
                max(0, (remaining - group_size) // group_size),
            )
        truncated = actual_generations < requested_generations

        walkers = self._new_positions(group_size)
        walker_fitness = self._evaluate_population(walkers)
        evaluations = group_size
        bound_repairs = 0
        dim = int(self.problem.dimension)

        for local_generation in range(1, actual_generations + 1):
            best_index = self._best_index(walker_fitness)
            x_best = np.asarray(walkers[best_index], dtype=float)
            sigma = np.abs(
                (np.log(float(local_generation)) / float(local_generation))
                * (walkers - x_best)
            )
            gaussian = np.random.normal(loc=x_best, scale=sigma, size=(group_size, dim))
            epsilon = np.random.rand(group_size, dim)
            epsilon_hat = np.random.rand(group_size, dim)
            candidates = gaussian + epsilon * x_best - epsilon_hat * walkers
            out_of_bounds = (candidates < self._lo) | (candidates > self._hi)
            bound_repairs += int(np.count_nonzero(out_of_bounds))
            candidates = np.clip(candidates, self._lo, self._hi)
            candidate_fitness = self._evaluate_population(candidates)
            evaluations += group_size

            # The paper does not state the intra-walk acceptance rule. Greedy
            # replacement is the conservative local-search interpretation.
            improved = self._better_mask(candidate_fitness, walker_fitness)
            if np.any(improved):
                walkers[improved] = candidates[improved]
                walker_fitness[improved] = candidate_fitness[improved]

        pop = pop.copy()
        replacement_count = 0
        replacement_gain = 0.0
        replace_k = min(group_size, int(pop.shape[0]))
        local_order = self._order(walker_fitness)[:replace_k]
        main_worst = self._order(pop[:, -1])[::-1][:replace_k]
        for local_index, population_index in zip(local_order, main_worst):
            local_fit = float(walker_fitness[local_index])
            main_fit = float(pop[population_index, -1])
            if self._is_better(local_fit, main_fit):
                replacement_gain += abs(main_fit - local_fit)
                replacement_count += 1
                pop[population_index, :-1] = walkers[local_index]
                pop[population_index, -1] = local_fit

        self._record(
            contributions,
            counts,
            "gaussian_walk_local_search",
            contribution=replacement_gain,
            count=evaluations,
        )
        self._record(contributions, counts, "bound_repair", count=bound_repairs)
        return (
            pop,
            int(evaluations),
            int(actual_generations),
            int(replacement_count),
            bool(truncated),
            int(bound_repairs),
        )

    # ------------------------------------------------------------------
    # Main native macro-step
    # ------------------------------------------------------------------
    def _step_impl(self, state, pop: np.ndarray):
        n = int(pop.shape[0])
        dim = int(self.problem.dimension)
        if n < 4:
            raise ValueError("LSHADE-EpSin requires at least four population members.")

        M_F = np.asarray(state.payload["M_F"], dtype=float).copy()
        M_CR = np.asarray(state.payload["M_CR"], dtype=float).copy()
        M_FREQ = np.asarray(state.payload["M_FREQ"], dtype=float).copy()
        memory_index = int(state.payload.get("k", 0)) % int(M_F.size)
        archive = np.asarray(state.payload.get("archive"), dtype=float).reshape(-1, dim)
        initial_n = int(state.payload.get("initial_n", n))
        generation_horizon = max(1, int(state.payload.get("generation_horizon", 100)))
        generation = int(state.step) + 1
        generation_ratio = self._generation_ratio(generation, generation_horizon)
        second_phase = self._is_second_phase(generation_ratio)

        contributions = self._blank_contributions()
        counts = self._blank_counts()
        order = self._order(pop[:, -1])
        parent_positions = pop[:, :-1].copy()
        parent_fitness = pop[:, -1].copy()

        trials = np.empty((n, dim), dtype=float)
        f_values = np.empty(n, dtype=float)
        cr_values = np.empty(n, dtype=float)
        frequency_values = np.full(n, np.nan, dtype=float)
        adaptation_labels: list[str] = []
        bound_repairs = 0

        for i in range(n):
            f_value, cr_value, adaptation_label, frequency = self._sample_parameters(
                M_F,
                M_CR,
                M_FREQ,
                generation,
                generation_ratio,
                second_phase,
            )
            trial, repaired = self._make_trial(
                pop, archive, order, i, f_value, cr_value
            )
            trials[i] = trial
            f_values[i] = f_value
            cr_values[i] = cr_value
            frequency_values[i] = np.nan if frequency is None else float(frequency)
            adaptation_labels.append(adaptation_label)
            bound_repairs += int(repaired)
            self._record(contributions, counts, adaptation_label, count=1)
            self._record(contributions, counts, "mutation", count=1)
            self._record(contributions, counts, "crossover", count=1)
            self._record(contributions, counts, "selection", count=1)

        trial_fitness = self._evaluate_population(trials)
        evaluations = n
        if self.problem.objective == "min":
            survival = trial_fitness <= parent_fitness
            strict = trial_fitness < parent_fitness
            gains = np.maximum(parent_fitness - trial_fitness, 0.0)
        else:
            survival = trial_fitness >= parent_fitness
            strict = trial_fitness > parent_fitness
            gains = np.maximum(trial_fitness - parent_fitness, 0.0)

        for index in np.flatnonzero(strict):
            gain_share = float(gains[index]) / 4.0
            contributions[f"{self.algorithm_id}.{adaptation_labels[index]}"] += gain_share
            contributions[f"{self.algorithm_id}.mutation"] += gain_share
            contributions[f"{self.algorithm_id}.crossover"] += gain_share
            contributions[f"{self.algorithm_id}.selection"] += gain_share
        self._record(contributions, counts, "bound_repair", count=bound_repairs)

        if np.any(strict):
            replaced_parents = parent_positions[strict]
            archive = (
                np.vstack((archive, replaced_parents))
                if archive.size
                else replaced_parents.copy()
            )
            self._record(
                contributions,
                counts,
                "archive_update",
                count=int(np.count_nonzero(strict)),
            )

            successful_indices = np.flatnonzero(strict)
            successful_gains = gains[successful_indices]
            M_CR[memory_index] = _weighted_lehmer(
                cr_values[successful_indices], successful_gains
            )
            if second_phase:
                M_F[memory_index] = _weighted_lehmer(
                    f_values[successful_indices], successful_gains
                )
            else:
                adaptive_indices = successful_indices[
                    np.isfinite(frequency_values[successful_indices])
                ]
                if adaptive_indices.size:
                    M_FREQ[memory_index] = _weighted_lehmer(
                        frequency_values[adaptive_indices], gains[adaptive_indices]
                    )
                    self._record(
                        contributions,
                        counts,
                        "adaptive_frequency_update",
                        count=1,
                    )
            memory_index = (memory_index + 1) % int(M_F.size)
            self._record(contributions, counts, "success_history_update", count=1)

        if np.any(survival):
            pop[survival, :-1] = trials[survival]
            pop[survival, -1] = trial_fitness[survival]

        evaluations_after_generation = int(state.evaluations + evaluations)
        target_n = self._target_population_size(
            evaluations_after_generation,
            initial_n,
            int(pop.shape[0]),
            generation_ratio,
        )
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = int(pop.shape[0] - target_n)
            pop = pop[keep]
            self._record(contributions, counts, "population_reduction", count=removed)

        # Section II.B: archive capacity equals current NP and excess members
        # are removed randomly.
        archive_capacity = max(
            1,
            int(round(float(self._params.get("archive_rate", 1.0)) * pop.shape[0])),
        )
        if archive.shape[0] > archive_capacity:
            archive = archive[
                np.random.choice(archive.shape[0], archive_capacity, replace=False)
            ]

        local_search_done = bool(state.payload.get("local_search_done", False))
        local_evaluations = 0
        local_generations = 0
        local_replacements = 0
        local_truncated = False
        local_bound_repairs = 0
        trigger_n = int(self._params.get("local_search_trigger_n", 20))
        if not local_search_done and int(pop.shape[0]) <= trigger_n:
            (
                pop,
                local_evaluations,
                local_generations,
                local_replacements,
                local_truncated,
                local_bound_repairs,
            ) = self._gaussian_walk_local_search(
                pop,
                evaluations_after_generation,
                contributions,
                counts,
            )
            evaluations += int(local_evaluations)
            local_search_done = True

        self._last_operator_contributions = {
            label: float(value) for label, value in contributions.items()
        }
        self._last_operator_counts = {
            label: int(value) for label, value in counts.items()
        }
        return pop, int(evaluations), {
            "M_F": M_F,
            "M_CR": M_CR,
            "M_FREQ": M_FREQ,
            "k": int(memory_index),
            "archive": archive,
            "initial_n": initial_n,
            "generation_horizon": generation_horizon,
            "local_search_done": bool(local_search_done),
            "last_phase": "lshade_second_half" if second_phase else "sinusoidal_first_half",
            "last_generation_ratio": float(generation_ratio),
            "last_local_search_evals": int(local_evaluations),
            "last_local_search_generations": int(local_generations),
            "last_local_search_replacements": int(local_replacements),
            "last_local_search_truncated": bool(local_truncated),
            "last_bound_repairs": int(bound_repairs + local_bound_repairs),
            "last_frequency_mean": float(np.nanmean(M_FREQ)),
            "last_accepted_trials": int(np.count_nonzero(survival)),
            "last_strict_improvements": int(np.count_nonzero(strict)),
        }

    def observe(self, state):
        observation = super().observe(state)
        observation["operator_contributions"] = dict(self._last_operator_contributions)
        observation["operator_counts"] = dict(self._last_operator_counts)
        observation["evomapx_delta_f"] = "direct_improvement"
        observation["evomapx_fidelity"] = "native"
        if "M_F" in state.payload:
            observation["mean_memory_f"] = float(
                np.nanmean(np.asarray(state.payload["M_F"], dtype=float))
            )
        if "M_CR" in state.payload:
            observation["mean_memory_cr"] = float(
                np.nanmean(np.asarray(state.payload["M_CR"], dtype=float))
            )
        if "M_FREQ" in state.payload:
            observation["mean_memory_frequency"] = float(
                np.nanmean(np.asarray(state.payload["M_FREQ"], dtype=float))
            )
        if "archive" in state.payload:
            observation["archive_size"] = int(
                np.asarray(state.payload["archive"]).shape[0]
            )
        observation["generation_horizon"] = int(
            state.payload.get("generation_horizon", 0)
        )
        observation["last_phase"] = str(state.payload.get("last_phase", "unknown"))
        observation["last_generation_ratio"] = float(
            state.payload.get("last_generation_ratio", 0.0)
        )
        observation["local_search_done"] = bool(
            state.payload.get("local_search_done", False)
        )
        observation["last_local_search_evals"] = int(
            state.payload.get("last_local_search_evals", 0)
        )
        observation["last_local_search_generations"] = int(
            state.payload.get("last_local_search_generations", 0)
        )
        observation["last_local_search_replacements"] = int(
            state.payload.get("last_local_search_replacements", 0)
        )
        observation["last_local_search_truncated"] = bool(
            state.payload.get("last_local_search_truncated", False)
        )
        observation["last_bound_repairs"] = int(
            state.payload.get("last_bound_repairs", 0)
        )
        observation["last_accepted_trials"] = int(
            state.payload.get("last_accepted_trials", 0)
        )
        observation["last_strict_improvements"] = int(
            state.payload.get("last_strict_improvements", 0)
        )
        return observation
