"""pyMetaheuristic src - Adaptive Gaining-Sharing Knowledge Engine.

Native NumPy implementation of the Adaptive Gaining-Sharing Knowledge based
algorithm (AGSK) proposed by Mohamed et al. for the CEC 2020 bound-constrained
benchmark.  The implementation follows the paper's junior/senior knowledge
phases, adaptive ``(K_F, K_R)`` setting pool, heterogeneous knowledge-rate
schedule, midpoint bound repair, greedy replacement, and linear population-size
reduction.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ._ported_common import PortedPopulationEngine
from .protocol import CapabilityProfile


class AGSKEngine(PortedPopulationEngine):
    """Adaptive Gaining-Sharing Knowledge based algorithm (AGSK)."""

    algorithm_id = "agsk"
    algorithm_name = "Adaptive Gaining-Sharing Knowledge Based Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1109/CEC48606.2020.9185901",
        "title": (
            "Evaluating the Performance of Adaptive Gaining-Sharing Knowledge "
            "Based Algorithm on CEC 2020 Benchmark Problems"
        ),
        "authors": "Ali Wagdy Mohamed, Anas A. Hadi, Ali Khater Mohamed, and Noor H. Awad",
        "year": 2020,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        population_size=None,              # paper default: 20 * dimension
        population_multiplier=20.0,
        min_population_size=12,
        senior_fraction=0.05,
        learning_rate=0.05,
        adaptation_start=0.10,
        minimum_setting_probability=0.05,
        kf_pool=(0.1, 1.0, 0.5, 1.0),
        kr_pool=(0.2, 0.1, 0.9, 0.9),
        initial_setting_probabilities=(0.85, 0.05, 0.05, 0.05),
        low_knowledge_rate_probability=0.50,
        integer_knowledge_rate_max=20,
    )

    _OPERATOR_LABELS = (
        "agsk.parameter_setting_sampling",
        "agsk.junior_gaining_sharing",
        "agsk.senior_gaining_sharing",
        "agsk.midpoint_bound_repair",
        "agsk.greedy_selection",
        "agsk.parameter_adaptation",
        "agsk.linear_population_size_reduction",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size")
        minimum = max(4, int(self._params.get("min_population_size", 12)))
        if pop_param in (None, 0, "paper", "auto"):
            multiplier = float(self._params.get("population_multiplier", 20.0))
            self._n = max(minimum, int(np.floor(multiplier * self.problem.dimension + 0.5)))
        else:
            self._n = max(minimum, int(pop_param))
        self._params["population_size"] = self._n

        self._kf_pool = self._as_vector(self._params.get("kf_pool"), "kf_pool")
        self._kr_pool = self._as_vector(self._params.get("kr_pool"), "kr_pool")
        self._initial_probabilities = self._as_vector(
            self._params.get("initial_setting_probabilities"),
            "initial_setting_probabilities",
        )
        self._validate_parameters()
        self._initial_probabilities = self._normalise_probabilities(self._initial_probabilities)
        self._last_operator_contributions = self._blank_contributions()
        self._last_operator_counts = self._blank_counts()
        self._last_accepted = 0

    @staticmethod
    def _as_vector(values: Iterable[float] | None, name: str) -> np.ndarray:
        try:
            raw_values = () if values is None else values
            vector = np.asarray(raw_values, dtype=float).reshape(-1)
        except Exception as exc:  # pragma: no cover - defensive parameter error path
            raise ValueError(f"{name} must be a one-dimensional numeric sequence.") from exc
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain finite numeric values.")
        return vector

    def _validate_parameters(self) -> None:
        if self._kf_pool.size != self._kr_pool.size:
            raise ValueError("kf_pool and kr_pool must have the same length.")
        if self._initial_probabilities.size != self._kf_pool.size:
            raise ValueError(
                "initial_setting_probabilities must have one entry per (kf, kr) setting."
            )
        if np.any(self._kf_pool <= 0.0):
            raise ValueError("All kf_pool values must be positive.")
        if np.any((self._kr_pool < 0.0) | (self._kr_pool > 1.0)):
            raise ValueError("All kr_pool values must lie in [0, 1].")
        if np.any(self._initial_probabilities < 0.0) or float(np.sum(self._initial_probabilities)) <= 0.0:
            raise ValueError("initial_setting_probabilities must be non-negative and sum to a positive value.")

        fraction = float(self._params.get("senior_fraction", 0.05))
        if not 0.0 < fraction < 0.5:
            raise ValueError("senior_fraction must lie strictly between 0 and 0.5.")
        learning_rate = float(self._params.get("learning_rate", 0.05))
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1].")
        adaptation_start = float(self._params.get("adaptation_start", 0.10))
        if not 0.0 <= adaptation_start <= 1.0:
            raise ValueError("adaptation_start must lie in [0, 1].")
        minimum_probability = float(self._params.get("minimum_setting_probability", 0.05))
        if not 0.0 <= minimum_probability <= 1.0 / self._kf_pool.size:
            raise ValueError(
                "minimum_setting_probability must lie in [0, 1 / number_of_settings]."
            )
        low_rate_probability = float(self._params.get("low_knowledge_rate_probability", 0.50))
        if not 0.0 <= low_rate_probability <= 1.0:
            raise ValueError("low_knowledge_rate_probability must lie in [0, 1].")
        if int(self._params.get("integer_knowledge_rate_max", 20)) < 1:
            raise ValueError("integer_knowledge_rate_max must be at least 1.")
        if int(self._params.get("min_population_size", 12)) < 4:
            raise ValueError("min_population_size must be at least 4.")

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    @staticmethod
    def _normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float).copy()
        probabilities[~np.isfinite(probabilities)] = 0.0
        probabilities = np.maximum(probabilities, 0.0)
        total = float(np.sum(probabilities))
        if total <= 0.0:
            probabilities[:] = 1.0 / probabilities.size
        else:
            probabilities /= total
        return probabilities

    def _sample_knowledge_rates(self, count: int) -> np.ndarray:
        low_probability = float(self._params.get("low_knowledge_rate_probability", 0.50))
        integer_max = int(self._params.get("integer_knowledge_rate_max", 20))
        low_mask = np.random.random(int(count)) < low_probability
        rates = np.empty(int(count), dtype=float)
        rates[low_mask] = np.random.random(int(np.count_nonzero(low_mask)))
        rates[~low_mask] = np.random.randint(
            1,
            integer_max + 1,
            size=int(np.count_nonzero(~low_mask)),
        ).astype(float)
        return rates

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        return {
            "knowledge_rates": self._sample_knowledge_rates(pop.shape[0]),
            "setting_probabilities": self._initial_probabilities.copy(),
            "setting_improvements": np.full(self._kf_pool.size, 1.0 / self._kf_pool.size),
            "initial_population_size": int(pop.shape[0]),
            "last_setting_assignments": np.zeros(pop.shape[0], dtype=int),
        }

    def _progress(self, state, *, evaluations: int | None = None, next_step: bool = False) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            value = state.evaluations if evaluations is None else int(evaluations)
            return float(np.clip(value / self.config.max_evaluations, 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        step = state.step + (1 if next_step else 0)
        return float(np.clip(step / horizon, 0.0, 1.0))

    def _remaining_evaluations(self, state, population_size: int) -> int:
        if self.config.max_evaluations is None:
            return int(population_size)
        remaining = max(0, int(self.config.max_evaluations) - int(state.evaluations))
        return min(int(population_size), remaining)

    def _draw_setting_assignments(self, probabilities: np.ndarray, count: int) -> np.ndarray:
        return np.random.choice(
            self._kf_pool.size,
            size=int(count),
            replace=True,
            p=self._normalise_probabilities(probabilities),
        ).astype(int)

    def _adapt_probabilities(self, probabilities: np.ndarray, improvements: np.ndarray) -> np.ndarray:
        learning_rate = float(self._params.get("learning_rate", 0.05))
        updated = (1.0 - learning_rate) * probabilities + learning_rate * improvements
        return self._normalise_probabilities(updated)

    def _paper_improvement_probabilities(self, gains: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        improvement = np.zeros(self._kf_pool.size, dtype=float)
        for setting in range(self._kf_pool.size):
            selected = assignments == setting
            if np.any(selected):
                improvement[setting] = float(np.sum(gains[selected]))
        total = float(np.sum(improvement))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(self._kf_pool.size, 1.0 / self._kf_pool.size, dtype=float)

        improvement /= total
        minimum = float(self._params.get("minimum_setting_probability", 0.05))
        order = np.argsort(improvement)
        for index in order[:-1]:
            improvement[index] = max(float(improvement[index]), minimum)
        improvement[order[-1]] = 1.0 - float(np.sum(improvement[order[:-1]]))
        if improvement[order[-1]] < 0.0 or not np.all(np.isfinite(improvement)):
            improvement = np.maximum(improvement, minimum)
            improvement = self._normalise_probabilities(improvement)
        return improvement

    def _junior_indices(self, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(order.size)
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n, dtype=int)
        better = np.empty(n, dtype=int)
        worse = np.empty(n, dtype=int)
        random_peer = np.empty(n, dtype=int)

        for i in range(n):
            r = int(rank[i])
            if r == 0:
                better[i], worse[i] = int(order[1]), int(order[2])
            elif r == n - 1:
                better[i], worse[i] = int(order[n - 3]), int(order[n - 2])
            else:
                better[i], worse[i] = int(order[r - 1]), int(order[r + 1])
            candidates = np.asarray(
                [j for j in range(n) if j not in {i, int(better[i]), int(worse[i])}],
                dtype=int,
            )
            random_peer[i] = int(np.random.choice(candidates))
        return better, worse, random_peer

    def _senior_indices(self, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(order.size)
        p = float(self._params.get("senior_fraction", 0.05))
        edge = max(1, int(np.floor(n * p + 0.5)))
        edge = min(edge, max(1, (n - 1) // 2))
        best_pool = order[:edge]
        middle_pool = order[edge : n - edge]
        worst_pool = order[n - edge :]
        if middle_pool.size == 0:
            middle_pool = order
        return (
            np.random.choice(best_pool, size=n, replace=True).astype(int),
            np.random.choice(middle_pool, size=n, replace=True).astype(int),
            np.random.choice(worst_pool, size=n, replace=True).astype(int),
        )

    def _midpoint_bound_repair(self, candidates: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        repaired = np.asarray(candidates, dtype=float).copy()
        below = repaired < self._lo
        above = repaired > self._hi
        repaired[below] = 0.5 * (parents[below] + np.broadcast_to(self._lo, repaired.shape)[below])
        repaired[above] = 0.5 * (parents[above] + np.broadcast_to(self._hi, repaired.shape)[above])
        return repaired, below | above

    def _target_population_size(self, progress_after: float, initial_size: int) -> int:
        minimum = max(4, int(self._params.get("min_population_size", 12)))
        target = initial_size + (minimum - initial_size) * float(np.clip(progress_after, 0.0, 1.0))
        return max(minimum, int(np.floor(target + 0.5)))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = int(pop.shape[0]), self.problem.dimension
        if n < 4:
            raise ValueError("AGSK requires at least four population members.")

        eval_count = self._remaining_evaluations(state, n)
        if eval_count <= 0:
            self._last_operator_contributions = self._blank_contributions()
            self._last_operator_counts = self._blank_counts()
            return pop, 0, {}

        knowledge_rates = np.asarray(state.payload["knowledge_rates"], dtype=float).copy()
        probabilities = np.asarray(state.payload["setting_probabilities"], dtype=float).copy()
        previous_improvements = np.asarray(state.payload["setting_improvements"], dtype=float).copy()
        initial_size = int(state.payload.get("initial_population_size", n))
        progress_before = self._progress(state)

        counts = self._blank_counts()
        contributions = self._blank_contributions()

        if progress_before < float(self._params.get("adaptation_start", 0.10)):
            probabilities = self._initial_probabilities.copy()
        else:
            probabilities = self._adapt_probabilities(probabilities, previous_improvements)
            counts["agsk.parameter_adaptation"] = 1

        assignments = self._draw_setting_assignments(probabilities, n)
        kf = self._kf_pool[assignments]
        kr = self._kr_pool[assignments]
        counts["agsk.parameter_setting_sampling"] = n

        parent_positions = pop[:, :-1].copy()
        parent_fitness = pop[:, -1].copy()
        order = self._order(parent_fitness)

        junior_better, junior_worse, junior_random = self._junior_indices(order)
        junior = np.empty_like(parent_positions)
        random_is_better = self._better_mask(parent_fitness[junior_random], parent_fitness)
        junior[random_is_better] = parent_positions[random_is_better] + kf[random_is_better, None] * (
            parent_positions[junior_better[random_is_better]]
            - parent_positions[junior_worse[random_is_better]]
            + parent_positions[junior_random[random_is_better]]
            - parent_positions[random_is_better]
        )
        not_better = ~random_is_better
        junior[not_better] = parent_positions[not_better] + kf[not_better, None] * (
            parent_positions[junior_better[not_better]]
            - parent_positions[junior_worse[not_better]]
            + parent_positions[not_better]
            - parent_positions[junior_random[not_better]]
        )

        senior_best, senior_middle, senior_worst = self._senior_indices(order)
        senior = np.empty_like(parent_positions)
        middle_is_better = self._better_mask(parent_fitness[senior_middle], parent_fitness)
        senior[middle_is_better] = parent_positions[middle_is_better] + kf[middle_is_better, None] * (
            parent_positions[senior_best[middle_is_better]]
            - parent_positions[middle_is_better]
            + parent_positions[senior_middle[middle_is_better]]
            - parent_positions[senior_worst[middle_is_better]]
        )
        not_better = ~middle_is_better
        senior[not_better] = parent_positions[not_better] + kf[not_better, None] * (
            parent_positions[senior_best[not_better]]
            - parent_positions[senior_middle[not_better]]
            + parent_positions[not_better]
            - parent_positions[senior_worst[not_better]]
        )

        junior, junior_repaired = self._midpoint_bound_repair(junior, parent_positions)
        senior, senior_repaired = self._midpoint_bound_repair(senior, parent_positions)

        junior_dimensions = np.ceil(
            dim * np.power(max(0.0, 1.0 - progress_before), knowledge_rates)
        ).astype(int)
        junior_dimensions = np.clip(junior_dimensions, 0, dim)
        junior_phase_mask = np.random.random((n, dim)) <= (junior_dimensions[:, None] / float(dim))
        senior_phase_mask = ~junior_phase_mask
        junior_mask = junior_phase_mask & (np.random.random((n, dim)) <= kr[:, None])
        senior_mask = senior_phase_mask & (np.random.random((n, dim)) <= kr[:, None])

        trial_positions = parent_positions.copy()
        trial_positions[junior_mask] = junior[junior_mask]
        trial_positions[senior_mask] = senior[senior_mask]
        trial_positions = np.clip(trial_positions, self._lo, self._hi)

        evaluated_positions = trial_positions[:eval_count]
        evaluated_fitness = self._evaluate_population(evaluated_positions)
        trial_fitness = parent_fitness.copy()
        trial_fitness[:eval_count] = evaluated_fitness
        strict = np.zeros(n, dtype=bool)
        strict[:eval_count] = self._better_mask(evaluated_fitness, parent_fitness[:eval_count])

        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[strict] = parent_fitness[strict] - trial_fitness[strict]
        else:
            gains[strict] = trial_fitness[strict] - parent_fitness[strict]
        gains = np.maximum(gains, 0.0)

        evaluated = np.arange(n) < eval_count
        counts["agsk.junior_gaining_sharing"] = int(
            np.count_nonzero(evaluated & np.any(junior_mask, axis=1))
        )
        counts["agsk.senior_gaining_sharing"] = int(
            np.count_nonzero(evaluated & np.any(senior_mask, axis=1))
        )
        counts["agsk.midpoint_bound_repair"] = int(
            np.count_nonzero((junior_repaired & junior_mask)[:eval_count])
            + np.count_nonzero((senior_repaired & senior_mask)[:eval_count])
        )
        counts["agsk.greedy_selection"] = eval_count

        for i in np.flatnonzero(strict):
            gain = float(gains[i])
            contributions["agsk.greedy_selection"] += gain / 3.0
            junior_delta = np.where(junior_mask[i], junior[i] - parent_positions[i], 0.0)
            senior_delta = np.where(senior_mask[i], senior[i] - parent_positions[i], 0.0)
            junior_norm = float(np.linalg.norm(junior_delta))
            senior_norm = float(np.linalg.norm(senior_delta))
            movement = junior_norm + senior_norm
            if movement <= 1.0e-30:
                contributions["agsk.junior_gaining_sharing"] += gain / 3.0
                contributions["agsk.senior_gaining_sharing"] += gain / 3.0
            else:
                remaining = 2.0 * gain / 3.0
                contributions["agsk.junior_gaining_sharing"] += remaining * junior_norm / movement
                contributions["agsk.senior_gaining_sharing"] += remaining * senior_norm / movement

        next_population = pop.copy()
        next_population[strict, :-1] = trial_positions[strict]
        next_population[strict, -1] = trial_fitness[strict]
        self._last_accepted = int(np.count_nonzero(strict))

        setting_improvements = self._paper_improvement_probabilities(
            gains[:eval_count], assignments[:eval_count]
        )

        evaluations_after = state.evaluations + eval_count
        progress_after = self._progress(state, evaluations=evaluations_after, next_step=True)
        target_size = self._target_population_size(progress_after, initial_size)
        if n > target_size:
            remove_count = n - target_size
            worst = self._order(next_population[:, -1])[-remove_count:]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            next_population = next_population[keep_mask]
            knowledge_rates = knowledge_rates[keep_mask]
            assignments = assignments[keep_mask]
            counts["agsk.linear_population_size_reduction"] = int(remove_count)

        self._last_operator_contributions = {
            key: float(max(0.0, value)) for key, value in contributions.items()
        }
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return next_population, eval_count, {
            "knowledge_rates": knowledge_rates,
            "setting_probabilities": probabilities,
            "setting_improvements": setting_improvements,
            "initial_population_size": initial_size,
            "last_setting_assignments": assignments,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "objective_consistent_positive"
        obs["evomapx_fidelity"] = "native"
        obs["setting_probabilities"] = np.asarray(
            state.payload.get("setting_probabilities", self._initial_probabilities), dtype=float
        ).tolist()
        obs["population_size"] = int(state.payload["population"].shape[0])
        obs["accepted_offspring"] = int(self._last_accepted)
        return obs
