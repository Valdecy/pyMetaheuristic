"""pyMetaheuristic src — j2020 Differential Evolution Engine.

Native NumPy implementation of the two-population self-adaptive differential
Evolution algorithm introduced by Brest, Sepesy Maucec, and Boskovic at CEC
2020.  The implementation preserves the paper's big/small population schedule,
jDE parameter self-adaptation, one-way best migration, progressive donor sharing,
independent restart tests, and crowding replacement in the big population.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import (
    CandidateRecord,
    CapabilityProfile,
    EngineState,
    OptimizationResult,
)
from ._ported_common import PortedPopulationEngine


class J2020Engine(PortedPopulationEngine):
    """Algorithm j2020 for single-objective box-constrained optimization."""

    algorithm_id = "j2020"
    algorithm_name = "Differential Evolution Algorithm for Single Objective Bound-Constrained Optimization"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC48606.2020.9185551",
        "authors": "Janez Brest, Mirjam Sepesy Maucec, and Borko Boskovic",
        "title": (
            "Differential Evolution Algorithm for Single Objective "
            "Bound-Constrained Optimization: Algorithm j2020"
        ),
        "year": 2020,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    _OPERATOR_LABELS = (
        "j2020.parameter_self_adaptation",
        "j2020.big_population_mutation",
        "j2020.small_population_mutation",
        "j2020.binomial_crossover",
        "j2020.bound_repair",
        "j2020.crowding_replacement",
        "j2020.greedy_selection",
        "j2020.best_migration",
        "j2020.big_population_restart",
        "j2020.small_population_restart",
        "j2020.candidate_injection",
    )

    _DEFAULTS: dict[str, Any] = dict(
        big_population_size=None,
        small_population_size=None,
        bNP=None,
        sNP=None,
        f_init=0.5,
        cr_init=0.9,
        f_lower_big=0.01,
        f_lower_small=0.17,
        f_span=1.1,
        cr_lower_big=0.0,
        cr_span_big=1.0,
        cr_lower_small=0.0,
        cr_span_small=0.7,
        tau1=0.1,
        tau2=0.1,
        stagnation_fraction=0.25,
        equal_fitness_tolerance=1.0e-16,
        minimum_equal_count=3,
        age_limit_fraction=0.1,
        age_limit_evaluations=None,
        max_small_donors=3,
        bounds_policy="wrap",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        dim = int(self.problem.dimension)

        total_requested = self._first_defined(config.params.get("population_size"))
        small_requested = self._first_defined(
            self._params.get("small_population_size"),
            self._params.get("sNP"),
        )
        big_requested = self._first_defined(
            self._params.get("big_population_size"),
            self._params.get("bNP"),
        )

        if total_requested is not None:
            total = int(total_requested)
            if total < 8:
                raise ValueError("j2020 population_size must be at least 8.")
            if small_requested is None and big_requested is None:
                candidates = []
                for small in range(4, total // 2 + 1):
                    big = total - small
                    if big >= small and big % small == 0:
                        ratio = big // small
                        score = (abs(small - max(4, dim)), abs(ratio - 7), small)
                        candidates.append((score, big, small))
                if not candidates:
                    raise ValueError(
                        "j2020 population_size must decompose as bNP + sNP with "
                        "bNP >= sNP >= 4 and bNP an integer multiple of sNP."
                    )
                _, self._bnp, self._snp = min(candidates, key=lambda item: item[0])
            elif small_requested is None:
                self._bnp = int(big_requested)
                self._snp = total - self._bnp
            elif big_requested is None:
                self._snp = int(small_requested)
                self._bnp = total - self._snp
            else:
                self._bnp = int(big_requested)
                self._snp = int(small_requested)
                if self._bnp + self._snp != total:
                    raise ValueError(
                        "j2020 population_size must equal big_population_size + "
                        "small_population_size."
                    )
        else:
            self._snp = max(4, dim) if small_requested is None else int(small_requested)
            self._bnp = 7 * self._snp if big_requested is None else int(big_requested)

        self._n = self._bnp + self._snp
        self._params["population_size"] = self._n
        self._params["big_population_size"] = self._bnp
        self._params["small_population_size"] = self._snp
        self._validate_parameters()

        if self.config.max_evaluations is not None and self.config.max_evaluations < self._n:
            raise ValueError(
                "j2020 max_evaluations must be at least the combined initial "
                f"population size ({self._n})."
            )

        self._last_operator_counts = self._blank_counts()
        self._last_operator_contributions = self._blank_contributions()

    @staticmethod
    def _first_defined(*values):
        for value in values:
            if value not in (None, "", 0, "auto", "paper"):
                return value
        return None

    def _validate_parameters(self) -> None:
        if self._bnp < 4:
            raise ValueError("j2020 big_population_size must be at least 4.")
        if self._snp < 4:
            raise ValueError("j2020 small_population_size must be at least 4.")
        if self._bnp < self._snp:
            raise ValueError("j2020 requires big_population_size >= small_population_size.")
        if self._bnp % self._snp != 0:
            raise ValueError(
                "j2020 requires big_population_size to be an integer multiple "
                "of small_population_size."
            )

        for key in ("tau1", "tau2", "stagnation_fraction", "age_limit_fraction"):
            value = float(self._params[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"j2020 {key} must lie in [0, 1].")

        if float(self._params["f_init"]) <= 0.0:
            raise ValueError("j2020 f_init must be positive.")
        if float(self._params["f_span"]) <= 0.0:
            raise ValueError("j2020 f_span must be positive.")
        if float(self._params["f_lower_big"]) < 0.0 or float(self._params["f_lower_small"]) < 0.0:
            raise ValueError("j2020 F lower limits must be non-negative.")

        for lower_key, span_key in (
            ("cr_lower_big", "cr_span_big"),
            ("cr_lower_small", "cr_span_small"),
        ):
            lower = float(self._params[lower_key])
            span = float(self._params[span_key])
            if lower < 0.0 or span < 0.0 or lower + span > 1.0 + 1.0e-15:
                raise ValueError(
                    f"j2020 {lower_key} + {span_key} must define a range inside [0, 1]."
                )

        if not 0.0 <= float(self._params["cr_init"]) <= 1.0:
            raise ValueError("j2020 cr_init must lie in [0, 1].")
        if float(self._params["equal_fitness_tolerance"]) < 0.0:
            raise ValueError("j2020 equal_fitness_tolerance must be non-negative.")
        if int(self._params["minimum_equal_count"]) < 1:
            raise ValueError("j2020 minimum_equal_count must be at least 1.")
        if int(self._params["max_small_donors"]) < 1:
            raise ValueError("j2020 max_small_donors must be at least 1.")

        policy = str(self._params.get("bounds_policy", "wrap")).strip().lower()
        if policy not in {"wrap", "reflect", "clip"}:
            raise ValueError("j2020 bounds_policy must be 'wrap', 'reflect', or 'clip'.")
        self._bounds_policy = policy

        explicit_age = self._params.get("age_limit_evaluations")
        if explicit_age not in (None, "", "auto", "paper") and int(explicit_age) < 1:
            raise ValueError("j2020 age_limit_evaluations must be at least 1 when provided.")

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    def _fitness_gain(self, old: float, new: float) -> float:
        if self.problem.objective == "min":
            return max(0.0, float(old) - float(new))
        return max(0.0, float(new) - float(old))

    def _not_worse(self, candidate: float, incumbent: float) -> bool:
        return self.problem.is_better(float(candidate), float(incumbent)) or float(candidate) == float(incumbent)

    def _best_index(self, fit: np.ndarray) -> int:
        return int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))

    def _age_limit(self) -> int:
        explicit = self._params.get("age_limit_evaluations")
        if explicit not in (None, "", "auto", "paper"):
            return max(1, int(explicit))
        fraction = float(self._params.get("age_limit_fraction", 0.1))
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return max(1, int(np.floor(fraction * self.config.max_evaluations)))
        if self.config.max_steps is not None and self.config.max_steps > 0:
            estimated = int(self.config.max_steps) * 2 * self._bnp
            return max(1, int(np.floor(fraction * estimated)))
        return max(1, 10 * self._bnp)

    def _remaining_evaluations(self, state: EngineState) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(0, int(self.config.max_evaluations) - int(state.evaluations))

    def _can_evaluate(self, state: EngineState, count: int = 1) -> bool:
        remaining = self._remaining_evaluations(state)
        return remaining is None or remaining >= int(count)

    def _progress(self, state: EngineState) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(state.evaluations / self.config.max_evaluations, 0.0, 1.0))
        if self.config.max_steps is not None and self.config.max_steps > 0:
            return float(np.clip(state.step / self.config.max_steps, 0.0, 1.0))
        return 0.0

    def _small_donor_count(self, state: EngineState) -> int:
        progress = self._progress(state)
        if progress <= 1.0 / 3.0:
            count = 1
        elif progress <= 2.0 / 3.0:
            count = 2
        else:
            count = 3
        return min(count, int(self._params["max_small_donors"]), self._snp)

    def _apply_bounds(self, vector: np.ndarray) -> tuple[np.ndarray, bool]:
        raw = np.asarray(vector, dtype=float)
        lo = self._lo
        hi = self._hi
        span = hi - lo
        repaired = raw.copy()
        fixed = span <= 0.0

        if self._bounds_policy == "clip":
            repaired = np.clip(repaired, lo, hi)
        elif self._bounds_policy == "wrap":
            free = ~fixed
            repaired[fixed] = lo[fixed]
            repaired[free] = lo[free] + np.mod(repaired[free] - lo[free], span[free])
        else:  # reflect
            free = ~fixed
            repaired[fixed] = lo[fixed]
            period = 2.0 * span[free]
            folded = np.mod(repaired[free] - lo[free], period)
            repaired[free] = lo[free] + np.where(
                folded <= span[free],
                folded,
                period - folded,
            )

        repaired = self.problem.apply_variable_types(repaired).astype(float)
        changed = bool(np.any(np.abs(repaired - raw) > 0.0))
        return repaired, changed

    def _too_many_equal(self, fitness: np.ndarray) -> bool:
        values = np.asarray(fitness, dtype=float)
        if values.size == 0:
            return False
        best = float(values[self._best_index(values)])
        tolerance = float(self._params["equal_fitness_tolerance"])
        equal_count = int(np.count_nonzero(np.abs(values - best) < tolerance))
        threshold = float(self._params["stagnation_fraction"]) * values.size
        minimum = int(self._params["minimum_equal_count"])
        return equal_count > threshold and equal_count >= minimum

    def _update_best_from_population(self, state: EngineState, pop: np.ndarray) -> None:
        index = self._best_index(pop[:, -1])
        row = pop[index]
        if state.best_fitness is None or self.problem.is_better(float(row[-1]), float(state.best_fitness)):
            state.best_fitness = float(row[-1])
            state.best_position = row[:-1].tolist()

    def _initialize_operator_payload(self) -> dict[str, Any]:
        return {
            "operator_counts": self._blank_counts(),
            "operator_contributions": self._blank_contributions(),
            "pending_operator_counts": self._blank_counts(),
            "pending_operator_contributions": self._blank_contributions(),
        }

    def initialize(self) -> EngineState:
        positions = self._new_positions(self._n)
        pop = self._pop_from_positions(positions)
        best_index = self._best_index(pop[:, -1])
        payload: dict[str, Any] = {
            "population": pop,
            "F": np.full(self._n, float(self._params["f_init"]), dtype=float),
            "CR": np.full(self._n, float(self._params["cr_init"]), dtype=float),
            "big_size": self._bnp,
            "small_size": self._snp,
            "big_no_improvement_evals": 0,
            "big_restart_count": 0,
            "small_restart_count": 0,
            "migration_count": 0,
            "accepted_big": 0,
            "accepted_small": 0,
            "last_small_donor_count": 1,
        }
        payload.update(self._initialize_operator_payload())
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=pop[best_index, :-1].tolist(),
            best_fitness=float(pop[best_index, -1]),
            initialized=True,
            payload=payload,
        )

    def _consume_pending_telemetry(
        self,
        state: EngineState,
        counts: dict[str, int],
        contributions: dict[str, float],
    ) -> None:
        pending_counts = state.payload.get("pending_operator_counts", {})
        pending_contrib = state.payload.get("pending_operator_contributions", {})
        for label in self._OPERATOR_LABELS:
            counts[label] += int(pending_counts.get(label, 0))
            contributions[label] += float(pending_contrib.get(label, 0.0))
        state.payload["pending_operator_counts"] = self._blank_counts()
        state.payload["pending_operator_contributions"] = self._blank_contributions()

    def _restart_slice(
        self,
        state: EngineState,
        start: int,
        stop: int,
        preserve_index: int | None,
        label: str,
        counts: dict[str, int],
        contributions: dict[str, float],
    ) -> int:
        indices = np.arange(start, stop, dtype=int)
        if preserve_index is not None:
            indices = indices[indices != int(preserve_index)]
        if indices.size == 0 or not self._can_evaluate(state, int(indices.size)):
            return 0

        pop = np.asarray(state.payload["population"], dtype=float)
        F = np.asarray(state.payload["F"], dtype=float)
        CR = np.asarray(state.payload["CR"], dtype=float)
        old_fitness = pop[indices, -1].copy()
        positions = self._new_positions(int(indices.size))
        fitness = self._evaluate_population(positions)
        pop[indices, :-1] = positions
        pop[indices, -1] = fitness
        F[indices] = float(self._params["f_init"])
        CR[indices] = float(self._params["cr_init"])

        gains = [self._fitness_gain(old, new) for old, new in zip(old_fitness, fitness)]
        counts[label] += int(indices.size)
        contributions[label] += float(np.sum(gains))
        state.evaluations += int(indices.size)
        state.payload["population"] = pop
        state.payload["F"] = F
        state.payload["CR"] = CR
        self._update_best_from_population(state, pop)
        return int(indices.size)

    def _maybe_restart_big(
        self,
        state: EngineState,
        counts: dict[str, int],
        contributions: dict[str, float],
    ) -> bool:
        pop = np.asarray(state.payload["population"], dtype=float)
        equal_trigger = self._too_many_equal(pop[: self._bnp, -1])
        age_trigger = int(state.payload.get("big_no_improvement_evals", 0)) >= self._age_limit()
        if not (equal_trigger or age_trigger):
            return False
        evaluated = self._restart_slice(
            state,
            0,
            self._bnp,
            preserve_index=None,
            label="j2020.big_population_restart",
            counts=counts,
            contributions=contributions,
        )
        if evaluated:
            state.payload["big_no_improvement_evals"] = 0
            state.payload["big_restart_count"] = int(state.payload.get("big_restart_count", 0)) + 1
            return True
        return False

    def _maybe_restart_small(
        self,
        state: EngineState,
        counts: dict[str, int],
        contributions: dict[str, float],
    ) -> bool:
        pop = np.asarray(state.payload["population"], dtype=float)
        small_fit = pop[self._bnp :, -1]
        if not self._too_many_equal(small_fit):
            return False
        preserve_local = self._best_index(small_fit)
        preserve_global = self._bnp + preserve_local
        evaluated = self._restart_slice(
            state,
            self._bnp,
            self._n,
            preserve_index=preserve_global,
            label="j2020.small_population_restart",
            counts=counts,
            contributions=contributions,
        )
        if evaluated:
            state.payload["small_restart_count"] = int(state.payload.get("small_restart_count", 0)) + 1
            return True
        return False

    def _adapt_parameters(self, index: int, big: bool, F: np.ndarray, CR: np.ndarray) -> tuple[float, float, int]:
        changes = 0
        f_value = float(F[index])
        cr_value = float(CR[index])
        if np.random.random() < float(self._params["tau1"]):
            lower = float(self._params["f_lower_big"] if big else self._params["f_lower_small"])
            f_value = lower + np.random.random() * float(self._params["f_span"])
            changes += 1
        if np.random.random() < float(self._params["tau2"]):
            lower = float(self._params["cr_lower_big"] if big else self._params["cr_lower_small"])
            span = float(self._params["cr_span_big"] if big else self._params["cr_span_small"])
            cr_value = lower + np.random.random() * span
            changes += 1
        return f_value, cr_value, changes

    def _sample_big_donors(self, target: int, small_donor_count: int) -> tuple[int, int, int]:
        r1_pool = np.asarray([i for i in range(self._bnp) if i != target], dtype=int)
        r1 = int(np.random.choice(r1_pool))
        shared = np.arange(self._bnp, self._bnp + small_donor_count, dtype=int)
        union = np.concatenate((np.arange(self._bnp, dtype=int), shared))
        pool = union[(union != target) & (union != r1)]
        r2, r3 = np.random.choice(pool, size=2, replace=False).astype(int)
        return r1, int(r2), int(r3)

    def _sample_small_donors(self, target: int) -> tuple[int, int, int]:
        indices = np.arange(self._bnp, self._n, dtype=int)
        pool = indices[indices != target]
        r1, r2, r3 = np.random.choice(pool, size=3, replace=False).astype(int)
        return int(r1), int(r2), int(r3)

    def _trial(
        self,
        state: EngineState,
        target: int,
        big: bool,
        small_donor_count: int,
        counts: dict[str, int],
        contributions: dict[str, float],
    ) -> bool:
        if not self._can_evaluate(state, 1):
            return False

        pop = np.asarray(state.payload["population"], dtype=float)
        F = np.asarray(state.payload["F"], dtype=float)
        CR = np.asarray(state.payload["CR"], dtype=float)
        f_value, cr_value, adaptations = self._adapt_parameters(target, big, F, CR)
        counts["j2020.parameter_self_adaptation"] += adaptations

        if big:
            r1, r2, r3 = self._sample_big_donors(target, small_donor_count)
            mutation_label = "j2020.big_population_mutation"
        else:
            r1, r2, r3 = self._sample_small_donors(target)
            mutation_label = "j2020.small_population_mutation"

        mutant = pop[r1, :-1] + f_value * (pop[r2, :-1] - pop[r3, :-1])
        crossover_mask = np.random.random(self.problem.dimension) <= cr_value
        crossover_mask[int(np.random.randint(self.problem.dimension))] = True
        trial = np.where(crossover_mask, mutant, pop[target, :-1])
        trial, repaired = self._apply_bounds(trial)
        trial_fitness = float(self.problem.evaluate(trial))
        state.evaluations += 1

        counts[mutation_label] += 1
        counts["j2020.binomial_crossover"] += 1
        counts["j2020.greedy_selection"] += 1
        if repaired:
            counts["j2020.bound_repair"] += 1

        competitor = target
        if big:
            distances = np.sum((pop[: self._bnp, :-1] - trial) ** 2, axis=1)
            competitor = int(np.argmin(distances))
            counts["j2020.crowding_replacement"] += 1

        incumbent = float(pop[competitor, -1])
        accepted = self._not_worse(trial_fitness, incumbent)
        gain = self._fitness_gain(incumbent, trial_fitness) if accepted else 0.0

        old_big_best = float(pop[: self._bnp, -1][self._best_index(pop[: self._bnp, -1])]) if big else None
        if accepted:
            pop[competitor, :-1] = trial
            pop[competitor, -1] = trial_fitness
            F[competitor] = f_value
            CR[competitor] = cr_value

            direct_labels = [mutation_label, "j2020.binomial_crossover", "j2020.greedy_selection"]
            if big:
                direct_labels.append("j2020.crowding_replacement")
            if repaired:
                direct_labels.append("j2020.bound_repair")
            share = gain / max(1, len(direct_labels))
            for label in direct_labels:
                contributions[label] += share

        if big:
            state.payload["big_no_improvement_evals"] = int(
                state.payload.get("big_no_improvement_evals", 0)
            ) + 1
            new_big_best = float(pop[: self._bnp, -1][self._best_index(pop[: self._bnp, -1])])
            if old_big_best is not None and self.problem.is_better(new_big_best, old_big_best):
                state.payload["big_no_improvement_evals"] = 0
            if accepted:
                state.payload["accepted_big"] = int(state.payload.get("accepted_big", 0)) + 1
        elif accepted:
            state.payload["accepted_small"] = int(state.payload.get("accepted_small", 0)) + 1

        state.payload["population"] = pop
        state.payload["F"] = F
        state.payload["CR"] = CR
        self._update_best_from_population(state, pop)
        return True

    def _migrate_best_to_small(
        self,
        state: EngineState,
        counts: dict[str, int],
    ) -> bool:
        pop = np.asarray(state.payload["population"], dtype=float)
        overall_index = self._best_index(pop[:, -1])
        if overall_index >= self._bnp:
            return False
        destination = self._bnp
        pop[destination] = pop[overall_index]
        state.payload["population"] = pop
        counts["j2020.best_migration"] += 1
        state.payload["migration_count"] = int(state.payload.get("migration_count", 0)) + 1
        return True

    def step(self, state: EngineState) -> EngineState:
        counts = self._blank_counts()
        contributions = self._blank_contributions()
        self._consume_pending_telemetry(state, counts, contributions)
        state.payload["accepted_big"] = 0
        state.payload["accepted_small"] = 0

        self._maybe_restart_big(state, counts, contributions)
        small_donor_count = self._small_donor_count(state)
        state.payload["last_small_donor_count"] = small_donor_count

        for target in range(self._bnp):
            if not self._trial(
                state,
                target,
                big=True,
                small_donor_count=small_donor_count,
                counts=counts,
                contributions=contributions,
            ):
                break

        if self._can_evaluate(state, 1):
            self._maybe_restart_small(state, counts, contributions)
            self._migrate_best_to_small(state, counts)

            small_generations = self._bnp // self._snp
            for _ in range(small_generations):
                for target in range(self._bnp, self._n):
                    if not self._trial(
                        state,
                        target,
                        big=False,
                        small_donor_count=small_donor_count,
                        counts=counts,
                        contributions=contributions,
                    ):
                        break
                if not self._can_evaluate(state, 1):
                    break

        state.step += 1
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        self._last_operator_contributions = {
            key: float(max(0.0, value)) for key, value in contributions.items()
        }
        state.payload["operator_counts"] = dict(self._last_operator_counts)
        state.payload["operator_contributions"] = dict(self._last_operator_contributions)
        self._update_best_from_population(state, np.asarray(state.payload["population"], dtype=float))
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        pop = np.asarray(state.payload["population"], dtype=float)
        positions = pop[:, :-1]
        fitness = pop[:, -1]
        denom = float(np.linalg.norm(self._hi - self._lo)) or 1.0
        centroid = positions.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / denom)
        active_labels = [
            label for label, count in self._last_operator_counts.items() if int(count) > 0
        ]
        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "diversity": diversity,
            "population_size": int(pop.shape[0]),
            "big_population_size": self._bnp,
            "small_population_size": self._snp,
            "accepted_big": int(state.payload.get("accepted_big", 0)),
            "accepted_small": int(state.payload.get("accepted_small", 0)),
            "big_no_improvement_evals": int(state.payload.get("big_no_improvement_evals", 0)),
            "big_restart_count": int(state.payload.get("big_restart_count", 0)),
            "small_restart_count": int(state.payload.get("small_restart_count", 0)),
            "migration_count": int(state.payload.get("migration_count", 0)),
            "small_donor_count": int(state.payload.get("last_small_donor_count", 1)),
            "operator_contributions": dict(self._last_operator_contributions),
            "operator_counts": dict(self._last_operator_counts),
            "evomapx_operator_labels": active_labels or list(self._OPERATOR_LABELS),
            "evomapx_delta_f": "objective_consistent_positive",
            "evomapx_fidelity": "native",
        }

    def _post_injection_repair(
        self,
        state: EngineState,
        replaced_indices: list[int],
        candidates: list[CandidateRecord],
    ) -> None:
        F = np.asarray(state.payload.get("F"), dtype=float)
        CR = np.asarray(state.payload.get("CR"), dtype=float)
        if F.shape[0] == self._n and CR.shape[0] == self._n:
            for index in replaced_indices:
                F[index] = float(self._params["f_init"])
                CR[index] = float(self._params["cr_init"])
            state.payload["F"] = F
            state.payload["CR"] = CR
        self._update_best_from_population(
            state,
            np.asarray(state.payload["population"], dtype=float),
        )

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        if not candidates:
            return state
        pop = np.asarray(state.payload["population"], dtype=float)
        fitness = pop[:, -1]
        worst_first = self._order(fitness)[::-1]
        remaining = self._remaining_evaluations(state)
        candidate_limit = len(candidates) if remaining is None else min(len(candidates), remaining)
        if candidate_limit <= 0:
            return state
        replaced: list[int] = []
        total_gain = 0.0
        for index, candidate in zip(worst_first, candidates[:candidate_limit]):
            position = self.problem.clip_position(candidate.position).astype(float)
            new_fitness = float(self.problem.evaluate(position))
            state.evaluations += 1
            old_fitness = float(pop[index, -1])
            pop[index, :-1] = position
            pop[index, -1] = new_fitness
            total_gain += self._fitness_gain(old_fitness, new_fitness)
            replaced.append(int(index))
        state.payload["population"] = pop
        self._post_injection_repair(state, replaced, candidates)
        pending_counts = state.payload.setdefault("pending_operator_counts", self._blank_counts())
        pending_contrib = state.payload.setdefault(
            "pending_operator_contributions", self._blank_contributions()
        )
        pending_counts["j2020.candidate_injection"] = int(
            pending_counts.get("j2020.candidate_injection", 0)
        ) + len(replaced)
        pending_contrib["j2020.candidate_injection"] = float(
            pending_contrib.get("j2020.candidate_injection", 0.0)
        ) + total_gain
        return state

    def restart(
        self,
        state: EngineState,
        seeds: list[CandidateRecord] | None = None,
        preserve_best: bool = True,
    ) -> EngineState:
        pop = np.asarray(state.payload["population"], dtype=float)
        preserve_index = self._best_index(pop[:, -1]) if preserve_best else None
        counts = self._blank_counts()
        contributions = self._blank_contributions()

        big_preserve = preserve_index if preserve_index is not None and preserve_index < self._bnp else None
        small_preserve = preserve_index if preserve_index is not None and preserve_index >= self._bnp else None
        big_restarted = self._restart_slice(
            state,
            0,
            self._bnp,
            big_preserve,
            "j2020.big_population_restart",
            counts,
            contributions,
        )
        small_restarted = self._restart_slice(
            state,
            self._bnp,
            self._n,
            small_preserve,
            "j2020.small_population_restart",
            counts,
            contributions,
        )
        if big_restarted:
            state.payload["big_no_improvement_evals"] = 0
            state.payload["big_restart_count"] = int(
                state.payload.get("big_restart_count", 0)
            ) + 1
        if small_restarted:
            state.payload["small_restart_count"] = int(
                state.payload.get("small_restart_count", 0)
            ) + 1

        pending_counts = state.payload.setdefault("pending_operator_counts", self._blank_counts())
        pending_contrib = state.payload.setdefault(
            "pending_operator_contributions", self._blank_contributions()
        )
        for label in self._OPERATOR_LABELS:
            pending_counts[label] = int(pending_counts.get(label, 0)) + counts[label]
            pending_contrib[label] = float(pending_contrib.get(label, 0.0)) + contributions[label]

        if seeds:
            state = self.inject_candidates(state, seeds, policy="native")
        return state

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "elapsed_time": float(state.elapsed_time),
                "population_size": self._n,
                "big_population_size": self._bnp,
                "small_population_size": self._snp,
                "small_generations_per_cycle": self._bnp // self._snp,
                "bounds_policy": self._bounds_policy,
                "big_restart_count": int(state.payload.get("big_restart_count", 0)),
                "small_restart_count": int(state.payload.get("small_restart_count", 0)),
                "migration_count": int(state.payload.get("migration_count", 0)),
                "evomapx_operator_labels": list(self._OPERATOR_LABELS),
                "implementation_notes": (
                    "Defaults follow the CEC 2020 paper for D >= 4. For lower-dimensional "
                    "problems, sNP is raised to 4 so DE/rand/1 can sample three distinct "
                    "donors and bNP retains the paper's 7:1 ratio. Population restarts "
                    "are evaluated immediately and charged to the evaluation budget so "
                    "snapshot fitness remains valid, rather than using temporary infinite "
                    "fitness placeholders. The supplied reference code's periodic boundary "
                    "repair is exposed as bounds_policy='wrap'."
                ),
            },
        )
