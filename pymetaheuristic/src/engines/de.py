"""pyMetaheuristic src — Differential Evolution Engine.

Faithful implementation of the Storn--Price DE/rand/1/bin search engine,
with DE/best/2/bin available as the paper's explicitly mentioned variant.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


class DEEngine(BaseEngine):
    """Differential Evolution for continuous-space global optimization."""

    algorithm_id = "de"
    algorithm_name = "Differential Evolution"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1023/A:1008202821328",
        "authors": "R. Storn and K. Price",
        "title": "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces",
        "year": 1997,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    # Section 4 of the paper gives NP in [5D, 10D] as a rule of thumb, NP >= 4,
    # F = 0.5 as a good initial choice, and CR = 0.1 as a good first choice.
    _DEFAULTS: dict[str, Any] = dict(
        size=None,
        population_factor=10.0,
        F=0.5,
        Cr=0.1,
        strategy="rand1bin",
        bounds_policy="clip",
    )

    _NATIVE_STRATEGIES = {"rand1bin", "best2bin"}
    _COMPATIBILITY_STRATEGIES = {"best1bin", "current-to-best1bin"}
    _OPERATOR_LABELS = ("de.mutation", "de.crossover", "de.selection", "de.bound_repair")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **(config.params or {})}
        dim = int(problem.dimension)
        requested_size = p.get("size", p.get("population_size", p.get("NP", p.get("n"))))
        if requested_size is None:
            factor = float(p.get("population_factor", 10.0))
            self._n = max(4, int(np.ceil(factor * dim)))
        else:
            self._n = max(4, int(requested_size))

        self._F = float(p["F"])
        self._Cr = float(p["Cr"])
        self._strategy = self._normalize_strategy(str(p.get("strategy", "rand1bin")))
        self._bounds_policy = str(p.get("bounds_policy", "clip")).strip().lower()

        if not (0.0 < self._F <= 2.0):
            raise ValueError("de F must be in (0, 2].")
        if not (0.0 <= self._Cr <= 1.0):
            raise ValueError("de Cr must be in [0, 1].")
        if self._strategy not in self._NATIVE_STRATEGIES | self._COMPATIBILITY_STRATEGIES:
            allowed = sorted(self._NATIVE_STRATEGIES | self._COMPATIBILITY_STRATEGIES)
            raise ValueError(f"de strategy must be one of {allowed}.")
        if self._bounds_policy not in {"clip", "none"}:
            raise ValueError("de bounds_policy must be 'clip' or 'none'.")
        if self._strategy == "best2bin" and self._n < 5:
            raise ValueError("de/best/2/bin requires population size >= 5.")

        if config.seed is not None:
            np.random.seed(config.seed)

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        s = strategy.strip().lower().replace("_", "-")
        aliases = {
            "de/rand/1/bin": "rand1bin",
            "rand/1/bin": "rand1bin",
            "rand-1-bin": "rand1bin",
            "rand1bin": "rand1bin",
            "de/best/2/bin": "best2bin",
            "best/2/bin": "best2bin",
            "best-2-bin": "best2bin",
            "best2bin": "best2bin",
            "de/best/1/bin": "best1bin",
            "best/1/bin": "best1bin",
            "best-1-bin": "best1bin",
            "best1bin": "best1bin",
            "current-to-best/1/bin": "current-to-best1bin",
            "current-to-best-1-bin": "current-to-best1bin",
            "current-to-best1bin": "current-to-best1bin",
        }
        return aliases.get(s, s)

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    def _fitness_gain(self, parent_fitness: float, child_fitness: float) -> float:
        if self.problem.objective == "min":
            return max(0.0, parent_fitness - child_fitness)
        return max(0.0, child_fitness - parent_fitness)

    def _apply_bounds(self, vector: np.ndarray) -> tuple[np.ndarray, bool]:
        if self._bounds_policy == "none":
            return np.asarray(vector, dtype=float).copy(), False
        before = np.asarray(vector, dtype=float)
        repaired = self.problem.clip_position(before)
        return repaired.astype(float), bool(np.any(np.abs(repaired - before) > 0.0))

    def _init_pop(self, n: int | None = None) -> np.ndarray:
        if n is None:
            n = self._n
        positions = np.random.uniform(self._lo, self._hi, (int(n), self.problem.dimension))
        positions = np.vstack([self.problem.clip_position(row) for row in positions])
        fitness = self._evaluate_population(positions)
        return np.hstack((positions, fitness[:, np.newaxis]))

    def initialize(self) -> EngineState:
        pop = self._init_pop()
        best_index = int(np.argmin(pop[:, -1]) if self.problem.objective == "min" else np.argmax(pop[:, -1]))
        elite = pop[best_index, :].copy()
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload=dict(
                population=pop,
                elite=elite,
                best_index=best_index,
                lineage=[],
                operator_counts={label: 0 for label in self._OPERATOR_LABELS},
                operator_contributions={label: 0.0 for label in self._OPERATOR_LABELS},
                acceptance_rate=0.0,
                bound_repairs=0,
                crossover_mean_fraction=0.0,
            ),
        )

    def _sample_indices(self, population_size: int, exclude: set[int], k: int) -> np.ndarray:
        pool = np.array([idx for idx in range(population_size) if idx not in exclude], dtype=int)
        if pool.size < k:
            raise ValueError(
                f"de strategy {self._strategy!r} requires at least {k} donor vectors "
                f"outside excluded indices; got {pool.size}."
            )
        return np.random.choice(pool, size=k, replace=False)

    def _make_mutant(
        self,
        old_pop: np.ndarray,
        target_index: int,
        best_vector: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        n = old_pop.shape[0]
        x_i = old_pop[target_index, :-1]

        if self._strategy == "best2bin":
            donor_ids = self._sample_indices(n, {target_index}, 4)
            r1, r2, r3, r4 = (old_pop[idx, :-1] for idx in donor_ids)
            mutant = best_vector + self._F * ((r1 + r2) - r3 - r4)
        elif self._strategy == "best1bin":
            donor_ids = self._sample_indices(n, {target_index}, 2)
            r1, r2 = (old_pop[idx, :-1] for idx in donor_ids)
            mutant = best_vector + self._F * (r1 - r2)
        elif self._strategy == "current-to-best1bin":
            donor_ids = self._sample_indices(n, {target_index}, 2)
            r1, r2 = (old_pop[idx, :-1] for idx in donor_ids)
            mutant = x_i + self._F * (best_vector - x_i) + self._F * (r1 - r2)
        else:
            donor_ids = self._sample_indices(n, {target_index}, 3)
            r1, r2, r3 = (old_pop[idx, :-1] for idx in donor_ids)
            mutant = r1 + self._F * (r2 - r3)

        return np.asarray(mutant, dtype=float), [int(idx) for idx in donor_ids]

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim = int(self.problem.dimension)
        mask = np.random.random(dim) <= self._Cr
        mask[np.random.randint(dim)] = True
        trial = np.where(mask, mutant, target)
        return trial.astype(float), mask

    def step(self, state: EngineState) -> EngineState:
        # Use a source generation and a separate successor generation, as in the
        # paper pseudocode's x1/x2 arrays. Donors are never sampled from candidates
        # already accepted earlier in the same generation.
        old_pop = np.asarray(state.payload["population"], dtype=float).copy()
        new_pop = old_pop.copy()
        elite = np.asarray(state.payload["elite"], dtype=float).copy()

        if self.problem.objective == "min":
            best_index = int(np.argmin(old_pop[:, -1]))
        else:
            best_index = int(np.argmax(old_pop[:, -1]))
        best_vector = old_pop[best_index, :-1].copy()

        counts = {label: 0 for label in self._OPERATOR_LABELS}
        contributions = {label: 0.0 for label in self._OPERATOR_LABELS}
        lineage: list[dict[str, Any]] = []
        accepted_count = 0
        bound_repairs = 0
        crossover_fraction_sum = 0.0
        evaluations = 0

        for i in range(old_pop.shape[0]):
            parent_position = old_pop[i, :-1].copy()
            parent_fitness = float(old_pop[i, -1])

            mutant, donor_ids = self._make_mutant(old_pop, i, best_vector)
            trial, mask = self._binomial_crossover(parent_position, mutant)
            trial, repaired = self._apply_bounds(trial)
            if repaired:
                bound_repairs += 1
                counts["de.bound_repair"] += 1

            trial_fitness = float(self.problem.evaluate(trial))
            evaluations += 1
            crossover_fraction = float(np.mean(mask))
            crossover_fraction_sum += crossover_fraction

            counts["de.mutation"] += 1
            counts["de.crossover"] += 1
            counts["de.selection"] += 1

            accepted = self.problem.is_better(trial_fitness, parent_fitness) or trial_fitness == parent_fitness
            gain = self._fitness_gain(parent_fitness, trial_fitness) if accepted else 0.0
            if accepted:
                new_pop[i, :-1] = trial
                new_pop[i, -1] = trial_fitness
                accepted_count += 1
                share = gain / 3.0
                contributions["de.mutation"] += share
                contributions["de.crossover"] += share
                contributions["de.selection"] += share

            lineage.append(
                {
                    "id": f"de:{state.step + 1}:{i}",
                    "index": i,
                    "operator": "de.selection",
                    "parent_ids": [f"de:{state.step}:{i}", *[f"de:{state.step}:{j}" for j in donor_ids]],
                    "parent_index": i,
                    "donor_indices": donor_ids,
                    "best_index": best_index,
                    "parent_fitness": parent_fitness,
                    "trial_fitness": trial_fitness,
                    "child_fitness": float(new_pop[i, -1]),
                    "accepted": bool(accepted),
                    "mutated_dimensions": int(np.sum(mask)),
                    "crossover_fraction": crossover_fraction,
                    "bound_repaired": bool(repaired),
                }
            )

        if self.problem.objective == "min":
            generation_best_index = int(np.argmin(new_pop[:, -1]))
        else:
            generation_best_index = int(np.argmax(new_pop[:, -1]))
        if self.problem.is_better(float(new_pop[generation_best_index, -1]), float(elite[-1])):
            elite = new_pop[generation_best_index, :].copy()

        state.step += 1
        state.evaluations += evaluations
        state.payload = dict(
            population=new_pop,
            elite=elite,
            best_index=generation_best_index,
            lineage=lineage,
            operator_counts=counts,
            operator_contributions=contributions,
            acceptance_rate=float(accepted_count / max(1, old_pop.shape[0])),
            bound_repairs=int(bound_repairs),
            crossover_mean_fraction=float(crossover_fraction_sum / max(1, old_pop.shape[0])),
        )
        if self.problem.is_better(float(elite[-1]), float(state.best_fitness)):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        pop = np.asarray(state.payload["population"], dtype=float)
        fitness = pop[:, -1]
        positions = pop[:, :-1]
        denom = float(np.linalg.norm(self._hi - self._lo)) or 1.0
        centroid = positions.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / denom)

        operator_counts = dict(state.payload.get("operator_counts", {}))
        operator_contributions = dict(state.payload.get("operator_contributions", {}))
        active_labels = [label for label, count in operator_counts.items() if int(count) > 0]

        return dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=float(state.best_fitness),
            mean_fitness=float(np.mean(fitness)),
            std_fitness=float(np.std(fitness)),
            diversity=diversity,
            strategy=self._strategy,
            F=self._F,
            Cr=self._Cr,
            population_size=int(pop.shape[0]),
            acceptance_rate=float(state.payload.get("acceptance_rate", 0.0)),
            crossover_mean_fraction=float(state.payload.get("crossover_mean_fraction", 0.0)),
            bound_repairs=int(state.payload.get("bound_repairs", 0)),
            operator_contributions=operator_contributions,
            operator_counts=operator_counts,
            evomapx_operator_labels=active_labels or list(self._OPERATOR_LABELS[:3]),
            evomapx_delta_f="improvement_positive",
            evomapx_fidelity="native" if self._bounds_policy == "none" else "native_with_framework_bound_repair",
        )

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata=dict(
                algorithm_name=self.algorithm_name,
                elapsed_time=state.elapsed_time,
                population_size=self._n,
                NP=self._n,
                F=self._F,
                Cr=self._Cr,
                strategy=self._strategy,
                bounds_policy=self._bounds_policy,
                paper_native_strategy=bool(self._strategy in self._NATIVE_STRATEGIES),
                evomapx_operator_labels=list(self._OPERATOR_LABELS),
            ),
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current",
            )
            for i in range(pop.shape[0])
        ]
