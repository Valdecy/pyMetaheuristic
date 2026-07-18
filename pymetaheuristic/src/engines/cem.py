"""pyMetaheuristic src — Cross-Entropy Method engine.

Continuous, independent-normal CE implementation following Rubinstein and
Kroese (2004), Chapters 4–5.  Each native macro-step samples a complete
population from the current reference distribution, selects the objective-
appropriate elite quantile, and updates the normal parameters from that same
sample with CE smoothing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


class CEMEngine(BaseEngine):
    algorithm_id = "cem"
    algorithm_name = "Cross Entropy Method"
    family = "distribution"
    _REFERENCE = {
        "authors": "Reuven Y. Rubinstein and Dirk P. Kroese",
        "title": "The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning",
        "year": 2004,
        "doi": "10.1007/978-1-4757-4321-0",
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_discrete=False,
        supports_integer=False,
        supports_mixed=False,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    # Chapter 5's introductory continuous example uses N=100, rho=0.1,
    # alpha=0.7 and a standard-deviation stopping threshold of 0.05.
    _DEFAULTS = {
        "size": 100,
        "elite_fraction": 0.10,
        "smoothing": 0.70,
        "initial_mean": None,
        "initial_std": None,
        "std_scale": 0.50,
        "dynamic_std": False,
        "beta": 0.90,
        "q": 6.0,
        "std_tolerance": 0.05,
        "stagnation_window": 5,
        "level_tolerance": 1.0e-12,
        "min_std": 1.0e-12,
        "bound_handling": "truncated_normal",
    }

    _OPERATOR_LABELS = (
        "cem.model_sampling",
        "cem.elite_quantile_selection",
        "cem.mean_update",
        "cem.standard_deviation_update",
        "cem.boundary_handling",
        "cem.candidate_injection",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        raw = dict(config.params or {})
        p = {**self._DEFAULTS, **raw}

        # Backward-compatible aliases.  Their semantics are normalized to the
        # paper's convention: alpha weights the newly estimated parameter.
        if "rho" in raw and "elite_fraction" not in raw:
            p["elite_fraction"] = raw["rho"]
        if "alpha" in raw and "smoothing" not in raw:
            p["smoothing"] = raw["alpha"]
        if "learning_rate" in raw and "smoothing" not in raw and "alpha" not in raw:
            p["smoothing"] = raw["learning_rate"]
        if "mu" in raw and "initial_mean" not in raw:
            p["initial_mean"] = raw["mu"]
        if "sigma" in raw and "initial_std" not in raw:
            p["initial_std"] = raw["sigma"]

        self._n = int(p["size"])
        if self._n < 2:
            raise ValueError("CEM size must be at least 2")

        self._rho = float(p["elite_fraction"])
        if not 0.0 < self._rho <= 1.0:
            raise ValueError("elite_fraction (rho) must be in (0, 1]")

        explicit_k = raw.get("k_samples", None)
        self._elite_count = (
            int(explicit_k)
            if explicit_k is not None
            else int(np.ceil(self._rho * self._n))
        )
        if not 2 <= self._elite_count <= self._n:
            raise ValueError("CEM elite count must satisfy 2 <= elite_count <= size")
        self._rho_effective = self._elite_count / self._n

        self._alpha = float(p["smoothing"])
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError("smoothing (alpha) must be in (0, 1]")

        self._dynamic_std = bool(p["dynamic_std"])
        self._beta = float(p["beta"])
        self._q = float(p["q"])
        if not 0.0 < self._beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        if self._q <= 0.0:
            raise ValueError("q must be positive")

        self._std_tolerance = float(p["std_tolerance"])
        self._stagnation_window = int(p["stagnation_window"])
        self._level_tolerance = float(p["level_tolerance"])
        self._min_std = float(p["min_std"])
        if self._min_std <= 0.0:
            raise ValueError("min_std must be positive")

        self._bound_handling = str(p["bound_handling"]).strip().lower()
        if self._bound_handling not in {"truncated_normal", "clip"}:
            raise ValueError("bound_handling must be 'truncated_normal' or 'clip'")

        self._lo = np.asarray(problem.min_values, dtype=float)
        self._hi = np.asarray(problem.max_values, dtype=float)
        if self._lo.shape != (problem.dimension,) or self._hi.shape != (problem.dimension,):
            raise ValueError("Problem bounds must match problem.dimension")
        if not np.all(np.isfinite(self._lo)) or not np.all(np.isfinite(self._hi)):
            raise ValueError("This bounded CEM engine requires finite bounds")
        if np.any(self._hi <= self._lo):
            raise ValueError("Each upper bound must be greater than its lower bound")

        self._initial_mean = self._as_vector(
            p["initial_mean"], (self._lo + self._hi) / 2.0, "initial_mean"
        )
        self._initial_mean = np.clip(self._initial_mean, self._lo, self._hi)
        default_std = float(p["std_scale"]) * (self._hi - self._lo)
        self._initial_std = self._as_vector(
            p["initial_std"], default_std, "initial_std"
        )
        if np.any(self._initial_std <= 0.0):
            raise ValueError("initial_std must be strictly positive")
        self._initial_std = np.maximum(self._initial_std, self._min_std)

        self._rng = np.random.default_rng(config.seed)

    def _as_vector(self, value: Any, default: Any, name: str) -> np.ndarray:
        source = default if value is None else value
        arr = np.asarray(source, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.problem.dimension, float(arr), dtype=float)
        else:
            arr = np.ravel(arr).astype(float)
        if arr.shape != (self.problem.dimension,):
            raise ValueError(f"{name} must be scalar or have length {self.problem.dimension}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _rank_indices(self, fitness: np.ndarray) -> np.ndarray:
        order = np.argsort(np.asarray(fitness, dtype=float), kind="stable")
        return order if self.problem.objective == "min" else order[::-1]

    def _signed_improvement(self, new: float, old: float) -> float:
        return float(old - new) if self.problem.objective == "min" else float(new - old)

    def _empty_int_map(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _empty_float_map(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    @staticmethod
    def _accumulate(target: dict, increment: dict) -> dict:
        result = dict(target)
        for key, value in increment.items():
            result[key] = result.get(key, 0) + value
        return result

    def _sample_population(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, int]:
        std = np.maximum(np.asarray(std, dtype=float), self._min_std)
        mean = np.asarray(mean, dtype=float)
        shape = (self._n, self.problem.dimension)

        if self._bound_handling == "clip":
            raw = self._rng.normal(loc=mean, scale=std, size=shape)
            clipped = np.clip(raw, self._lo, self._hi)
            repairs = int(np.count_nonzero(raw != clipped))
            positions = clipped
        else:
            # Inverse-CDF sampling is equivalent to acceptance-rejection from
            # the independent normal conditioned on the finite search box, but
            # avoids the point masses produced by clipping.
            a = (self._lo - mean) / std
            b = (self._hi - mean) / std
            cdf_lo = ndtr(a)
            cdf_hi = ndtr(b)
            width = cdf_hi - cdf_lo
            if np.any(width <= np.finfo(float).tiny):
                raise FloatingPointError(
                    "Truncated-normal probability underflow; increase initial_std or min_std"
                )
            u = cdf_lo + self._rng.random(shape) * width
            u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            positions = mean + std * ndtri(u)
            positions = np.clip(positions, self._lo, self._hi)  # round-off only
            repairs = int(self._n * self.problem.dimension)

        # ProblemSpec remains the single source of truth for domain projection.
        positions = np.vstack(
            [self.problem.apply_variable_types(row) for row in positions]
        )
        return positions, repairs

    def _std_smoothing(self, iteration: int) -> float:
        if not self._dynamic_std:
            return self._alpha
        t = max(1, int(iteration))
        beta_t = self._beta - self._beta * (1.0 - 1.0 / t) ** self._q
        return float(np.clip(beta_t, np.finfo(float).eps, 1.0))

    def _distribution_update(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        old_mean: np.ndarray,
        old_std: np.ndarray,
        iteration: int,
    ) -> dict[str, Any]:
        order = self._rank_indices(fitness)
        elite_indices = order[: self._elite_count]
        elite = positions[elite_indices]
        raw_mean = np.mean(elite, axis=0)
        raw_std = np.std(elite, axis=0, ddof=0)
        std_alpha = self._std_smoothing(iteration)
        mean = self._alpha * raw_mean + (1.0 - self._alpha) * old_mean
        std = std_alpha * raw_std + (1.0 - std_alpha) * old_std
        mean = np.clip(mean, self._lo, self._hi)
        std = np.maximum(std, self._min_std)
        level = float(fitness[elite_indices[-1]])
        return {
            "order": order,
            "elite_indices": elite_indices,
            "elite": elite,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "mean": mean,
            "std": std,
            "std_alpha": std_alpha,
            "level": level,
        }

    def _evaluate_generation(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        iteration: int,
        previous_generation_best: float | None,
    ) -> dict[str, Any]:
        positions, boundary_count = self._sample_population(mean, std)
        fitness = self._evaluate_population(positions)
        update = self._distribution_update(
            positions, fitness, mean, std, iteration
        )
        generation_best_index = int(update["order"][0])
        generation_best = float(fitness[generation_best_index])

        counts = self._empty_int_map()
        counts["cem.model_sampling"] = self._n
        counts["cem.elite_quantile_selection"] = self._elite_count
        counts["cem.mean_update"] = 1
        counts["cem.standard_deviation_update"] = 1
        counts["cem.boundary_handling"] = boundary_count

        contributions = self._empty_float_map()
        if previous_generation_best is not None:
            contributions["cem.model_sampling"] = max(
                0.0,
                self._signed_improvement(generation_best, previous_generation_best),
            )

        return {
            "positions": positions,
            "fitness": fitness,
            "population": np.column_stack((positions, fitness)),
            "generation_best_index": generation_best_index,
            "generation_best": generation_best,
            "counts": counts,
            "contributions": contributions,
            **update,
        }

    def _termination_check(self, state: EngineState) -> None:
        std = np.asarray(state.payload["std"], dtype=float)
        if self._std_tolerance > 0.0 and float(np.max(std)) <= self._std_tolerance:
            state.terminated = True
            state.termination_reason = "cem_std_tolerance"
            return

        if self._stagnation_window <= 0:
            return
        history = list(state.payload.get("level_history", []))
        required = self._stagnation_window + 1
        if len(history) < required:
            return
        recent = np.asarray(history[-required:], dtype=float)
        if float(np.max(recent) - np.min(recent)) <= self._level_tolerance:
            state.terminated = True
            state.termination_reason = "cem_level_stagnation"

    def initialize(self) -> EngineState:
        generation = self._evaluate_generation(
            self._initial_mean,
            self._initial_std,
            iteration=1,
            previous_generation_best=None,
        )
        best_index = generation["generation_best_index"]
        best_row = generation["population"][best_index].copy()
        counts_total = dict(generation["counts"])
        contributions_total = dict(generation["contributions"])

        state = EngineState(
            step=0,
            evaluations=self._n,
            best_position=best_row[:-1].tolist(),
            best_fitness=float(best_row[-1]),
            initialized=True,
            payload={
                "population": generation["population"],
                "elite": generation["elite"].copy(),
                "elite_indices": generation["elite_indices"].copy(),
                "best_row": best_row,
                "generation_best": generation["generation_best"],
                "mean": generation["mean"].copy(),
                "std": generation["std"].copy(),
                "raw_elite_mean": generation["raw_mean"].copy(),
                "raw_elite_std": generation["raw_std"].copy(),
                "initial_mean": self._initial_mean.copy(),
                "initial_std": self._initial_std.copy(),
                "level": generation["level"],
                "level_history": [generation["level"]],
                "cem_iterations": 1,
                "std_smoothing": generation["std_alpha"],
                "operator_counts": counts_total,
                "operator_contributions": contributions_total,
                "last_operator_counts": dict(generation["counts"]),
                "last_operator_contributions": dict(generation["contributions"]),
                "injections": 0,
            },
        )
        self._termination_check(state)
        return state

    def step(self, state: EngineState) -> EngineState:
        p = state.payload
        iteration = int(p.get("cem_iterations", 1)) + 1
        generation = self._evaluate_generation(
            np.asarray(p["mean"], dtype=float),
            np.asarray(p["std"], dtype=float),
            iteration=iteration,
            previous_generation_best=float(p["generation_best"]),
        )

        candidate_index = generation["generation_best_index"]
        candidate_row = generation["population"][candidate_index].copy()
        best_row = np.asarray(p["best_row"], dtype=float).copy()
        if self.problem.is_better(float(candidate_row[-1]), float(best_row[-1])):
            best_row = candidate_row
            state.best_position = best_row[:-1].tolist()
            state.best_fitness = float(best_row[-1])

        level_history = list(p.get("level_history", []))
        level_history.append(generation["level"])
        keep = max(32, self._stagnation_window + 1)
        if len(level_history) > keep:
            level_history = level_history[-keep:]

        state.payload = {
            **p,
            "population": generation["population"],
            "elite": generation["elite"].copy(),
            "elite_indices": generation["elite_indices"].copy(),
            "best_row": best_row,
            "generation_best": generation["generation_best"],
            "mean": generation["mean"].copy(),
            "std": generation["std"].copy(),
            "raw_elite_mean": generation["raw_mean"].copy(),
            "raw_elite_std": generation["raw_std"].copy(),
            "level": generation["level"],
            "level_history": level_history,
            "cem_iterations": iteration,
            "std_smoothing": generation["std_alpha"],
            "operator_counts": self._accumulate(
                p.get("operator_counts", self._empty_int_map()),
                generation["counts"],
            ),
            "operator_contributions": self._accumulate(
                p.get("operator_contributions", self._empty_float_map()),
                generation["contributions"],
            ),
            "last_operator_counts": dict(generation["counts"]),
            "last_operator_contributions": dict(generation["contributions"]),
        }
        state.step += 1
        state.evaluations += self._n
        self._termination_check(state)
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        p = state.payload
        pop = np.asarray(p["population"], dtype=float)
        positions = pop[:, :-1]
        fitness = pop[:, -1]
        centroid = np.mean(positions, axis=0)
        scale = float(np.linalg.norm(self._hi - self._lo)) or 1.0
        diversity = float(
            np.mean(np.linalg.norm(positions - centroid, axis=1)) / scale
        )
        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "diversity": diversity,
            "population_size": int(self._n),
            "cem_iterations": int(p.get("cem_iterations", state.step + 1)),
            "cem_elite_count": int(self._elite_count),
            "cem_elite_fraction": float(self._rho_effective),
            "cem_level": float(p["level"]),
            "cem_mean": np.asarray(p["mean"], dtype=float).tolist(),
            "cem_std": np.asarray(p["std"], dtype=float).tolist(),
            "cem_raw_elite_mean": np.asarray(
                p["raw_elite_mean"], dtype=float
            ).tolist(),
            "cem_raw_elite_std": np.asarray(
                p["raw_elite_std"], dtype=float
            ).tolist(),
            "cem_mean_smoothing": float(self._alpha),
            "cem_std_smoothing": float(p["std_smoothing"]),
            "cem_dynamic_std": bool(self._dynamic_std),
            "cem_bound_handling": self._bound_handling,
            "injections": int(p.get("injections", 0)),
            "operator_counts": dict(
                p.get("last_operator_counts", p.get("operator_counts", {}))
            ),
            "operator_contributions": dict(
                p.get(
                    "last_operator_contributions",
                    p.get("operator_contributions", {}),
                )
            ),
            "operator_counts_total": dict(p.get("operator_counts", {})),
            "operator_contributions_total": dict(
                p.get("operator_contributions", {})
            ),
            "evomapx_delta_f": "objective_consistent_positive",
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=int(state.step),
            role="best",
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
        elite_indices = set(
            np.asarray(state.payload.get("elite_indices", []), dtype=int).tolist()
        )
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="elite" if i in elite_indices else "current",
            )
            for i in range(pop.shape[0])
        ]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        if not candidates:
            return state

        p = state.payload
        pop = np.asarray(p["population"], dtype=float).copy()
        order = self._rank_indices(pop[:, -1])
        worst = order[::-1]
        accepted = min(len(candidates), pop.shape[0])
        injection_contribution = 0.0

        for j, candidate in enumerate(candidates[:accepted]):
            index = int(worst[j])
            old_fitness = float(pop[index, -1])
            position = self.problem.apply_variable_types(candidate.position)
            fitness = float(self.problem.evaluate(position))
            pop[index, :-1] = position
            pop[index, -1] = fitness
            injection_contribution += max(
                0.0, self._signed_improvement(fitness, old_fitness)
            )

        iteration = int(p.get("cem_iterations", 1))
        update = self._distribution_update(
            pop[:, :-1],
            pop[:, -1],
            np.asarray(p["mean"], dtype=float),
            np.asarray(p["std"], dtype=float),
            iteration=iteration,
        )
        best_index = int(update["order"][0])
        generation_best = float(pop[best_index, -1])
        candidate_row = pop[best_index].copy()
        best_row = np.asarray(p["best_row"], dtype=float).copy()
        if self.problem.is_better(float(candidate_row[-1]), float(best_row[-1])):
            best_row = candidate_row
            state.best_position = best_row[:-1].tolist()
            state.best_fitness = float(best_row[-1])

        counts = self._empty_int_map()
        counts["cem.candidate_injection"] = accepted
        counts["cem.elite_quantile_selection"] = self._elite_count
        counts["cem.mean_update"] = 1
        counts["cem.standard_deviation_update"] = 1
        contributions = self._empty_float_map()
        contributions["cem.candidate_injection"] = injection_contribution

        level_history = list(p.get("level_history", []))
        level_history.append(update["level"])
        state.payload = {
            **p,
            "population": pop,
            "elite": update["elite"].copy(),
            "elite_indices": update["elite_indices"].copy(),
            "best_row": best_row,
            "generation_best": generation_best,
            "mean": update["mean"].copy(),
            "std": update["std"].copy(),
            "raw_elite_mean": update["raw_mean"].copy(),
            "raw_elite_std": update["raw_std"].copy(),
            "level": update["level"],
            "level_history": level_history[-max(32, self._stagnation_window + 1) :],
            "std_smoothing": update["std_alpha"],
            "operator_counts": self._accumulate(
                p.get("operator_counts", self._empty_int_map()), counts
            ),
            "operator_contributions": self._accumulate(
                p.get("operator_contributions", self._empty_float_map()),
                contributions,
            ),
            "last_operator_counts": counts,
            "last_operator_contributions": contributions,
            "injections": int(p.get("injections", 0)) + accepted,
        }
        state.evaluations += accepted
        self._termination_check(state)
        return state

    def finalize(self, state: EngineState) -> OptimizationResult:
        p = state.payload
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
                "reference": dict(self._REFERENCE),
                "elapsed_time": float(state.elapsed_time),
                "population_size": int(self._n),
                "elite_count": int(self._elite_count),
                "elite_fraction": float(self._rho_effective),
                "smoothing": float(self._alpha),
                "dynamic_std": bool(self._dynamic_std),
                "beta": float(self._beta),
                "q": float(self._q),
                "bound_handling": self._bound_handling,
                "cem_iterations": int(p.get("cem_iterations", state.step + 1)),
                "final_mean": np.asarray(p["mean"], dtype=float).tolist(),
                "final_std": np.asarray(p["std"], dtype=float).tolist(),
                "final_level": float(p["level"]),
                "operator_counts": dict(p.get("operator_counts", {})),
                "operator_contributions": dict(
                    p.get("operator_contributions", {})
                ),
            },
        )
