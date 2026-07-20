"""pyMetaheuristic src — GRASP engine.

Feo and Resende (1995) define GRASP as repeated independent iterations with
(1) adaptive greedy randomized construction from a restricted candidate list
and (2) local search, retaining the best solution over all iterations.

The cited paper is a problem-specific combinatorial framework rather than a
black-box continuous optimizer.  This engine therefore implements the paper's
control structure natively and provides a documented continuous-box bridge:
a solution is constructed one variable assignment at a time; candidate values
are rescored after every assignment; an RCL applies the paper's value and/or
cardinality restriction; and a bounded sampled-neighborhood local search runs
to the first neighborhood with no improving candidate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile, EngineState
from ._restart_common import RestartLocalSearchEngine


class GRASPEngine(RestartLocalSearchEngine):
    """GRASP with native construction, RCL selection, and local search phases."""

    algorithm_id = "grasp"
    algorithm_name = "Greedy Randomized Adaptive Search Procedures"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1007/BF01096763",
        "title": "Greedy Randomized Adaptive Search Procedures",
        "authors": "Thomas A. Feo, Mauricio G. C. Resende",
        "year": 1995,
    }
    capabilities = CapabilityProfile(
        has_population=False,
        has_archive=False,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=False,
        supports_snapshot_fit=False,
    )

    _OPERATOR_LABELS = (
        "grasp.construction",
        "grasp.rcl_selection",
        "grasp.local_search",
        "grasp.incumbent_update",
    )

    _DEFAULTS = {
        # Continuous-box construction bridge.  The paper supplies no universal
        # numerical defaults because its greedy function is problem-specific.
        "construction_pool_size": 12,
        "construction_order": "random",  # random | sequential
        "rcl_alpha": 0.80,               # paper convention: 0=random, 1=greedy
        "rcl_size": None,                # optional cardinality restriction
        "rcl_fraction": None,            # optional cardinality fraction
        # Sampled continuous neighborhood used by the generic package bridge.
        "local_search_steps": 12,
        "neighborhood_size": None,
        "step_size": 0.12,
        "min_step_size": 1.0e-8,
        # Retained only for compatibility with RestartLocalSearchEngine's
        # validation contract; GRASP itself has no temperature/step schedule.
        "contraction": 1.0,
        "expansion": 1.0,
        "restart_stagnation_steps": 1,
    }

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_grasp_parameters()

    def _validate_grasp_parameters(self) -> None:
        pool_size = int(self._params.get("construction_pool_size", 12))
        if pool_size < 1:
            raise ValueError("grasp construction_pool_size must be >= 1.")

        alpha = float(self._params.get("rcl_alpha", 0.80))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("grasp rcl_alpha must be in [0, 1].")

        rcl_size = self._params.get("rcl_size", None)
        if rcl_size is not None and int(rcl_size) < 1:
            raise ValueError("grasp rcl_size must be >= 1 when supplied.")

        rcl_fraction = self._params.get("rcl_fraction", None)
        if rcl_fraction is not None and not 0.0 < float(rcl_fraction) <= 1.0:
            raise ValueError("grasp rcl_fraction must be in (0, 1] when supplied.")

        order = str(self._params.get("construction_order", "random")).strip().lower()
        if order not in {"random", "sequential"}:
            raise ValueError("grasp construction_order must be 'random' or 'sequential'.")

    @staticmethod
    def _zero_counts() -> dict[str, int]:
        return {label: 0 for label in GRASPEngine._OPERATOR_LABELS}

    @staticmethod
    def _zero_contributions() -> dict[str, float]:
        return {label: 0.0 for label in GRASPEngine._OPERATOR_LABELS}

    def _signed_improvement(self, old_fitness: float, new_fitness: float) -> float:
        if self.problem.objective == "min":
            return float(old_fitness) - float(new_fitness)
        return float(new_fitness) - float(old_fitness)

    def _coordinate_order(self) -> np.ndarray:
        order = np.arange(int(self.problem.dimension), dtype=int)
        if str(self._params.get("construction_order", "random")).lower() == "random":
            self._rng.shuffle(order)
        return order

    def _rcl_indices(self, fitness: np.ndarray) -> np.ndarray:
        """Build the RCL with the paper's greediness convention.

        With ``alpha=0`` every candidate passes the value restriction; with
        ``alpha=1`` only candidates tied with the greedy best pass.  Optional
        cardinality restrictions are intersected with the value-restricted RCL,
        as allowed by Feo and Resende.
        """
        fitness = np.asarray(fitness, dtype=float)
        n = int(fitness.size)
        if n <= 0:
            return np.empty(0, dtype=int)

        alpha = float(self._params.get("rcl_alpha", 0.80))
        if self.problem.objective == "min":
            best = float(np.min(fitness))
            worst = float(np.max(fitness))
            threshold = worst - alpha * (worst - best)
            value_mask = fitness <= threshold + 1.0e-15 * (1.0 + abs(threshold))
            order = np.argsort(fitness, kind="stable")
        else:
            best = float(np.max(fitness))
            worst = float(np.min(fitness))
            threshold = worst + alpha * (best - worst)
            value_mask = fitness >= threshold - 1.0e-15 * (1.0 + abs(threshold))
            order = np.argsort(fitness, kind="stable")[::-1]

        allowed = set(np.flatnonzero(value_mask).tolist())

        cardinality = n
        rcl_size = self._params.get("rcl_size", None)
        if rcl_size is not None:
            cardinality = min(cardinality, int(rcl_size))
        rcl_fraction = self._params.get("rcl_fraction", None)
        if rcl_fraction is not None:
            cardinality = min(cardinality, max(1, int(np.ceil(float(rcl_fraction) * n))))
        if cardinality < n:
            allowed.intersection_update(int(idx) for idx in order[:cardinality])

        if not allowed:
            return np.asarray([int(order[0])], dtype=int)
        # Stable candidate order makes seeded runs reproducible across Python versions.
        return np.asarray(sorted(allowed), dtype=int)

    def _construct(
        self,
        *,
        max_evaluations: int | None,
        iteration: int,
    ) -> tuple[np.ndarray, float, int, dict[str, Any]]:
        """Construct one feasible vector by adaptive randomized greedy assignment."""
        if max_evaluations is not None:
            max_evaluations = max(0, int(max_evaluations))

        counts = self._zero_counts()
        contributions = self._zero_contributions()
        lineage: list[dict[str, Any]] = []
        evals = 0

        current = self._random_position()
        if max_evaluations is not None and max_evaluations <= 0:
            return current, self.problem.worst_fitness(), 0, {
                "operator_counts": counts,
                "operator_contributions": contributions,
                "lineage": lineage,
                "construction_stages": 0,
                "construction_candidates": 0,
                "mean_rcl_size": 0.0,
            }

        current_fit = float(self.problem.evaluate(current))
        evals += 1
        parent_id = f"grasp:{iteration}:construction:seed"
        lineage.append(
            {
                "id": parent_id,
                "parent_ids": [],
                "operator": "grasp.construction",
                "fitness": float(current_fit),
                "role": "continuous_completion_seed",
            }
        )
        counts["grasp.construction"] += 1

        pool_size = int(self._params.get("construction_pool_size", 12))
        total_candidates = 0
        rcl_sizes: list[int] = []
        completed_stages = 0

        for stage, coordinate in enumerate(self._coordinate_order()):
            remaining = None if max_evaluations is None else max_evaluations - evals
            if remaining is not None and remaining <= 0:
                break
            n_candidates = pool_size if remaining is None else min(pool_size, remaining)
            if n_candidates <= 0:
                break

            values = self._rng.uniform(
                float(self._lo[coordinate]),
                float(self._hi[coordinate]),
                size=int(n_candidates),
            )
            candidates = np.repeat(current[None, :], int(n_candidates), axis=0)
            candidates[:, int(coordinate)] = values
            candidate_fit = np.asarray(
                [float(self.problem.evaluate(row)) for row in candidates],
                dtype=float,
            )
            evals += int(n_candidates)
            total_candidates += int(n_candidates)
            counts["grasp.construction"] += int(n_candidates)

            rcl = self._rcl_indices(candidate_fit)
            selected = int(self._rng.choice(rcl))
            previous_fit = float(current_fit)
            previous_id = parent_id
            current = candidates[selected].copy()
            current_fit = float(candidate_fit[selected])
            gain = self._signed_improvement(previous_fit, current_fit)

            counts["grasp.rcl_selection"] += 1
            contributions["grasp.rcl_selection"] += float(gain)
            rcl_sizes.append(int(rcl.size))
            completed_stages += 1
            parent_id = f"grasp:{iteration}:construction:{stage}"
            lineage.append(
                {
                    "id": parent_id,
                    "parent_ids": [previous_id],
                    "operator": "grasp.rcl_selection",
                    "coordinate": int(coordinate),
                    "candidate_count": int(n_candidates),
                    "rcl_size": int(rcl.size),
                    "fitness_before": previous_fit,
                    "fitness_after": float(current_fit),
                    "signed_improvement": float(gain),
                }
            )

        return current, float(current_fit), int(evals), {
            "operator_counts": counts,
            "operator_contributions": contributions,
            "lineage": lineage,
            "construction_stages": int(completed_stages),
            "construction_candidates": int(total_candidates),
            "mean_rcl_size": float(np.mean(rcl_sizes)) if rcl_sizes else 0.0,
            "last_lineage_id": parent_id,
        }

    def _run_local_search(
        self,
        start: np.ndarray,
        *,
        start_fit: float,
        max_evaluations: int | None,
        iteration: int,
        parent_id: str,
    ) -> tuple[np.ndarray, float, int, dict[str, Any]]:
        """Apply sampled-neighborhood improvement until no better neighbor exists."""
        if max_evaluations is not None:
            max_evaluations = max(0, int(max_evaluations))

        current = self._clip(start)
        current_fit = float(start_fit)
        evals = 0
        counts = self._zero_counts()
        contributions = self._zero_contributions()
        lineage: list[dict[str, Any]] = []
        accepted_moves = 0
        neighborhoods = 0
        repairs = 0

        max_steps = max(0, int(self._params.get("local_search_steps", 12)))
        nbh_param = self._params.get("neighborhood_size", None)
        neighborhood_size = int(nbh_param) if nbh_param is not None else max(4, 2 * int(self.problem.dimension))
        radius = float(self._params.get("step_size", 0.12))

        for sweep in range(max_steps):
            remaining = None if max_evaluations is None else max_evaluations - evals
            if remaining is not None and remaining <= 0:
                break
            n_trials = neighborhood_size if remaining is None else min(neighborhood_size, remaining)
            if n_trials <= 0:
                break

            raw_trials = current + self._rng.normal(
                0.0,
                radius,
                size=(int(n_trials), int(self.problem.dimension)),
            ) * self._span
            repairs += int(np.count_nonzero((raw_trials < self._lo) | (raw_trials > self._hi)))
            trials = np.clip(raw_trials, self._lo, self._hi)
            trial_fit = np.asarray(
                [float(self.problem.evaluate(row)) for row in trials],
                dtype=float,
            )
            evals += int(n_trials)
            neighborhoods += 1
            counts["grasp.local_search"] += int(n_trials)

            best_idx = self._best_index(trial_fit)
            best_fit = float(trial_fit[best_idx])
            if not self._is_better(best_fit, current_fit):
                # Figure 3 terminates local search when no better solution is
                # found in the chosen neighborhood.
                break

            previous_fit = float(current_fit)
            previous_id = parent_id
            current = trials[best_idx].copy()
            current_fit = best_fit
            gain = self._signed_improvement(previous_fit, current_fit)
            contributions["grasp.local_search"] += float(gain)
            accepted_moves += 1
            parent_id = f"grasp:{iteration}:local:{sweep}"
            lineage.append(
                {
                    "id": parent_id,
                    "parent_ids": [previous_id],
                    "operator": "grasp.local_search",
                    "neighborhood_size": int(n_trials),
                    "fitness_before": previous_fit,
                    "fitness_after": float(current_fit),
                    "signed_improvement": float(gain),
                }
            )

        return current, float(current_fit), int(evals), {
            "operator_counts": counts,
            "operator_contributions": contributions,
            "lineage": lineage,
            "local_search_moves": int(accepted_moves),
            "local_search_neighborhoods": int(neighborhoods),
            "bound_repairs": int(repairs),
            "last_lineage_id": parent_id,
        }

    # Keep inherited candidate injection/restart on the same neighborhood rule.
    def _local_search(
        self,
        start: np.ndarray,
        start_fit: float | None = None,
        step_size: float | None = None,
        max_steps: int | None = None,
        neighborhood_size: int | None = None,
        max_evaluations: int | None = None,
    ) -> tuple[np.ndarray, float, int, float]:
        original: dict[str, Any] = {}
        for key, value in (
            ("step_size", step_size),
            ("local_search_steps", max_steps),
            ("neighborhood_size", neighborhood_size),
        ):
            if value is not None:
                original[key] = self._params.get(key)
                self._params[key] = value
        try:
            clipped = self._clip(start)
            evals = 0
            if start_fit is None:
                if max_evaluations is not None and int(max_evaluations) <= 0:
                    return clipped, self.problem.worst_fitness(), 0, float(self._params.get("step_size", 0.12))
                start_fit = float(self.problem.evaluate(clipped))
                evals = 1
            remaining = None if max_evaluations is None else max(0, int(max_evaluations) - evals)
            pos, fit, extra, _ = self._run_local_search(
                clipped,
                start_fit=float(start_fit),
                max_evaluations=remaining,
                iteration=-1,
                parent_id="grasp:external:seed",
            )
            return pos, float(fit), int(evals + extra), float(self._params.get("step_size", 0.12))
        finally:
            for key, value in original.items():
                self._params[key] = value

    def initialize(self) -> EngineState:
        """Initialize an incumbent without executing a GRASP iteration."""
        position = self._random_position()
        fitness = float(self.problem.evaluate(position))
        counts = self._zero_counts()
        contributions = self._zero_contributions()
        return EngineState(
            step=0,
            evaluations=1,
            best_position=position.tolist(),
            best_fitness=float(fitness),
            initialized=True,
            payload={
                "current": position,
                "current_fit": float(fitness),
                "delta": float(self._params.get("step_size", 0.12)),
                "stagnation": 0,
                "restarts": 0,
                "grasp_iterations": 0,
                "last_accepted": True,
                "lineage": [
                    {
                        "id": "grasp:0:incumbent",
                        "parent_ids": [],
                        "operator": "grasp.incumbent_update",
                        "fitness": float(fitness),
                        "role": "initial_incumbent",
                    }
                ],
                "operator_counts": counts,
                "operator_contributions": contributions,
                "construction_stages": 0,
                "construction_candidates": 0,
                "mean_rcl_size": 0.0,
                "local_search_moves": 0,
                "local_search_neighborhoods": 0,
                "bound_repairs": 0,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state

        iteration = int(state.step) + 1
        candidate, candidate_fit, construction_evals, construction = self._construct(
            max_evaluations=remaining,
            iteration=iteration,
        )
        if construction_evals <= 0:
            return state

        remaining_after_construction = self._remaining_evaluations(state, used=construction_evals)
        candidate, candidate_fit, local_evals, local = self._run_local_search(
            candidate,
            start_fit=float(candidate_fit),
            max_evaluations=remaining_after_construction,
            iteration=iteration,
            parent_id=str(construction.get("last_lineage_id", f"grasp:{iteration}:construction:seed")),
        )

        counts = self._zero_counts()
        contributions = self._zero_contributions()
        for label in self._OPERATOR_LABELS:
            counts[label] = int(construction["operator_counts"].get(label, 0)) + int(local["operator_counts"].get(label, 0))
            contributions[label] = float(construction["operator_contributions"].get(label, 0.0)) + float(local["operator_contributions"].get(label, 0.0))

        old_best = float(state.best_fitness)
        improved_incumbent = self._is_better(float(candidate_fit), old_best)
        incumbent_gain = self._signed_improvement(old_best, float(candidate_fit)) if improved_incumbent else 0.0
        counts["grasp.incumbent_update"] = 1
        contributions["grasp.incumbent_update"] = float(incumbent_gain)

        lineage = list(construction.get("lineage", [])) + list(local.get("lineage", []))
        last_candidate_id = str(local.get("last_lineage_id", construction.get("last_lineage_id", "")))
        lineage.append(
            {
                "id": f"grasp:{iteration}:incumbent",
                "parent_ids": [f"grasp:{iteration - 1}:incumbent", last_candidate_id],
                "operator": "grasp.incumbent_update",
                "accepted": bool(improved_incumbent),
                "fitness_before": old_best,
                "candidate_fitness": float(candidate_fit),
                "fitness_after": float(candidate_fit if improved_incumbent else old_best),
                "signed_improvement": float(incumbent_gain),
            }
        )

        if improved_incumbent:
            state.best_position = candidate.tolist()
            state.best_fitness = float(candidate_fit)
            stagnation = 0
        else:
            stagnation = int(state.payload.get("stagnation", 0)) + 1

        state.payload.update(
            current=candidate,
            current_fit=float(candidate_fit),
            delta=float(self._params.get("step_size", 0.12)),
            stagnation=int(stagnation),
            restarts=int(state.payload.get("restarts", 0)) + 1,
            grasp_iterations=int(state.payload.get("grasp_iterations", 0)) + 1,
            last_accepted=bool(improved_incumbent),
            lineage=lineage,
            operator_counts=counts,
            operator_contributions=contributions,
            construction_stages=int(construction.get("construction_stages", 0)),
            construction_candidates=int(construction.get("construction_candidates", 0)),
            mean_rcl_size=float(construction.get("mean_rcl_size", 0.0)),
            local_search_moves=int(local.get("local_search_moves", 0)),
            local_search_neighborhoods=int(local.get("local_search_neighborhoods", 0)),
            bound_repairs=int(local.get("bound_repairs", 0)),
        )
        state.step += 1
        state.evaluations += int(construction_evals + local_evals)
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        observation = super().observe(state)
        counts = dict(state.payload.get("operator_counts", self._zero_counts()))
        contributions = dict(state.payload.get("operator_contributions", self._zero_contributions()))
        observation.update(
            grasp_iterations=int(state.payload.get("grasp_iterations", state.step)),
            construction_stages=int(state.payload.get("construction_stages", 0)),
            construction_candidates=int(state.payload.get("construction_candidates", 0)),
            mean_rcl_size=float(state.payload.get("mean_rcl_size", 0.0)),
            local_search_moves=int(state.payload.get("local_search_moves", 0)),
            local_search_neighborhoods=int(state.payload.get("local_search_neighborhoods", 0)),
            bound_repairs=int(state.payload.get("bound_repairs", 0)),
            operator_counts=counts,
            operator_contributions=contributions,
            evomapx_operator_labels=list(self._OPERATOR_LABELS),
            evomapx_delta_f="signed",
            evomapx_fidelity="native_continuous_adaptation",
            native_evomapx_operator_labels=True,
        )
        return observation
