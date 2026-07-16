"""Exponential-Trigonometric Optimization (ETO).

Paper-faithful implementation of Luan et al. (2024), including the
constrained-exploration search-domain update (Eqs. 1-4), the two exploration
and two exploitation phases (Eqs. 7-17), and the changeover mechanism (Eq. 18).

Two paper ambiguities require explicit interpretations:

* ``X_s`` in Eqs. (3)-(4) is interpreted as the second-ranked (suboptimal)
  member of the current population.
* Eqs. (10)-(11) make ``d2 == -d1``.  At the isolated zeros of both terms,
  their limiting ratio ``d1/d2 == -1`` is used to avoid an undefined 0/0.

The constrained-exploration recurrence is implemented exactly as printed in
Eq. (1), despite its unusually large next trigger for the paper defaults.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._ported_common import PortedPopulationEngine
from .protocol import CapabilityProfile


class ETOEngine(PortedPopulationEngine):
    """Exponential-Trigonometric Optimization engine."""

    algorithm_id = "eto"
    algorithm_name = "Exponential-Trigonometric Optimization"
    family = "math"
    _REFERENCE = {
        "doi": "10.1016/j.cma.2024.117411",
        "title": "Exponential-trigonometric optimization algorithm for solving complicated engineering problems",
        "authors": "Tran Minh Luan, Samir Khatir, Minh Thi Tran, Bernard De Baets, and Thanh Cuong-Le",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=False,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        max_iterations=1000,
        constrained_exploration_a=4.6,
        constrained_exploration_b=1.55,
    )

    _OPERATOR_LABELS = (
        "eto.constrained_search_domain_update",
        "eto.changeover_mode_selection",
        "eto.first_exploration_best_guided_update",
        "eto.first_exploitation_best_neighborhood_update",
        "eto.second_exploration_self_position_update",
        "eto.second_exploitation_intensification_update",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._a = float(self._params.get("constrained_exploration_a", 4.6))
        self._b = float(self._params.get("constrained_exploration_b", 1.55))
        if not np.isfinite(self._a) or self._a <= 0.0:
            raise ValueError("constrained_exploration_a must be finite and positive.")
        if not np.isfinite(self._b) or self._b <= 0.0:
            raise ValueError("constrained_exploration_b must be finite and positive.")
        self._last_operator_contributions = self._blank_contributions()
        self._last_operator_counts = self._blank_counts()
        self._last_mode_counts = {"exploration": 0, "exploitation": 0}

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _horizon(self) -> int:
        # EngineConfig is the package-level termination source; the paper's
        # Max_Iter parameter remains the fallback for manual stepping.
        if self.config.max_steps is not None:
            return max(1, int(self.config.max_steps))
        return max(1, int(self._params.get("max_iterations", 1000)))

    def _initial_constrained_iteration(self, horizon: int) -> int:
        # Eq. (2).
        return max(1, int(np.floor(1.0 + float(horizon) / self._b)))

    def _next_constrained_iteration(self, current: int, t: int, horizon: int) -> int:
        # Eq. (1), preserved exactly as printed in the article.
        value = np.floor(2.0 - 2.0 * float(t) * (float(horizon) - float(current) * self._a))
        if not np.isfinite(value):
            return horizon + 1
        next_iteration = int(value) + int(current)
        # A malformed/non-advancing trigger must not create an infinite loop.
        return next_iteration if next_iteration > t else t + 1

    def _phase_threshold(self, horizon: int) -> int:
        # Eq. (7).
        return max(1, int(np.floor(1.2 + float(horizon) / 2.25)))

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        horizon = self._horizon()
        return {
            "eto_lower_bound": self._lo.copy(),
            "eto_upper_bound": self._hi.copy(),
            "eto_next_constrained_iteration": self._initial_constrained_iteration(horizon),
            "eto_phase_threshold": self._phase_threshold(horizon),
            "eto_boundary_updates": 0,
            "operator_labels": [],
            "operator_counts": self._blank_counts(),
            "operator_contributions": self._blank_contributions(),
        }

    def _remaining_evaluations(self, state, population_size: int) -> int:
        if self.config.max_evaluations is None:
            return int(population_size)
        remaining = max(0, int(self.config.max_evaluations) - int(state.evaluations))
        return min(int(population_size), remaining)

    def _best_so_far(self, state, pop: np.ndarray) -> np.ndarray:
        if state.best_position is not None:
            best = np.asarray(state.best_position, dtype=float)
            if best.shape == (self.problem.dimension,) and np.all(np.isfinite(best)):
                return best.copy()
        return pop[self._best_index(pop[:, -1]), :-1].copy()

    def _suboptimal_position(self, pop: np.ndarray) -> np.ndarray:
        order = self._order(pop[:, -1])
        index = int(order[1]) if order.size > 1 else int(order[0])
        return pop[index, :-1].copy()

    @staticmethod
    def _d_terms(t: int, horizon: int) -> tuple[float, float, float]:
        # Eqs. (10)-(11).  Their ratio is -1 except at simultaneous zeros.
        oscillation = np.cos(0.5 * float(horizon) * (1.0 - float(t) / float(horizon)))
        d1 = 0.1 * np.exp(-0.01 * float(t)) * oscillation
        d2 = -d1
        ratio = -1.0 if abs(d2) <= np.finfo(float).tiny else float(d1 / d2)
        return float(d1), float(d2), ratio

    def _update_search_domain(
        self,
        best: np.ndarray,
        suboptimal: np.ndarray,
        t: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Eqs. (3)-(4), evaluated component-wise as in Algorithm 2's i,j loop.
        r1 = np.random.random(self.problem.dimension)
        r2 = np.random.random(self.problem.dimension)
        radius = r1 * (1.0 - float(t) / float(horizon)) * np.abs(r2 * best - suboptimal)
        prospective_upper = best + radius
        prospective_lower = best - radius
        lower = np.maximum(self._lo, np.minimum(prospective_lower, prospective_upper))
        upper = np.minimum(self._hi, np.maximum(prospective_lower, prospective_upper))
        invalid = lower > upper
        if np.any(invalid):
            midpoint = np.clip(best, self._lo, self._hi)
            lower[invalid] = midpoint[invalid]
            upper[invalid] = midpoint[invalid]
        return lower, upper

    def _changeover(self, t: int, horizon: int, ratio: float, shape: tuple[int, int]) -> np.ndarray:
        # Eq. (18).  Iterations are one-based, avoiding the t=0 singularity.
        base = np.sqrt(float(t) / float(horizon))
        exponent = np.tan(ratio)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = float(np.power(base, exponent))
        if not np.isfinite(scale):
            scale = float(np.finfo(float).max)
        return 0.01 * np.random.random(shape) * scale

    def _candidate_generation(
        self,
        parents: np.ndarray,
        best: np.ndarray,
        t: int,
        horizon: int,
        threshold: int,
        ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
        """Generate one ETO population update from Eqs. (8), (12), (14), (16)."""
        n, dim = parents.shape
        cm = self._changeover(t, horizon, ratio, (n, dim))
        exploration = cm > 1.0
        exploitation = ~exploration
        candidates = np.empty_like(parents)
        label_matrix = np.empty((n, dim), dtype=object)

        counts = self._blank_counts()
        counts["eto.changeover_mode_selection"] = int(n * dim)
        mode_counts = {
            "exploration": int(np.count_nonzero(exploration)),
            "exploitation": int(np.count_nonzero(exploitation)),
        }

        if t <= threshold:
            # Eq. (9) and Eq. (8): first exploration phase.
            alpha1 = (
                3.0
                * np.random.random((n, dim))
                * (float(t) / float(horizon) - 0.85)
                * np.exp(ratio - 1.0)
            )
            q1 = np.random.random((n, dim))
            r = np.random.random((n, dim))
            displacement = r * alpha1 * np.abs(best[None, :] - parents)
            first_exploration = np.where(q1 <= 0.5, best[None, :] + displacement, best[None, :] - displacement)

            # Eq. (15) and Eq. (14): first exploitation phase.
            alpha3 = (
                3.0
                * np.random.random((n, dim))
                * (float(t) / float(horizon) - 0.85)
                * np.exp(abs(ratio) - 1.3)
            )
            q3 = np.random.random((n, dim))
            q4 = np.random.random((n, dim))
            r = np.random.random((n, dim))
            displacement = q4 * alpha3 * np.abs(r * best[None, :] - parents)
            first_exploitation = np.where(q3 <= 0.5, best[None, :] + displacement, best[None, :] - displacement)

            candidates[:] = np.where(exploration, first_exploration, first_exploitation)
            label_matrix[exploration] = "eto.first_exploration_best_guided_update"
            label_matrix[exploitation] = "eto.first_exploitation_best_neighborhood_update"
            counts["eto.first_exploration_best_guided_update"] = mode_counts["exploration"]
            counts["eto.first_exploitation_best_neighborhood_update"] = mode_counts["exploitation"]
        else:
            # Eq. (13), used in both second-stage update equations.
            alpha2_random = np.random.random((n, dim))
            alpha2 = alpha2_random * np.exp(
                np.tanh(1.5 * (-float(t) / float(horizon) - 0.75) - np.random.random((n, dim)))
            )

            # Eq. (12): second exploration phase.
            q2 = np.random.random((n, dim))
            r = np.random.random((n, dim))
            displacement = 3.0 * np.abs(r * alpha2 * best[None, :] - parents)
            second_exploration = np.where(q2 <= 0.5, parents + displacement, parents - displacement)

            # Eqs. (16)-(17): second exploitation phase.
            c = float(np.exp(np.tan(abs(ratio))))
            r = np.random.random((n, dim))
            second_exploitation = parents + c * np.abs(r * alpha2 * best[None, :] - parents)

            candidates[:] = np.where(exploration, second_exploration, second_exploitation)
            label_matrix[exploration] = "eto.second_exploration_self_position_update"
            label_matrix[exploitation] = "eto.second_exploitation_intensification_update"
            counts["eto.second_exploration_self_position_update"] = mode_counts["exploration"]
            counts["eto.second_exploitation_intensification_update"] = mode_counts["exploitation"]

        return candidates, label_matrix, counts, mode_counts

    def _objective_gain(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        if self.problem.objective == "min":
            return np.maximum(0.0, previous - current)
        return np.maximum(0.0, current - previous)

    def _step_impl(self, state, pop: np.ndarray):
        n = int(pop.shape[0])
        evaluations = self._remaining_evaluations(state, n)
        if evaluations <= 0:
            self._last_operator_contributions = self._blank_contributions()
            self._last_operator_counts = self._blank_counts()
            self._last_mode_counts = {"exploration": 0, "exploitation": 0}
            return pop, 0, {
                "operator_labels": [],
                "operator_counts": self._blank_counts(),
                "operator_contributions": self._blank_contributions(),
            }

        horizon = self._horizon()
        t = min(horizon, max(1, int(state.step) + 1))
        threshold = int(state.payload.get("eto_phase_threshold", self._phase_threshold(horizon)))
        lower = np.asarray(state.payload.get("eto_lower_bound", self._lo), dtype=float).copy()
        upper = np.asarray(state.payload.get("eto_upper_bound", self._hi), dtype=float).copy()
        next_ce = int(
            state.payload.get(
                "eto_next_constrained_iteration",
                self._initial_constrained_iteration(horizon),
            )
        )
        boundary_updates = int(state.payload.get("eto_boundary_updates", 0))

        best = self._best_so_far(state, pop)
        _, _, ratio = self._d_terms(t, horizon)
        boundary_updated = False
        if t == next_ce:
            lower, upper = self._update_search_domain(
                best,
                self._suboptimal_position(pop),
                t,
                horizon,
            )
            next_ce = self._next_constrained_iteration(next_ce, t, horizon)
            boundary_updates += 1
            boundary_updated = True

        parents = pop[:evaluations, :-1].copy()
        old_fitness = pop[:evaluations, -1].copy()
        candidates, label_matrix, counts, mode_counts = self._candidate_generation(
            parents,
            best,
            t,
            horizon,
            threshold,
            ratio,
        )
        raw_candidates = candidates.copy()
        candidates = np.clip(candidates, lower, upper)
        for i in range(candidates.shape[0]):
            candidates[i] = self.problem.apply_variable_types(candidates[i])
        repaired_coordinates = int(np.count_nonzero(~np.isclose(raw_candidates, candidates, rtol=0.0, atol=0.0)))

        new_fitness = self._evaluate_population(candidates)
        next_population = pop.copy()
        next_population[:evaluations, :-1] = candidates
        next_population[:evaluations, -1] = new_fitness

        contributions = self._blank_contributions()
        gains = self._objective_gain(old_fitness, new_fitness)
        for i, gain in enumerate(gains):
            if gain <= 0.0:
                continue
            labels, label_counts = np.unique(label_matrix[i], return_counts=True)
            for label, count in zip(labels.tolist(), label_counts.tolist()):
                contributions[str(label)] += float(gain) * float(count) / float(self.problem.dimension)

        if boundary_updated:
            counts["eto.constrained_search_domain_update"] = 1

        # One lineage label per candidate is required by the package observer;
        # mixed-coordinate updates use the dominant native operator.
        operator_labels: list[str] = []
        lineage: list[dict[str, Any]] = []
        for i in range(evaluations):
            labels, label_counts = np.unique(label_matrix[i], return_counts=True)
            dominant = str(labels[int(np.argmax(label_counts))])
            operator_labels.append(dominant)
            lineage.append(
                {
                    "id": f"eto:{state.step + 1}:{i}",
                    "parent_ids": [f"eto:{state.step}:{i}"],
                    "parent_index": i,
                    "operator": dominant,
                    "lineage_delta": float(gains[i]),
                }
            )

        self._last_operator_contributions = {
            label: float(max(0.0, value)) for label, value in contributions.items()
        }
        self._last_operator_counts = {label: int(value) for label, value in counts.items()}
        self._last_mode_counts = dict(mode_counts)

        return next_population, evaluations, {
            "eto_lower_bound": lower,
            "eto_upper_bound": upper,
            "eto_next_constrained_iteration": int(next_ce),
            "eto_phase_threshold": int(threshold),
            "eto_boundary_updates": int(boundary_updates),
            "eto_iteration": int(t),
            "eto_d1_d2_ratio": float(ratio),
            "eto_repaired_coordinates": int(repaired_coordinates),
            "operator_labels": operator_labels,
            "operator_counts": dict(self._last_operator_counts),
            "operator_contributions": dict(self._last_operator_contributions),
            "lineage": lineage,
            "native_evomapx_operator_labels": True,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["operator_labels"] = list(state.payload.get("operator_labels", []))
        obs["lineage"] = list(state.payload.get("lineage", []))
        obs["evomapx_delta_f"] = "objective_consistent_positive"
        obs["evomapx_fidelity"] = "native"
        obs["native_evomapx_operator_labels"] = True
        obs["mode_counts"] = dict(self._last_mode_counts)
        obs["phase"] = "first" if int(state.payload.get("eto_iteration", 1)) <= int(
            state.payload.get("eto_phase_threshold", self._phase_threshold(self._horizon()))
        ) else "second"
        obs["constrained_search_domain"] = {
            "lower": np.asarray(state.payload.get("eto_lower_bound", self._lo), dtype=float).tolist(),
            "upper": np.asarray(state.payload.get("eto_upper_bound", self._hi), dtype=float).tolist(),
            "updates": int(state.payload.get("eto_boundary_updates", 0)),
            "next_iteration": int(
                state.payload.get(
                    "eto_next_constrained_iteration",
                    self._initial_constrained_iteration(self._horizon()),
                )
            ),
        }
        return obs
