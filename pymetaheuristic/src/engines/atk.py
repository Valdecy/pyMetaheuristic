"""Aitken Optimizer (ATK).

Paper-faithful NumPy implementation of the population-based optimizer proposed
by Zhao, Fu, Zhang, and Huang.  The engine implements the two paper phases:

1. Aitken acceleration method search mechanism (AAMSM), Eqs. (7)-(10).
2. Random weighted exponential operator (RWEO), Eqs. (11)-(14).

The paper contains two notational ambiguities in Eq. (13).  This implementation
uses ``X_D == X_d`` and samples ``S_d`` from the combined current/historical
matrix, which is the only interpretation that makes both the stated indices and
the historical matrix operational.  These assumptions are isolated in
``_aitken_refinement`` and do not alter the published Eqs. (7)-(12).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._ported_common import PortedPopulationEngine
from .protocol import CapabilityProfile


class ATKEngine(PortedPopulationEngine):
    """Aitken Optimizer (ATK), a mathematics-driven population optimizer."""

    algorithm_id = "atk"
    algorithm_name = "Aitken Optimizer"
    family = "math"
    _REFERENCE = {
        "doi": "10.1007/s11227-024-06709-2",
        "title": "Aitken optimizer: an efficient optimization algorithm based on the Aitken acceleration method",
        "authors": "Yongpeng Zhao, Shengwei Fu, Langlang Zhang, and Haisong Huang",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        epsilon=float(np.finfo(float).eps),
        maximum_acceleration_scale=1.0e12,
        exponent_clip=50.0,
    )

    _OPERATOR_LABELS = (
        "atk.aitken_acceleration_search",
        "atk.random_weighted_exponential_search",
        "atk.aitken_refinement",
        "atk.greedy_selection",
        "atk.reflective_bound_repair",
        "atk.historical_best_update",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._epsilon = float(self._params.get("epsilon", np.finfo(float).eps))
        self._maximum_acceleration_scale = float(
            self._params.get("maximum_acceleration_scale", 1.0e12)
        )
        self._exponent_clip = float(self._params.get("exponent_clip", 50.0))
        self._validate_parameters()
        self._last_operator_contributions = self._blank_contributions()
        self._last_operator_counts = self._blank_counts()
        self._last_phase_acceptance = {"aamsm": 0, "rweo": 0, "refinement": 0}

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("ATK requires a population_size of at least 2.")
        if not np.isfinite(self._epsilon) or self._epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number.")
        if (
            not np.isfinite(self._maximum_acceleration_scale)
            or self._maximum_acceleration_scale <= 0.0
        ):
            raise ValueError("maximum_acceleration_scale must be finite and positive.")
        if not np.isfinite(self._exponent_clip) or self._exponent_clip <= 0.0:
            raise ValueError("exponent_clip must be finite and positive.")

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        return {
            "historical_best_positions": pop[:, :-1].copy(),
            "historical_best_fitness": pop[:, -1].copy(),
        }

    def _remaining_evaluations(self, state) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(0, int(self.config.max_evaluations) - int(state.evaluations))

    def _horizon(self) -> int:
        return max(1, int(self.config.max_steps or 500))

    def _progress_iteration(self, state) -> tuple[int, int]:
        # The paper indexes t from 1 to T.
        return max(1, int(state.step) + 1), self._horizon()

    def _reflect_bounds(
        self,
        candidate: np.ndarray,
        parent: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Apply the symmetric reflection model of Eqs. (15)-(16).

        A modulo reflection is used so candidates that cross a bound by more
        than one interval are still mapped into the box without clipping away
        their reflected displacement.
        """
        raw = np.asarray(candidate, dtype=float).reshape(self.problem.dimension)
        parent = np.asarray(parent, dtype=float).reshape(self.problem.dimension)
        nonfinite = ~np.isfinite(raw)
        if np.any(nonfinite):
            raw = raw.copy()
            raw[nonfinite] = parent[nonfinite]

        span = self._hi - self._lo
        fixed = span <= 0.0
        safe_span = np.where(fixed, 1.0, span)
        period = 2.0 * safe_span
        folded = np.mod(raw - self._lo, period)
        reflected = np.where(
            folded <= safe_span,
            self._lo + folded,
            self._hi - (folded - safe_span),
        )
        reflected[fixed] = self._lo[fixed]
        repaired = bool(np.any(nonfinite) or np.any(raw < self._lo) or np.any(raw > self._hi))
        return self.problem.apply_variable_types(reflected), repaired

    def _aitken_position(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Return the AAMSM accelerated point from Eq. (7)."""
        r1 = float(np.random.random())
        r2 = float(np.random.random())
        p3 = r1 * p2 - (0.5 - r2) * p1
        denominator = p3 - 2.0 * p2 + p1
        numerator = np.square(p1 - p2)
        safe = np.abs(denominator) > self._epsilon
        accelerated = np.asarray(p2, dtype=float).copy()
        accelerated[safe] = p2[safe] - numerator[safe] / denominator[safe]
        accelerated[~np.isfinite(accelerated)] = p2[~np.isfinite(accelerated)]
        return accelerated

    def _aamsm_candidate(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        i: int,
        t: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the phase-1 candidate using Eqs. (7)-(10)."""
        best_idx = self._best_index(fitness)
        x_i = positions[i].copy()
        x_best = positions[best_idx].copy()
        f_i = float(fitness[i])
        f_best = float(fitness[best_idx])

        # Eq. (8): p1 is the better of the current and best positions.
        if self._is_better(f_i, f_best):
            p1, p2 = x_i, x_best
        else:
            p1, p2 = x_best, x_i
        p = self._aitken_position(p1, p2)

        mean_position = np.mean(positions, axis=0)
        mean_fitness = float(np.mean(fitness))
        r3 = float(np.random.random())
        if r3 <= 0.5:
            denominator = mean_fitness + self._epsilon
            if abs(denominator) <= self._epsilon:
                denominator = self._epsilon if denominator >= 0.0 else -self._epsilon
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                scale = float(np.square(abs(f_best / denominator)))
            if not np.isfinite(scale):
                scale = self._maximum_acceleration_scale
            scale = min(scale, self._maximum_acceleration_scale)
            r1 = float(np.random.random())
            candidate = x_best + p * scale + r1 * (x_best - mean_position)
        else:
            r2 = float(np.random.random())
            r3_move = float(np.random.random())
            a, b = np.random.randint(0, positions.shape[0], size=2)
            decay = float(np.exp(-(float(t) ** 2) / float(horizon)))
            candidate = (
                x_best
                + p * (1.0 - 2.0 * r2) * decay
                + r3_move * (positions[a] - positions[b])
            )
        return candidate, p

    def _rweo_candidate(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        p: np.ndarray,
        t: int,
        horizon: int,
    ) -> np.ndarray:
        """Generate the phase-2 RWEO candidate from Eqs. (11)-(12)."""
        best_idx = self._best_index(fitness)
        x_best = positions[best_idx].copy()
        mean_position = np.mean(positions, axis=0)

        omega = 2.0 * np.random.random(self.problem.dimension) * np.exp(
            -5.0 * float(t) / float(horizon)
        )
        lam = float(np.random.choice(np.asarray([-1.0, 0.0, 1.0])))
        mu = np.random.normal(0.0, 1.0, self.problem.dimension)
        r4 = float(np.random.random())

        candidate = np.empty(self.problem.dimension, dtype=float)
        low_omega = omega < 1.0
        candidate[low_omega] = x_best[low_omega] + lam * (
            omega[low_omega] * np.abs(x_best[low_omega] - p[low_omega])
            + mu[low_omega]
        )
        candidate[~low_omega] = p[~low_omega] - mean_position[~low_omega] + lam * (
            r4 * x_best[~low_omega] - mean_position[~low_omega]
        )
        return candidate

    def _aitken_refinement(
        self,
        phase2_positions: np.ndarray,
        current_positions: np.ndarray,
        historical_best: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Generate the additional AAMSM refinement from Eqs. (13)-(14).

        The paper writes ``X_D`` but defines only indices ``c`` and ``d``;
        therefore ``X_D`` is interpreted as ``X_d``.  The combined matrix ``S``
        is explicitly stated to contain current and historical-best solutions,
        so ``S_d`` is sampled over all rows of that combined matrix.
        """
        n = current_positions.shape[0]
        c = int(np.random.randint(0, n))
        d_population = int(np.random.randint(0, n))
        combined = np.vstack((current_positions, historical_best))
        d_combined = int(np.random.randint(0, combined.shape[0]))

        r = int(np.random.randint(0, n))
        x_r = -6.0 + 12.0 * float(r) / float(max(1, n - 1))
        r5 = float(np.clip(np.random.random(), self._epsilon, 1.0 - self._epsilon))
        alpha_raw = (
            0.2 / (1.0 + np.exp(-x_r))
            + 0.6
            + 0.1 * np.tan(np.pi * (r5 - 0.5))
        )
        alpha = float(min(1.0, alpha_raw))
        if not np.isfinite(alpha):
            alpha = 1.0 if alpha_raw > 0.0 else -1.0

        x_i = phase2_positions[i]
        denominator = np.sum(current_positions, axis=0)
        denominator = np.where(
            np.abs(denominator) > self._epsilon,
            denominator,
            np.where(denominator >= 0.0, self._epsilon, -self._epsilon),
        )
        rho = x_i / denominator
        exp_rho = np.exp(np.clip(rho, -self._exponent_clip, self._exponent_clip))
        infinity_norm = float(np.linalg.norm(exp_rho, ord=np.inf))

        return x_i + alpha * (
            (phase2_positions[d_population] - x_i) * infinity_norm
            + (phase2_positions[c] - combined[d_combined])
        )

    def _gain(self, old_fitness: float, new_fitness: float) -> float:
        if self.problem.objective == "min":
            return float(max(0.0, old_fitness - new_fitness))
        return float(max(0.0, new_fitness - old_fitness))

    def _record_accepted_gain(
        self,
        contributions: dict[str, float],
        move_label: str,
        gain: float,
    ) -> None:
        # Candidate generation is the causal search operator; greedy selection
        # receives a smaller direct share for accepting the already-evaluated move.
        contributions[move_label] += 0.8 * float(gain)
        contributions["atk.greedy_selection"] += 0.2 * float(gain)

    def _step_impl(self, state, pop: np.ndarray):
        positions = pop[:, :-1].copy()
        fitness = pop[:, -1].copy()
        historical_best = np.asarray(
            state.payload.get("historical_best_positions", positions), dtype=float
        ).copy()
        historical_fitness = np.asarray(
            state.payload.get("historical_best_fitness", fitness), dtype=float
        ).copy()
        if historical_best.shape != positions.shape:
            historical_best = positions.copy()
        if historical_fitness.shape != fitness.shape:
            historical_fitness = fitness.copy()

        remaining = self._remaining_evaluations(state)
        if remaining == 0:
            self._last_operator_contributions = self._blank_contributions()
            self._last_operator_counts = self._blank_counts()
            self._last_phase_acceptance = {"aamsm": 0, "rweo": 0, "refinement": 0}
            return pop, 0, {}

        t, horizon = self._progress_iteration(state)
        contributions = self._blank_contributions()
        counts = self._blank_counts()
        accepted = {"aamsm": 0, "rweo": 0, "refinement": 0}
        phase2_positions = positions.copy()
        evaluations = 0

        def budget_available() -> bool:
            return remaining is None or evaluations < remaining

        for i in range(positions.shape[0]):
            if not budget_available():
                break

            # Phase 1: Aitken acceleration method search mechanism.
            old_fit = float(fitness[i])
            aamsm_raw, p = self._aamsm_candidate(positions, fitness, i, t, horizon)
            aamsm, repaired = self._reflect_bounds(aamsm_raw, positions[i])
            if repaired:
                counts["atk.reflective_bound_repair"] += 1
            aamsm_fit = float(self.problem.evaluate(aamsm))
            evaluations += 1
            counts["atk.aitken_acceleration_search"] += 1
            counts["atk.greedy_selection"] += 1
            if self._is_better(aamsm_fit, old_fit):
                gain = self._gain(old_fit, aamsm_fit)
                positions[i] = aamsm
                fitness[i] = aamsm_fit
                accepted["aamsm"] += 1
                self._record_accepted_gain(
                    contributions, "atk.aitken_acceleration_search", gain
                )
                if self._is_better(aamsm_fit, historical_fitness[i]):
                    historical_best[i] = aamsm
                    historical_fitness[i] = aamsm_fit
                    counts["atk.historical_best_update"] += 1

            if not budget_available():
                break

            # Phase 2: random weighted exponential operator.
            old_fit = float(fitness[i])
            rweo_raw = self._rweo_candidate(positions, fitness, p, t, horizon)
            rweo, repaired = self._reflect_bounds(rweo_raw, positions[i])
            if repaired:
                counts["atk.reflective_bound_repair"] += 1
            phase2_positions[i] = rweo
            rweo_fit = float(self.problem.evaluate(rweo))
            evaluations += 1
            counts["atk.random_weighted_exponential_search"] += 1
            counts["atk.greedy_selection"] += 1
            if self._is_better(rweo_fit, old_fit):
                gain = self._gain(old_fit, rweo_fit)
                positions[i] = rweo
                fitness[i] = rweo_fit
                accepted["rweo"] += 1
                self._record_accepted_gain(
                    contributions, "atk.random_weighted_exponential_search", gain
                )
                if self._is_better(rweo_fit, historical_fitness[i]):
                    historical_best[i] = rweo
                    historical_fitness[i] = rweo_fit
                    counts["atk.historical_best_update"] += 1

            if not budget_available():
                break

            # Additional AAMSM refinement of the RWEO solution.
            old_fit = float(fitness[i])
            refine_raw = self._aitken_refinement(
                phase2_positions, positions, historical_best, i
            )
            refine, repaired = self._reflect_bounds(refine_raw, positions[i])
            if repaired:
                counts["atk.reflective_bound_repair"] += 1
            refine_fit = float(self.problem.evaluate(refine))
            evaluations += 1
            counts["atk.aitken_refinement"] += 1
            counts["atk.greedy_selection"] += 1
            if self._is_better(refine_fit, old_fit):
                gain = self._gain(old_fit, refine_fit)
                positions[i] = refine
                fitness[i] = refine_fit
                accepted["refinement"] += 1
                self._record_accepted_gain(contributions, "atk.aitken_refinement", gain)
                if self._is_better(refine_fit, historical_fitness[i]):
                    historical_best[i] = refine
                    historical_fitness[i] = refine_fit
                    counts["atk.historical_best_update"] += 1

        next_population = np.hstack((positions, fitness[:, None]))
        self._last_operator_contributions = {
            key: float(max(0.0, value)) for key, value in contributions.items()
        }
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        self._last_phase_acceptance = dict(accepted)
        return next_population, evaluations, {
            "historical_best_positions": historical_best,
            "historical_best_fitness": historical_fitness,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "objective_consistent_positive"
        obs["evomapx_fidelity"] = "native"
        obs["phase_acceptance"] = dict(self._last_phase_acceptance)
        obs["population_size"] = int(state.payload["population"].shape[0])
        return obs
