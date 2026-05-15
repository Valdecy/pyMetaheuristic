"""pyMetaheuristic src — Physical Education Teacher Inspired Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class PETIOEngine(PortedPopulationEngine):
    """
    Physical Education Teacher Inspired Optimization.
.
    """

    algorithm_id = "petio"
    algorithm_name = "Physical Education Teacher Inspired Optimization"
    family = "human"
    _REFERENCE = {
        "doi": "10.13140/RG.2.2.12097.06245",
        "title": "Physical Education Teacher Inspired Optimization",
        "authors": "Jincheng Zhang",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        layers=3,
        groups_per_layer=(1, 1, 1),
        training_intensity=0.5,
        global_intensity=0.5,
        incentive_probability=0.2,
        incentive_jump=0.5,
        breakthrough_jump=0.5,
        skill_weight=0.5,
        cross_group_borrowing=0.2,
        random_innovation=0.1,
        feedback_gain=0.5,
        noise_scale=0.1,
        fine_tune_factor=0.1,
        rotation_fraction=0.3,
        min_intensity=1.0e-8,
        max_intensity=1.0e3,
        max_feedback_gain=10.0,
        max_effective_jump=1.0e3,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 3:
            raise ValueError("petio requires population_size >= 3.")
        if int(self._params.get("layers", 3)) != 3:
            raise ValueError("petio currently supports the three strata described in the paper.")
        if float(self._params.get("training_intensity", 0.5)) <= 0.0:
            raise ValueError("petio training_intensity must be positive.")
        if float(self._params.get("global_intensity", 0.5)) < 0.0:
            raise ValueError("petio global_intensity must be non-negative.")
        if float(self._params.get("incentive_jump", 0.5)) < 0.0:
            raise ValueError("petio incentive_jump must be non-negative.")
        if float(self._params.get("breakthrough_jump", 0.5)) < 0.0:
            raise ValueError("petio breakthrough_jump must be non-negative.")
        if float(self._params.get("skill_weight", 0.5)) < 0.0 or float(self._params.get("skill_weight", 0.5)) > 1.0:
            raise ValueError("petio skill_weight must be in [0, 1].")
        if float(self._params.get("cross_group_borrowing", 0.2)) < 0.0:
            raise ValueError("petio cross_group_borrowing must be non-negative.")
        if float(self._params.get("random_innovation", 0.1)) < 0.0:
            raise ValueError("petio random_innovation must be non-negative.")
        if float(self._params.get("feedback_gain", 0.5)) < 0.0:
            raise ValueError("petio feedback_gain must be non-negative.")
        if float(self._params.get("noise_scale", 0.1)) < 0.0:
            raise ValueError("petio noise_scale must be non-negative.")
        if float(self._params.get("fine_tune_factor", 0.1)) < 0.0:
            raise ValueError("petio fine_tune_factor must be non-negative.")
        rf = float(self._params.get("rotation_fraction", 0.3))
        if not 0.0 < rf <= 1.0:
            raise ValueError("petio rotation_fraction must be in (0, 1].")
        min_intensity = float(self._params.get("min_intensity", 1.0e-8))
        max_intensity = float(self._params.get("max_intensity", 1.0e3))
        if min_intensity <= 0.0 or max_intensity <= min_intensity:
            raise ValueError("petio requires 0 < min_intensity < max_intensity.")
        if float(self._params.get("max_feedback_gain", 10.0)) < 1.0:
            raise ValueError("petio max_feedback_gain must be >= 1.")
        if float(self._params.get("max_effective_jump", 1.0e3)) <= 0.0:
            raise ValueError("petio max_effective_jump must be positive.")
        pm = float(self._params.get("incentive_probability", 0.2))
        if not 0.0 <= pm <= 1.0:
            raise ValueError("petio incentive_probability must be in [0, 1].")
        groups = self._params.get("groups_per_layer", (1, 1, 1))
        if isinstance(groups, int):
            if groups < 1:
                raise ValueError("petio groups_per_layer must be at least 1.")
        else:
            if len(groups) != 3 or any(int(g) < 1 for g in groups):
                raise ValueError("petio groups_per_layer must contain three positive integers.")

    def _groups_per_layer(self) -> tuple[int, int, int]:
        groups = self._params.get("groups_per_layer", (1, 1, 1))
        if isinstance(groups, int):
            return (int(groups), int(groups), int(groups))
        return tuple(int(g) for g in groups)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        base = float(self._params.get("training_intensity", 0.5))
        return {"layer_intensity": np.array([base, base, base], dtype=float)}

    def _underperforming(self, fitness: float, mean_fitness: float) -> bool:
        if self.problem.objective == "min":
            return float(fitness) > float(mean_fitness)
        return float(fitness) < float(mean_fitness)

    def _safe_scalar(self, value: float, default: float = 0.0, lower: float | None = None, upper: float | None = None) -> float:
        value = float(value)
        if not np.isfinite(value):
            value = float(default)
        if lower is not None:
            value = max(float(lower), value)
        if upper is not None:
            value = min(float(upper), value)
        return float(value)

    def _safe_positions(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        midpoint = 0.5 * (self._lo + self._hi)
        if values.ndim == 1:
            values = np.where(np.isfinite(values), values, midpoint)
        else:
            values = np.where(np.isfinite(values), values, midpoint.reshape(1, -1))
        return np.clip(values, self._lo, self._hi)

    def _safe_candidate(self, candidate: np.ndarray) -> np.ndarray:
        candidate = np.asarray(candidate, dtype=float)
        midpoint = 0.5 * (self._lo + self._hi)
        candidate = np.where(np.isfinite(candidate), candidate, midpoint)
        return np.clip(candidate, self._lo, self._hi)

    def _safe_fitness_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        limit = 1.0e100
        if self.problem.objective == "min":
            return np.nan_to_num(values, nan=limit, posinf=limit, neginf=-limit)
        return np.nan_to_num(values, nan=-limit, posinf=limit, neginf=-limit)

    def _bounded_intensities(self, layer_intensity: np.ndarray) -> np.ndarray:
        base = float(self._params.get("training_intensity", 0.5))
        min_intensity = float(self._params.get("min_intensity", 1.0e-8))
        max_intensity = float(self._params.get("max_intensity", 1.0e3))
        layer_intensity = np.asarray(layer_intensity, dtype=float)
        layer_intensity = np.nan_to_num(layer_intensity, nan=base, posinf=max_intensity, neginf=min_intensity)
        return np.clip(layer_intensity, min_intensity, max_intensity)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        current = pop.copy()
        positions = self._safe_positions(current[:, :-1].copy())
        fitness = self._safe_fitness_array(current[:, -1].copy())
        order = self._order(fitness)

        # Best -> worst order; map to elite/core/beginner strata.
        elite_idx, core_idx, beginner_idx = np.array_split(order, 3)
        layer_members = [beginner_idx, core_idx, elite_idx]
        groups_per_layer = self._groups_per_layer()
        layer_intensity = np.asarray(state.payload.get("layer_intensity"), dtype=float).copy()
        if layer_intensity.shape != (3,):
            layer_intensity = np.full(3, float(self._params.get("training_intensity", 0.5)), dtype=float)
        layer_intensity = self._bounded_intensities(layer_intensity)

        global_best = positions[order[0]].copy()
        global_intensity = self._safe_scalar(self._params.get("global_intensity", 0.5), 0.5, 0.0, float(self._params.get("max_intensity", 1.0e3)))
        incentive_probability = float(self._params.get("incentive_probability", 0.2))
        incentive_jump = self._safe_scalar(self._params.get("incentive_jump", 0.5), 0.5, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        breakthrough_jump = self._safe_scalar(self._params.get("breakthrough_jump", 0.5), 0.5, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        skill_weight = float(self._params.get("skill_weight", 0.5))
        borrow = self._safe_scalar(self._params.get("cross_group_borrowing", 0.2), 0.2, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        innovation = self._safe_scalar(self._params.get("random_innovation", 0.1), 0.1, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        feedback_gain = self._safe_scalar(self._params.get("feedback_gain", 0.5), 0.5, 0.0, float(self._params.get("max_feedback_gain", 10.0)))
        noise_scale = self._safe_scalar(self._params.get("noise_scale", 0.1), 0.1, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        fine_tune_factor = self._safe_scalar(self._params.get("fine_tune_factor", 0.1), 0.1, 0.0, float(self._params.get("max_effective_jump", 1.0e3)))
        rotation_fraction = float(self._params.get("rotation_fraction", 0.3))
        min_intensity = float(self._params.get("min_intensity", 1.0e-8))
        max_intensity = float(self._params.get("max_intensity", 1.0e3))
        max_feedback_gain = float(self._params.get("max_feedback_gain", 10.0))
        max_effective_jump = float(self._params.get("max_effective_jump", 1.0e3))
        evals = 0
        updated_positions = positions.copy()

        # Immediate upper layer: beginner -> core, core -> elite, elite -> none.
        upper_best = {
            0: positions[core_idx[0]].copy() if core_idx.size else global_best,
            1: positions[elite_idx[0]].copy() if elite_idx.size else global_best,
            2: None,
        }

        for layer_id, layer_idx in enumerate(layer_members):
            if layer_idx.size == 0:
                continue
            group_splits = [g for g in np.array_split(layer_idx, groups_per_layer[layer_id]) if g.size > 0]
            for subgroup in group_splits:
                group_positions = updated_positions[subgroup].copy()
                group_fitness = fitness[subgroup].copy()
                subgroup_order = np.argsort(group_fitness)
                if self.problem.objective == "max":
                    subgroup_order = subgroup_order[::-1]
                x_best_group = group_positions[subgroup_order[0]].copy()
                mean_fitness = self._safe_scalar(np.mean(group_fitness), 0.0)
                variance_fitness = self._safe_scalar(np.var(group_fitness), 0.0, 0.0)
                denom = max(abs(mean_fitness), 1.0e-12)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    raw_delta = breakthrough_jump * (1.0 + variance_fitness / denom)
                delta_eff = self._safe_scalar(raw_delta, breakthrough_jump, 0.0, max_effective_jump)
                intensity = self._safe_scalar(layer_intensity[layer_id], self._params.get("training_intensity", 0.5), min_intensity, max_intensity)

                for local_pos, idx in enumerate(subgroup):
                    xi = updated_positions[idx].copy()
                    if layer_id == 0:  # beginner / exploratory
                        candidate = xi + intensity * np.random.rand(dim) * (x_best_group - xi)
                        candidate += noise_scale * np.random.normal(0.0, 1.0, dim) * self._span
                    elif layer_id == 1:  # core / balanced
                        candidate = xi + intensity * np.random.rand(dim) * (x_best_group - xi)
                        candidate += global_intensity * np.random.rand(dim) * (global_best - xi)
                    else:  # elite / development
                        candidate = xi + intensity * (x_best_group - xi) * fine_tune_factor

                    if self._underperforming(group_fitness[local_pos], mean_fitness) and np.random.rand() < incentive_probability:
                        peer_choices = [j for j in subgroup.tolist() if int(j) != int(idx)]
                        random_idx = int(np.random.choice(peer_choices)) if peer_choices else int(idx)
                        x_random = updated_positions[random_idx]
                        candidate = candidate + incentive_jump * (x_random - xi) + delta_eff * (x_best_group - xi)

                    # Skill rotation: paper leaves x_best_d ambiguous; use global best component.
                    subset_size = max(1, int(np.ceil(rotation_fraction * dim)))
                    subset = np.random.choice(dim, size=subset_size, replace=False)
                    candidate[subset] = skill_weight * candidate[subset] + (1.0 - skill_weight) * global_best[subset]

                    ubest = upper_best[layer_id]
                    if ubest is not None:
                        lower_choices = subgroup.tolist()
                        lower_rand = updated_positions[int(np.random.choice(lower_choices))]
                        candidate = candidate + borrow * (ubest - candidate) + innovation * (lower_rand - candidate)

                    updated_positions[idx] = self._safe_candidate(candidate)

            layer_positions = updated_positions[layer_idx]
            layer_new_fitness = self._safe_fitness_array(self._evaluate_population(layer_positions))
            evals += layer_idx.size
            fitness[layer_idx] = layer_new_fitness
            layer_mean = self._safe_scalar(np.mean(layer_new_fitness), 0.0)
            layer_var = self._safe_scalar(np.var(layer_new_fitness), 0.0, 0.0)
            denom = max(abs(layer_mean), 1.0e-12)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                raw_gain = 1.0 + feedback_gain * layer_var / denom
            gain = self._safe_scalar(raw_gain, 1.0, 1.0, max_feedback_gain)
            layer_intensity[layer_id] = self._safe_scalar(layer_intensity[layer_id] * gain, self._params.get("training_intensity", 0.5), min_intensity, max_intensity)

        updated_positions = self._safe_positions(updated_positions)
        fitness = self._safe_fitness_array(fitness)
        layer_intensity = self._bounded_intensities(layer_intensity)
        new_pop = np.hstack((updated_positions, fitness[:, None]))
        return new_pop, evals, {"layer_intensity": layer_intensity}
