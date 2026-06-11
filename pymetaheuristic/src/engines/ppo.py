"""pyMetaheuristic src — Philoponella prominens Optimizer Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight


class PPOEngine(PortedPopulationEngine):
    """Philoponella prominens Optimizer — escape, cannibalism, and predation."""

    algorithm_id = "ppo"
    algorithm_name = "Philoponella prominens Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-04761-4",
        "source": "Gao, Wang and Li (2025), Escape after love: Philoponella prominens optimizer.",
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
    _DEFAULTS = dict(population_size=30, levy_beta=1.5, epsilon=1.0e-12)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._beta = float(self._params.get("levy_beta", 1.5))
        self._eps = float(self._params.get("epsilon", 1.0e-12))

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {
            "historical_best_positions": pop[:, :-1].copy(),
            "historical_best_fitness": pop[:, -1].copy(),
            "food_position": pop[self._best_index(pop[:, -1]), :-1].copy(),
            "food_fitness": float(pop[self._best_index(pop[:, -1]), -1]),
        }

    def _energy(self, fitness: np.ndarray) -> np.ndarray:
        # The paper's Eq. (2) assumes a positive minimization scale.  This
        # equivalent rank/quality normalization preserves "larger energy is better"
        # on shifted or negative objectives such as Easom.
        q = self._quality(fitness)
        return (q + self._eps) / (float(np.max(q)) + self._eps)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, int(state.step) + 1)
        total = max(1, int(self.config.max_steps or 500))
        ratio = min(1.0, float(t) / float(total))

        hpos = np.asarray(state.payload.get("historical_best_positions", pop[:, :-1]), dtype=float).copy()
        hfit = np.asarray(state.payload.get("historical_best_fitness", pop[:, -1]), dtype=float).copy()
        if hpos.shape != pop[:, :-1].shape or hfit.shape[0] != n:
            hpos = pop[:, :-1].copy()
            hfit = pop[:, -1].copy()
        food = np.asarray(state.payload.get("food_position", pop[self._best_index(pop[:, -1]), :-1]), dtype=float).copy()
        food_fit = float(state.payload.get("food_fitness", pop[self._best_index(pop[:, -1]), -1]))

        perm = np.random.permutation(n)
        females = hpos[perm].copy()
        males = pop[:, :-1].copy()
        energy = self._energy(pop[:, -1])

        # Eq. (5): escape by ejecting from the paired female.
        distances_pre = np.linalg.norm(males - females, axis=1)
        theta = np.random.uniform(0.0, np.pi, (n, dim))
        ejected = females + (energy * distances_pre)[:, None] * np.cos(theta)
        ejected = np.clip(ejected, self._lo, self._hi)
        distances = np.linalg.norm(ejected - females, axis=1)
        avg_distance = float(np.sum(distances) / max(1, n))
        survival_factor = avg_distance * (1.0 - ratio + 0.5)
        x_scale = float(np.sum(distances_pre) / max(1, n * dim))

        new_positions = ejected.copy()
        labels = ["carryover"] * n
        for i in range(n):
            if distances[i] < survival_factor:
                # Eq. (7)-(8): sexual cannibalism and juvenile generation.
                female_new = females[i] + np.random.rand() * energy[i] * (ejected[i] - females[i])
                new_positions[i] = female_new + np.exp(1.0 - ratio) * levy_flight(dim, beta=self._beta, scale=1.0) * x_scale
                labels[i] = "ppo.escape_sexual_cannibalism_juvenile_generation"
            else:
                # Eq. (10): successful escape followed by predation/local search around food.
                new_positions[i] = food + np.cos(np.random.rand() * np.pi) * (ejected[i] - food)
                labels[i] = "ppo.escape_predation_local_search"

        new_positions = np.clip(new_positions, self._lo, self._hi)
        new_fit = self._evaluate_population(new_positions)
        new_pop = np.hstack((new_positions, new_fit[:, None]))
        evals = n

        # Per-male historical best H and food ξ.
        improved_h = self._better_mask(new_fit, hfit)
        hpos[improved_h] = new_positions[improved_h]
        hfit[improved_h] = new_fit[improved_h]
        best_idx = self._best_index(new_fit)
        if self._is_better(new_fit[best_idx], food_fit):
            food = new_positions[best_idx].copy()
            food_fit = float(new_fit[best_idx])

        return new_pop, evals, {
            "historical_best_positions": hpos,
            "historical_best_fitness": hfit,
            "food_position": food,
            "food_fitness": float(food_fit),
            "operator_labels": labels,
            "native_evomapx_operator_labels": True,
            "ppo_survival_factor": float(survival_factor),
            "ppo_average_distance": float(avg_distance),
        }

    def step(self, state):
        state = super().step(state)
        food_fit = float(state.payload.get("food_fitness", state.best_fitness))
        food = np.asarray(state.payload.get("food_position", state.best_position), dtype=float)
        if state.best_fitness is None or self._is_better(food_fit, state.best_fitness):
            state.best_fitness = food_fit
            state.best_position = food.tolist()
        return state

    def observe(self, state):
        obs = super().observe(state)
        obs["food_fitness"] = float(state.payload.get("food_fitness", state.best_fitness))
        obs["survival_factor"] = float(state.payload.get("ppo_survival_factor", 0.0))
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_fidelity", "native_labels")
        return obs
