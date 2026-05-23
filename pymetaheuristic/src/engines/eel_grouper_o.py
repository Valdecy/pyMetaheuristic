"""pyMetaheuristic src — Eel and Grouper Optimizer Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class EelGrouperOEngine(PortedPopulationEngine):
    """Eel and Grouper Optimizer (EGO).

    The package already uses ``ego`` for Efficient Global Optimization, so this
    engine is registered as ``eel_grouper_o`` to avoid an acronym collision.
    """

    algorithm_id = "eel_grouper_o"
    algorithm_name = "Eel and Grouper Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-04545-w",
        "authors": "Ali Mohammadzadeh, Seyedali Mirjalili",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("Eel and Grouper Optimizer requires population_size >= 2.")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        eel_idx = int(np.random.randint(0, pop.shape[0]))
        return {"eel_position": pop[eel_idx, :-1].copy()}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 500)))
        a = 2.0 - 2.0 * (t / max_iter)
        starvation_rate = 100.0 * (t / max_iter)
        positions = pop[:, :-1].copy()
        fitness = pop[:, -1].copy()
        grouper_pos = np.asarray(state.best_position, dtype=float).copy()
        grouper_score = float(state.best_fitness)
        eel_pos = np.asarray(state.payload.get("eel_position", positions[np.random.randint(0, n)]), dtype=float).copy()
        evals = 0

        for i in range(n):
            r1 = float(np.random.random())
            r2 = float(np.random.random())
            r3 = (a - 2.0) * r1 + 2.0
            r4 = 100.0 * r2
            C1 = 2.0 * a * r1 - a
            C2 = 2.0 * r1
            b = a * r2

            scout = positions[i].copy()
            for j in range(dim):
                leader_idx = int(np.floor(n * np.random.random()))
                x_rand = positions[leader_idx]
                d_x_rand = abs(C2 * positions[i, j] - x_rand[j])
                scout[j] = x_rand[j] + C1 * d_x_rand
            scout = np.clip(scout, self._lo, self._hi)
            scout_fit = float(self.problem.evaluate(scout))
            evals += 1
            if self._is_better(scout_fit, grouper_score):
                grouper_score = scout_fit
                grouper_pos = scout.copy()

            if r4 <= starvation_rate:
                eel_pos = np.abs(C2 * grouper_pos)
            else:
                eel_pos = C2 * positions[int(np.random.randint(0, n))]

            new_pos = scout.copy()
            for j in range(dim):
                p = float(np.random.random())
                distance2eel = abs(scout[j] - C2 * eel_pos[j])
                X1 = C1 * distance2eel * np.exp(b * r3) * np.sin(r3 * 2.0 * np.pi) + eel_pos[j]
                distance2grouper = abs(C2 * grouper_pos[j] - scout[j])
                X2 = grouper_pos[j] + C1 * distance2grouper
                if p < 0.5:
                    new_pos[j] = (0.8 * X1 + 0.2 * X2) / 2.0
                else:
                    new_pos[j] = (0.2 * X1 + 0.8 * X2) / 2.0
            positions[i] = new_pos

        positions = np.clip(positions, self._lo, self._hi)
        for i in range(n):
            fitness[i] = float(self.problem.evaluate(positions[i]))
            evals += 1
            if self._is_better(fitness[i], grouper_score):
                grouper_score = float(fitness[i])
                grouper_pos = positions[i].copy()

        pop[:, :-1] = positions
        pop[:, -1] = fitness
        if self._is_better(grouper_score, float(state.best_fitness)):
            state.best_fitness = float(grouper_score)
            state.best_position = grouper_pos.tolist()
        return pop, evals, {"eel_position": eel_pos.copy(), "grouper_position": grouper_pos.copy()}
