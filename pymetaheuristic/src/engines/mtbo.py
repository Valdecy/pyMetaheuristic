"""pyMetaheuristic src — Mountaineering Team-Based Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, EngineState
from ._ported_common import PortedPopulationEngine


class MTBOEngine(PortedPopulationEngine):
    """Mountaineering Team-Based Optimization with leader, avalanche, and mean-motion phases."""

    algorithm_id = "mtbo"
    algorithm_name = "Mountaineering Team-Based Optimization"
    family = "human"
    _REFERENCE = {
        "doi": "10.3390/math11051273",
        "authors": "Iman Faridmehr, Moncef L. Nehdi, Iraj Faraji Davoudkhani, Alireza Poolad",
        "year": 2023,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def initialize(self) -> EngineState:
        state = super().initialize()
        pop = np.asarray(state.payload["population"], dtype=float)
        order = self._order(pop[:, -1])
        pop = pop[order]
        state.payload["population"] = pop
        state.best_position = pop[0, :-1].tolist()
        state.best_fitness = float(pop[0, -1])
        return state

    def _step_impl(self, state: EngineState, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        pop = pop[self._order(pop[:, -1])].copy()
        mean_position = np.mean(pop[:, :-1], axis=0)
        leader_position = pop[0, :-1].copy()
        leader_fitness = float(pop[0, -1])
        worst_position = pop[-1, :-1].copy()
        operator_labels = ["carryover"] * n

        for i in range(n):
            current = pop[i, :-1]
            teammate = pop[(i + 1) % n, :-1]

            # Probabilities from the MATLAB reference implementation:
            # Li in [0.25, 0.50], Ai in [0.75, 1.00], Mi in [0.75, 1.00].
            li = 0.25 + 0.25 * np.random.random()
            ai = 0.75 + 0.25 * np.random.random()
            mi = 0.75 + 0.25 * np.random.random()
            r = np.random.random()

            if r < li:
                # Coordinated movement with the next mountaineer and the leader.
                trial = (
                    current
                    + np.random.random(dim) * (teammate - current)
                    + np.random.random(dim) * (leader_position - teammate)
                )
                attempted_label = "mtbo.team_leader_coordinated_movement"
            elif r < ai:
                # Avalanche/disaster response: move relative to the weakest member.
                trial = current + np.random.random(dim) * (current - worst_position)
                attempted_label = "mtbo.avalanche_worst_avoidance"
            elif r < mi:
                # Movement toward team mean.
                trial = current + np.random.random(dim) * (mean_position - current)
                attempted_label = "mtbo.team_mean_movement"
            else:
                # Random relocation phase.
                trial = self._new_positions(1)[0]
                attempted_label = "mtbo.random_relocation_phase"

            trial = np.clip(trial, self._lo, self._hi)
            trial_fitness = float(self.problem.evaluate(trial))
            if self._is_better(trial_fitness, float(pop[i, -1])):
                pop[i, :-1] = trial
                pop[i, -1] = trial_fitness
                operator_labels[i] = attempted_label
                if self._is_better(trial_fitness, leader_fitness):
                    leader_position = trial.copy()
                    leader_fitness = trial_fitness

        pop = pop[self._order(pop[:, -1])]
        return pop, n, {"operator_labels": operator_labels}
