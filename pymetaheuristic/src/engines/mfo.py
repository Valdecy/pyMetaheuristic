"""pyMetaheuristic src — Magnificent Frigatebird Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, EngineConfig, ProblemSpec
from ._ported_common import PortedPopulationEngine


class MFOEngine(PortedPopulationEngine):
    """Magnificent Frigatebird Optimization.

    Implements the two kleptoparasitic phases from the paper: selection/attack of
    better seabirds for exploration, followed by a local dive toward the best
    known prey for exploitation.
    """

    algorithm_id = "mfo"
    algorithm_name = "Magnificent Frigatebird Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.32604/cmc.2024.054317",
        "title": "Magnificent Frigatebird Optimization: A New Bio-Inspired Metaheuristic Approach for Solving Optimization Problems",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("population_size must be at least 2 for Magnificent Frigatebird Optimization.")

    def _better_indices_than(self, pop: np.ndarray, i: int) -> np.ndarray:
        """Eq. (4): candidate seabirds with better objective values."""
        if self.problem.objective == "min":
            idx = np.where(pop[:, -1] < pop[i, -1])[0]
        else:
            idx = np.where(pop[:, -1] > pop[i, -1])[0]
        idx = idx[idx != i]
        if idx.size == 0:
            # Eq. (4) is empty for the current best. Use the current best as the
            # target fallback, matching the common population-best convention.
            best_idx = self._best_index(pop[:, -1])
            if best_idx == i and pop.shape[0] > 1:
                idx = self._rand_indices(pop.shape[0], i, 1)
            else:
                idx = np.array([best_idx], dtype=int)
        return idx

    def _accept_if_not_worse(self, new_fitness: float, old_fitness: float) -> bool:
        if self.problem.objective == "min":
            return float(new_fitness) <= float(old_fitness)
        return float(new_fitness) >= float(old_fitness)

    def _greedy_update(self, pop: np.ndarray, i: int, trial: np.ndarray) -> bool:
        trial = np.clip(np.asarray(trial, dtype=float), self._lo, self._hi)
        fitness = float(self.problem.evaluate(trial))
        if self._accept_if_not_worse(fitness, float(pop[i, -1])):
            pop[i, :-1] = trial
            pop[i, -1] = fitness
            return True
        return False

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, state.step + 1)
        evals = 0
        operator_labels = ["carryover"] * n

        # Phase 1 — Eqs. (4)–(6): select a better seabird and attack it.
        for i in range(n):
            selected_idx = int(np.random.choice(self._better_indices_than(pop, i)))
            selected_seabird = pop[selected_idx, :-1]
            r = np.random.rand(dim)
            intensity = np.random.randint(1, 3, size=dim)  # I_ij in {1, 2}
            trial = pop[i, :-1] + (1.0 - 2.0 * r) * (selected_seabird - intensity * pop[i, :-1])
            if self._greedy_update(pop, i, trial):
                operator_labels[i] = "mfo.seabird_attack_exploration"
            evals += 1

        # Phase 2 — Eqs. (7)–(8): local dive toward the best known prey.
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        for i in range(n):
            r = np.random.rand(dim)
            trial = pop[i, :-1] + (1.0 - 2.0 * r) * ((best_pos - pop[i, :-1]) / float(t))
            if self._greedy_update(pop, i, trial):
                operator_labels[i] = "mfo.local_dive_exploitation"
            evals += 1

        return pop, evals, {"operator_labels": operator_labels}
