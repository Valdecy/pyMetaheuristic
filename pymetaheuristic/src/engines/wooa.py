"""pyMetaheuristic src — Wolverine Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class WoOAEngine(PortedPopulationEngine):
    """Wolverine Optimization Algorithm.

    Implements the paper's two feeding strategies: scavenging toward better
    predators and hunting through attack plus fighting/chasing phases.
    """

    algorithm_id = "wooa"
    algorithm_name = "Wolverine Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.32604/cmes.2024.055171",
        "title": "Using the Novel Wolverine Optimization Algorithm for Solving Engineering Applications",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, scavenging_probability=0.5)

    def _better_indices_than(self, pop: np.ndarray, i: int) -> np.ndarray:
        if self.problem.objective == "min":
            idx = np.where(pop[:, -1] < pop[i, -1])[0]
        else:
            idx = np.where(pop[:, -1] > pop[i, -1])[0]
        idx = idx[idx != i]
        if idx.size == 0:
            idx = np.array([self._best_index(pop[:, -1])], dtype=int)
        return idx

    def _greedy_single(self, pop: np.ndarray, i: int, x: np.ndarray) -> bool:
        x = np.clip(x, self._lo, self._hi)
        fx = float(self.problem.evaluate(x))
        if self._is_better(fx, float(pop[i, -1])):
            pop[i, :-1] = x
            pop[i, -1] = fx
            return True
        return False

    def _toward_target(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        r = np.random.rand(self.problem.dimension)
        I = np.random.randint(1, 3, size=self.problem.dimension)
        return x + r * (target - I * x)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, state.step + 1)
        p_scavenge = float(self._params.get("scavenging_probability", 0.5))
        p_scavenge = min(1.0, max(0.0, p_scavenge))
        evals = 0

        for i in range(n):
            if np.random.rand() <= p_scavenge:
                # Strategy 1 — scavenging: follow a randomly selected better predator.
                predators = self._better_indices_than(pop, i)
                sp = pop[int(np.random.choice(predators)), :-1]
                trial = self._toward_target(pop[i, :-1], sp)
                self._greedy_single(pop, i, trial)
                evals += 1
            else:
                # Strategy 2, phase 1 — attack the prey, represented by the best member.
                prey = pop[self._best_index(pop[:, -1]), :-1].copy()
                trial = self._toward_target(pop[i, :-1], prey)
                self._greedy_single(pop, i, trial)
                evals += 1

                # Strategy 2, phase 2 — fight/chase with a shrinking local step.
                r = np.random.rand(dim)
                trial = pop[i, :-1] + (1.0 - 2.0 * r) * self._span / float(t)
                self._greedy_single(pop, i, trial)
                evals += 1

        return pop, evals, {}
