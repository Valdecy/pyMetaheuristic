"""pyMetaheuristic src — Divine Religions Algorithm Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class DRAEngine(PortedPopulationEngine):
    """Divine Religions Algorithm (DRA).

    Native NumPy port of the MATLAB reference operators: belief profile
    consideration, miracle/proselytism, reward/penalty, and worst-belief
    replacement by a new follower.
    """

    algorithm_id = "dra"
    algorithm_name = "Divine Religions Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-04954-x",
        "authors": "Nima Khodadadi et al.",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, number_groups=5, bpsp=0.5, reward_penalty_rate=0.2)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._groups = max(1, int(self._params.get("number_groups", 5)))
        if self._n <= self._groups:
            raise ValueError("DRA requires population_size > number_groups.")
        self._bpsp = float(self._params.get("bpsp", 0.5))
        self._rp = float(self._params.get("reward_penalty_rate", 0.2))
        if not (0.0 <= self._bpsp <= 1.0):
            raise ValueError("DRA bpsp must be in [0, 1].")
        if not (0.0 <= self._rp <= 1.0):
            raise ValueError("DRA reward_penalty_rate must be in [0, 1].")

    def _sort_population(self, pop: np.ndarray) -> np.ndarray:
        return pop[self._order(pop[:, -1])].copy()

    def initialize(self):
        state = super().initialize()
        state.payload["population"] = self._sort_population(state.payload["population"])
        best = state.payload["population"][0]
        state.best_position = best[:-1].tolist()
        state.best_fitness = float(best[-1])
        return state

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 500)))
        t = state.step + 1
        MP = float(np.random.random() * (1.0 - (t / max_iter * 2.0)) * np.random.random())
        pop = self._sort_population(pop)
        leader = pop[0, :-1].copy()
        evals = 0

        new_follower = np.random.random(dim) * (self._hi - self._lo) + self._lo
        if np.random.random() <= self._bpsp:
            d_new = int(np.random.randint(0, dim))
            source = int(np.random.randint(0, n))
            d_src = int(np.random.randint(0, dim))
            new_follower[d_new] = pop[source, d_src]
        new_follower = np.clip(new_follower, self._lo, self._hi)
        new_follower_fit = float(self.problem.evaluate(new_follower))
        evals += 1

        for i in range(n):
            current = pop[i, :-1].copy()
            if np.random.random() <= MP:
                if np.random.random() <= 0.5:
                    trial = current * np.cos(np.pi / 2.0) * (np.random.random() - np.cos(np.random.random()))
                else:
                    trial = current + np.random.random() * (current - round(1.0 ** np.random.random()) * current)
            else:
                if np.random.random() > (1.0 - MP):
                    mean_scalar = float(np.mean(current))
                    trial = (current * 0.01) + (
                        mean_scalar * (1.0 - MP)
                        + (1.0 - mean_scalar)
                        - (np.random.random() - 4.0 * np.sin(np.sin(np.pi * np.random.random())))
                    )
                else:
                    trial = leader * (np.random.random() - np.cos(np.random.random()))

            trial = np.clip(trial, self._lo, self._hi)
            trial_fit = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(trial_fit, pop[i, -1]):
                pop[i, :-1] = trial
                pop[i, -1] = trial_fit

        if self._is_better(new_follower_fit, pop[self._worst_index(pop[:, -1]), -1]):
            wi = self._worst_index(pop[:, -1])
            pop[wi, :-1] = new_follower
            pop[wi, -1] = new_follower_fit

        index = int(np.random.randint(0, n))
        scale = 1.0 - np.random.randn() if np.random.random() >= self._rp else 1.0 + np.random.randn()
        trial = np.clip(pop[index, :-1] * scale, self._lo, self._hi)
        trial_fit = float(self.problem.evaluate(trial))
        evals += 1
        if self._is_better(trial_fit, pop[index, -1]):
            pop[index, :-1] = trial
            pop[index, -1] = trial_fit

        pop = self._sort_population(pop)
        return pop, evals, {"miracle_rate": MP, "leader_position": pop[0, :-1].copy()}
