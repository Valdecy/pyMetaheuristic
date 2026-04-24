"""pyMetaheuristic src — Hybrid Bat Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class HBAEngine(PortedPopulationEngine):
    """Hybrid Bat Algorithm — bat movement with differential local search."""
    algorithm_id = "hba"
    algorithm_name = "Hybrid Bat Algorithm"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, loudness=1.0, pulse_rate=0.5, alpha=0.97, gamma=0.1, min_frequency=0.0, max_frequency=2.0, differential_weight=0.5, crossover_probability=0.9, use_de_local_search=True)

    def _initialize_payload(self, pop):
        n, dim = pop.shape[0], self.problem.dimension
        return {"velocities": np.zeros((n, dim)), "loudness": np.full(n, float(self._params.get("loudness", self._params.get("starting_loudness", 1.0)))), "pulse_rate": np.full(n, float(self._params.get("pulse_rate", 0.5)))}

    def _bat_candidate(self, i, pop, velocities, best, loudness, pulse_rate, state):
        fmin = float(self._params.get("min_frequency", 0.0)); fmax = float(self._params.get("max_frequency", 2.0))
        freq = fmin + (fmax - fmin) * np.random.rand()
        velocities[i] += (pop[i, :-1] - best) * freq
        if np.random.rand() < pulse_rate[i]:
            sol = best + 0.1 * np.random.normal(size=self.problem.dimension) * np.mean(loudness) * self._span
        else:
            sol = pop[i, :-1] + velocities[i]
        return np.clip(sol, self._lo, self._hi)

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        velocities = np.asarray(state.payload.get("velocities", np.zeros((n, self.problem.dimension))), dtype=float)
        loudness = np.asarray(state.payload.get("loudness", np.ones(n)), dtype=float)
        pulse_rate = np.asarray(state.payload.get("pulse_rate", np.full(n, 0.5)), dtype=float)
        if velocities.shape != (n, self.problem.dimension): velocities = np.zeros((n, self.problem.dimension))
        if loudness.shape[0] != n: loudness = np.ones(n)
        if pulse_rate.shape[0] != n: pulse_rate = np.full(n, float(self._params.get("pulse_rate", 0.5)))
        best = pop[self._best_index(pop[:, -1]), :-1]
        evals = 0
        for i in range(n):
            sol = self._bat_candidate(i, pop, velocities, best, loudness, pulse_rate, state)
            if bool(self._params.get("use_de_local_search", False)):
                de = de_trial(self, pop, i, float(self._params.get("differential_weight", 0.5)), float(self._params.get("crossover_probability", 0.9)), best=best)
                fsol = float(self.problem.evaluate(sol)); fde = float(self.problem.evaluate(de)); evals += 2
                sol, fit = (de, fde) if self._is_better(fde, fsol) else (sol, fsol)
            else:
                fit = float(self.problem.evaluate(sol)); evals += 1
            if self._is_better(fit, pop[i, -1]) and np.random.rand() < max(1e-12, loudness[i]):
                pop[i, :-1], pop[i, -1] = sol, fit
                loudness[i] *= float(self._params.get("alpha", 0.97))
                pulse_rate[i] = pulse_rate[i] * (1.0 - np.exp(-float(self._params.get("gamma", 0.1)) * (state.step + 1))) + 1e-12
        return pop, evals, {"velocities": velocities, "loudness": loudness, "pulse_rate": pulse_rate}
