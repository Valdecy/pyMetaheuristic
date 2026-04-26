"""pyMetaheuristic src — Hybrid Self-Adaptive Bat Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class HSABAEngine(PortedPopulationEngine):
    """Hybrid Self-Adaptive Bat Algorithm — self-adaptive bat plus DE local search."""
    algorithm_id = "hsaba"
    algorithm_name = "Hybrid Self-Adaptive Bat Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "10.1155/2014/709738"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=100, min_loudness=0.9, max_loudness=1.0, min_pulse_rate=0.001, max_pulse_rate=0.1, tao_1=0.1, tao_2=0.1, min_frequency=0.0, max_frequency=2.0, differential_weight=0.9, crossover_probability=0.85)

    def _initialize_payload(self, pop):
        n, dim = pop.shape[0], self.problem.dimension
        return {"velocities": np.zeros((n, dim)), "loudness": np.random.uniform(float(self._params.get("min_loudness", 0.9)), float(self._params.get("max_loudness", 1.0)), n), "pulse_rate": np.random.uniform(float(self._params.get("min_pulse_rate", 0.001)), float(self._params.get("max_pulse_rate", 0.1)), n)}

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        velocities = np.asarray(state.payload.get("velocities", np.zeros((n, self.problem.dimension))), dtype=float)
        loudness = np.asarray(state.payload.get("loudness", np.ones(n)), dtype=float)
        pulse = np.asarray(state.payload.get("pulse_rate", np.full(n, 0.05)), dtype=float)
        best = pop[self._best_index(pop[:, -1]), :-1]
        evals = 0
        for i in range(n):
            if np.random.rand() < float(self._params.get("tao_1", 0.1)):
                loudness[i] = np.random.uniform(float(self._params.get("min_loudness", 0.9)), float(self._params.get("max_loudness", 1.0)))
            if np.random.rand() < float(self._params.get("tao_2", 0.1)):
                pulse[i] = np.random.uniform(float(self._params.get("min_pulse_rate", 0.001)), float(self._params.get("max_pulse_rate", 0.1)))
            freq = float(self._params.get("min_frequency", 0.0)) + (float(self._params.get("max_frequency", 2.0)) - float(self._params.get("min_frequency", 0.0))) * np.random.rand()
            velocities[i] += (pop[i, :-1] - best) * freq
            bat = best + 0.1 * np.random.normal(size=self.problem.dimension) * np.mean(loudness) * self._span if np.random.rand() < pulse[i] else pop[i, :-1] + velocities[i]
            de = de_trial(self, pop, i, float(self._params.get("differential_weight", 0.9)), float(self._params.get("crossover_probability", 0.85)), best=best)
            fbat = float(self.problem.evaluate(np.clip(bat, self._lo, self._hi))); fde = float(self.problem.evaluate(de)); evals += 2
            sol, fit = (de, fde) if self._is_better(fde, fbat) else (np.clip(bat, self._lo, self._hi), fbat)
            if self._is_better(fit, pop[i, -1]) and np.random.rand() < loudness[i]:
                pop[i, :-1], pop[i, -1] = sol, fit
        return pop, evals, {"velocities": velocities, "loudness": loudness, "pulse_rate": pulse}
