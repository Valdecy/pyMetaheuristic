"""pyMetaheuristic src — Parameter-Free Bat Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class PLBAEngine(PortedPopulationEngine):
    """Parameter-Free Bat Algorithm — time-varying parameter-light bat search."""
    algorithm_id = "plba"
    algorithm_name = "Parameter-Free Bat Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "https://www.iztok-jr-fister.eu/static/publications/124.pdf"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40)

    def _initialize_payload(self, pop):
        return {"velocities": np.zeros((pop.shape[0], self.problem.dimension))}

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        velocities = np.asarray(state.payload.get("velocities", np.zeros((n, self.problem.dimension))), dtype=float)
        best = pop[self._best_index(pop[:, -1]), :-1]
        T = max(1, self.config.max_steps or 100); t = state.step + 1
        loudness = 1.0 - t / (T + 1.0)
        pulse = t / (T + 1.0)
        evals = 0
        for i in range(n):
            freq = np.random.rand() * 2.0
            velocities[i] += (pop[i, :-1] - best) * freq
            sol = best + np.random.normal(0, 0.1, self.problem.dimension) * self._span * loudness if np.random.rand() < pulse else pop[i, :-1] + velocities[i]
            sol = np.clip(sol, self._lo, self._hi)
            fit = float(self.problem.evaluate(sol)); evals += 1
            if self._is_better(fit, pop[i, -1]) and np.random.rand() < max(loudness, 1e-3):
                pop[i, :-1], pop[i, -1] = sol, fit
        return pop, evals, {"velocities": velocities, "loudness": loudness, "pulse_rate": pulse}
