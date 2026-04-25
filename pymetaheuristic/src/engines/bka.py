"""pyMetaheuristic src — Black-winged Kite Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BKAEngine(PortedPopulationEngine):
    algorithm_id   = "bka"
    algorithm_name = "Black-winged Kite Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-024-10723-4"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, p=0.9)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        p = float(self._params.get("p", 0.9))
        r = np.random.random()
        evals = 0
        t = state.step
        max_iter = self._params.get("max_iterations", 1000)
        order = self._order(pop[:, -1])
        leader_pos = pop[order[0], :-1].copy()
        for i in range(n):
            n_val = 0.05 * np.exp(-2 * (t / max_iter)**2)
            if p < r:
                new_pos = pop[i, :-1] + n_val * (1 + np.sin(r)) * pop[i, :-1]
            else:
                new_pos = pop[i, :-1] * (n_val * (2*np.random.random(d)-1) + 1)
            new_pos = np.clip(new_pos, lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
            m = 2 * np.sin(r + np.pi/2)
            s = np.random.randint(0, n)
            cauchy = np.tan((np.random.random(d) - 0.5) * np.pi)
            if pop[i, -1] < pop[s, -1]:
                new_pos2 = np.clip(pop[i, :-1] + cauchy * (pop[i, :-1] - leader_pos), lo, hi)
            else:
                new_pos2 = np.clip(pop[i, :-1] + cauchy * (leader_pos - m * pop[i, :-1]), lo, hi)
            new_fit2 = float(self._evaluate_population(new_pos2[None])[0]); evals += 1
            if self._is_better(new_fit2, pop[i, -1]):
                pop[i] = np.append(new_pos2, new_fit2)
        return pop, evals, {}
