"""pyMetaheuristic src — Anarchic Society Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class ASOEngine(PortedPopulationEngine):
    """Anarchic Society Optimization — externality, internality and anarchic movement."""
    algorithm_id = "aso"
    algorithm_name = "Anarchic Society Optimization"
    family = "swarm"
    _REFERENCE     = {"doi": "10.1109/CEC.2011.5949940"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=43, alpha=(1.0, 0.83), gamma=(1.17, 0.56), theta=(0.932, 0.832), noise=0.05)

    def _initialize_payload(self, pop):
        return {"personal_best": pop.copy()}

    def _pair(self, name):
        v = self._params.get(name)
        return np.asarray(v if isinstance(v, (tuple, list)) else (v, v), dtype=float)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        pbest = np.asarray(state.payload.get("personal_best", pop.copy()), dtype=float)
        if pbest.shape != pop.shape: pbest = pop.copy()
        gbest = pop[self._best_index(pop[:, -1]), :-1]
        alpha, gamma, theta = self._pair("alpha"), self._pair("gamma"), self._pair("theta")
        trials = []
        for i in range(n):
            a = np.random.uniform(alpha.min(), alpha.max())
            g = np.random.uniform(gamma.min(), gamma.max())
            th = np.random.uniform(theta.min(), theta.max())
            peer = pop[np.random.randint(n), :-1]
            y = pop[i, :-1] + a * np.random.rand(dim) * (gbest - pop[i, :-1]) + g * np.random.rand(dim) * (pbest[i, :-1] - pop[i, :-1]) + th * np.random.rand(dim) * (peer - pop[i, :-1])
            y += np.random.normal(0, float(self._params.get("noise", 0.05)), dim) * self._span
            trials.append(np.clip(y, self._lo, self._hi))
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1]); pop[mask] = trial_pop[mask]
        pmask = self._better_mask(pop[:, -1], pbest[:, -1]); pbest[pmask] = pop[pmask]
        return pop, n, {"personal_best": pbest}
