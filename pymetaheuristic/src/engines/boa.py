"""pyMetaheuristic src — Butterfly Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BOAEngine(PortedPopulationEngine):
    algorithm_id   = "boa"
    algorithm_name = "Butterfly Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00500-018-3102-4"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, p=0.6, power_exponent=0.1, sensory_modality=0.01)

    def _initialize_payload(self, pop):
        return {"sm": float(self._params.get("sensory_modality", 0.01))}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        p   = float(self._params.get("p", 0.6))
        pe  = float(self._params.get("power_exponent", 0.1))
        sm  = state.payload["sm"]
        evals = 0
        best_idx = self._best_index(pop[:, -1])
        best_pos = pop[best_idx, :-1].copy()
        S = pop[:, :-1].copy()
        for i in range(n):
            FP = sm * (pop[i, -1] ** pe)
            if np.random.random() < p:
                dis = np.random.random() * np.random.random() * best_pos - pop[i, :-1]
                S[i] = pop[i, :-1] + dis * FP
            else:
                jk = np.random.choice(n, 2, replace=False)
                dis = np.random.random()**2 * pop[jk[0], :-1] - pop[jk[1], :-1]
                S[i] = pop[i, :-1] + dis * FP
            S[i] = np.clip(S[i], lo, hi)
        new_fits = self._evaluate_population(S); evals += n
        mask = self._better_mask(new_fits, pop[:, -1])
        pop[mask] = np.hstack([S, new_fits[:, None]])[mask]
        max_iter = self._params.get("max_iterations", 1000)
        sm = sm + 0.025 / (sm * max_iter)
        state.payload["sm"] = sm
        return pop, evals, {}
