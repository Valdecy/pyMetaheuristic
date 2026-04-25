"""pyMetaheuristic src — Circulatory System Based Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CSBOEngine(PortedPopulationEngine):
    algorithm_id   = "csbo"
    algorithm_name = "Circulatory System Based Optimization"
    family         = "bio"
    _REFERENCE     = {"doi": "10.1007/s10462-021-10044-y"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        n, d = pop.shape[0], self.problem.dimension
        return {"V": np.zeros((n, d)), "T": 0.0, "T0": 1.0}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        V = state.payload["V"]
        T0 = state.payload["T0"]; T = T0 * np.exp(-t / max_iter)
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        # Systolic phase — move toward best
        w = 0.5 + 0.4 * (1 - t / max_iter)
        c1 = 1.5; c2 = 1.5
        for i in range(n):
            r1 = np.random.random(d); r2 = np.random.random(d)
            V[i] = w * V[i] + c1 * r1 * (best_pos - pop[i, :-1]) + c2 * r2 * (np.mean(pop[:, :-1], axis=0) - pop[i, :-1])
            new_pos = np.clip(pop[i, :-1] + V[i], lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
        # Diastolic phase — random perturbation
        for i in range(n):
            delta = T * np.random.randn(d) * (hi - lo)
            new_pos = np.clip(pop[i, :-1] + delta, lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            delta_E = new_fit - pop[i, -1]
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / (T + 1e-300)):
                pop[i] = np.append(new_pos, new_fit)
        state.payload.update({"V": V, "T": T, "T0": T0})
        return pop, evals, {}
