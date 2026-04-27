"""pyMetaheuristic src — Tornado Optimizer with Coriolis Force Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TOCEngine(PortedPopulationEngine):
    algorithm_id   = "toc"
    algorithm_name = "Tornado Optimizer with Coriolis Force"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1007/s10462-025-11118-9"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"V": np.zeros_like(pop[:, :-1])}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        V = state.payload["V"]
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        # Spiral radius decreases over time
        r = (1 - t / max_iter) ** 2
        # Angular velocity (Coriolis-like rotation)
        omega = 2 * np.pi * (1 - t / max_iter)
        for i in range(n):
            # Tangential velocity component
            diff = best_pos - pop[i, :-1]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                tangent = np.zeros(d)
                tangent[0] = -diff[1] if d > 1 else 0
                if d > 1: tangent[1] = diff[0]
                tangent /= (np.linalg.norm(tangent) + 1e-300)
            else:
                tangent = np.random.randn(d); tangent /= (np.linalg.norm(tangent) + 1e-300)
            # Velocity update: radial + tangential + random
            V[i] = (0.5 * V[i]
                    + r * np.random.random(d) * diff
                    + omega * np.random.random() * tangent * dist
                    + 0.1 * (1 - t / max_iter) * np.random.randn(d) * (hi - lo))
            new_pos = np.clip(pop[i, :-1] + V[i], lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
                if self._is_better(pop[i, -1], pop[best_idx, -1]):
                    best_pos = pop[i, :-1].copy(); best_idx = i
        state.payload["V"] = V
        return pop, evals, {}
