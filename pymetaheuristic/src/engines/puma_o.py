"""pyMetaheuristic src — Puma Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class PumaOEngine(PortedPopulationEngine):
    algorithm_id   = "puma_o"
    algorithm_name = "Puma Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2024.111257"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"pbest": pop[:, :-1].copy(), "pfit": pop[:, -1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        pbest = state.payload["pbest"]; pfit = state.payload["pfit"]
        best_idx = self._best_index(pfit); gbest = pbest[best_idx].copy(); gbest_fit = pfit[best_idx]
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        w = 0.9 - 0.5 * t / max_iter
        c1 = 1.0 / phi ** 2; c2 = 1.0 / phi
        for i in range(n):
            # Stalking phase
            r1 = np.random.random(d); r2 = np.random.random(d)
            vel = w * (pop[i, :-1] - pbest[i]) + c1 * r1 * (pbest[i] - pop[i, :-1]) + c2 * r2 * (gbest - pop[i, :-1])
            new_pos = np.clip(pop[i, :-1] + vel, lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
            # Attack phase — aggressive jump toward prey
            if np.random.random() < 0.1:
                alpha = 2 * np.exp(-4 * t / max_iter)
                leap = np.clip(gbest + alpha * (2 * np.random.random(d) - 1) * (hi - lo) / 2, lo, hi)
                leap_fit = float(self._evaluate_population(leap[None])[0]); evals += 1
                if self._is_better(leap_fit, pop[i, -1]):
                    pop[i] = np.append(leap, leap_fit)
            if self._is_better(pop[i, -1], pfit[i]):
                pbest[i] = pop[i, :-1].copy(); pfit[i] = pop[i, -1]
            if self._is_better(pfit[i], gbest_fit):
                gbest = pbest[i].copy(); gbest_fit = pfit[i]
        state.payload.update({"pbest": pbest, "pfit": pfit})
        return pop, evals, {}
