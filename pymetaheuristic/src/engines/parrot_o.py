"""pyMetaheuristic src — Parrot Optimizer Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d, beta=1.5):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = np.random.randn(d) * sigma; v = np.random.randn(d)
    return u / np.abs(v) ** (1 / beta)

class ParrotOEngine(PortedPopulationEngine):
    algorithm_id   = "parrot_o"
    algorithm_name = "Parrot Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.heliyon.2024.e27743"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"stay": np.zeros(pop.shape[0], dtype=int),
                "pbest": pop[:, :-1].copy(), "pfit": pop[:, -1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        stay = state.payload["stay"]; pbest = state.payload["pbest"]; pfit = state.payload["pfit"]
        best_idx = self._best_index(pfit); gbest = pbest[best_idx].copy(); gbest_fit = pfit[best_idx]
        learn = 1 - t / (2 * max_iter)
        for i in range(n):
            r1 = np.random.random(d); r2 = np.random.random(d)
            St = stay[i]
            if St < max_iter / 10:
                # Foraging
                new_pos = np.clip(pop[i, :-1] + learn * r1 * (gbest - pop[i, :-1]) + r2 * (pbest[i] - pop[i, :-1]), lo, hi)
            elif St < max_iter / 5:
                # Staying with Lévy exploration
                new_pos = np.clip(pop[i, :-1] + _levy1d(d) * (gbest - pop[i, :-1]), lo, hi)
            else:
                # Fly to new area
                new_pos = np.random.uniform(lo, hi)
                stay[i] = 0
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            stay[i] += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit); stay[i] = 0
            if self._is_better(pop[i, -1], pfit[i]):
                pbest[i] = pop[i, :-1].copy(); pfit[i] = pop[i, -1]
            if self._is_better(pfit[i], gbest_fit):
                gbest = pbest[i].copy(); gbest_fit = pfit[i]
        state.payload.update({"stay": stay, "pbest": pbest, "pfit": pfit})
        return pop, evals, {}
