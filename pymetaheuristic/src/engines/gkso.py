"""pyMetaheuristic src — Genghis Khan Shark Optimizer Engine"""
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

class GKSOEngine(PortedPopulationEngine):
    algorithm_id   = "gkso"
    algorithm_name = "Genghis Khan Shark Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-023-10618-w"}
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
        w = 0.9 - 0.5 * (t / max_iter)
        # Phase 1: GK exploration (crossover with random partner)
        for i in range(n):
            j = np.random.randint(n)
            r = np.random.random(d)
            trial = r * pop[i, :-1] + (1 - r) * pop[j, :-1]
            trial = np.clip(trial, lo, hi)
            trial_fit = float(self._evaluate_population(trial[None])[0]); evals += 1
            if self._is_better(trial_fit, pop[i, -1]):
                pop[i] = np.append(trial, trial_fit)
        # Phase 2: Shark hunt — PSO-like toward gbest
        for i in range(n):
            r1 = np.random.random(d); r2 = np.random.random(d)
            vel = w * (pbest[i] - pop[i, :-1]) + 1.5 * r1 * (gbest - pop[i, :-1]) + 1.5 * r2 * _levy1d(d)
            new_pos = np.clip(pop[i, :-1] + vel, lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            if self._is_better(new_fit, pop[i, -1]):
                pop[i] = np.append(new_pos, new_fit)
            if self._is_better(pop[i, -1], pfit[i]):
                pbest[i] = pop[i, :-1].copy(); pfit[i] = pop[i, -1]
            if self._is_better(pfit[i], gbest_fit):
                gbest = pbest[i].copy(); gbest_fit = pfit[i]
        state.payload.update({"pbest": pbest, "pfit": pfit})
        return pop, evals, {}
