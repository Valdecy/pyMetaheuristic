"""pyMetaheuristic src — Moss Growth Optimization Engine"""
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

class MossGOEngine(PortedPopulationEngine):
    algorithm_id   = "moss_go"
    algorithm_name = "Moss Growth Optimization"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1093/jcde/qwae080"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"spore": pop[:, :-1].copy(), "spore_fit": pop[:, -1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        spore = state.payload["spore"]; spore_fit = state.payload["spore_fit"]
        best_idx = self._best_index(pop[:, -1]); best_pos = pop[best_idx, :-1].copy()
        # Spore dispersal
        alpha = 1 - t / max_iter
        for i in range(n):
            if np.random.random() < 0.5:
                # Wind dispersal (Lévy flight)
                step = _levy1d(d)
                new_spore = np.clip(best_pos + alpha * step * (best_pos - pop[i, :-1]), lo, hi)
            else:
                # Water dispersal (gradient-like)
                j = np.random.randint(n)
                new_spore = np.clip(pop[i, :-1] + alpha * np.random.random(d) * (pop[j, :-1] - pop[i, :-1]), lo, hi)
            new_fit = float(self._evaluate_population(new_spore[None])[0]); evals += 1
            if self._is_better(new_fit, spore_fit[i]):
                spore[i] = new_spore; spore_fit[i] = new_fit
        # Growth phase — merge spore with current
        for i in range(n):
            if self._is_better(spore_fit[i], pop[i, -1]):
                pop[i] = np.append(spore[i], spore_fit[i])
        state.payload.update({"spore": spore, "spore_fit": spore_fit})
        return pop, evals, {}
