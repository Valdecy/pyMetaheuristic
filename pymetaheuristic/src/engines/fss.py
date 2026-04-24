"""pyMetaheuristic src — Fish School Search Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class FSSEngine(PortedPopulationEngine):
    """Fish School Search — individual, instinctive and volitive movements."""
    algorithm_id = "fss"
    algorithm_name = "Fish School Search"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, step_individual_init=0.1, step_individual_final=0.0001, step_volitive_init=0.01, step_volitive_final=0.001, min_weight=1.0, max_weight=5.0)

    def _initialize_payload(self, pop):
        w = np.full(pop.shape[0], (float(self._params.get("min_weight", 1.0)) + float(self._params.get("max_weight", 5.0))) / 2.0)
        return {"weight": w, "school_weight": float(w.sum())}

    def _anneal(self, a, b, state):
        T = max(1, self.config.max_steps or 100)
        t = min(T, state.step + 1)
        return float(a) + (float(b) - float(a)) * (t / T)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        weights = np.asarray(state.payload.get("weight", np.ones(n)), dtype=float)
        if weights.shape[0] != n:
            weights = np.ones(n)
        si = self._anneal(self._params.get("step_individual_init", 0.1), self._params.get("step_individual_final", 0.0001), state) * self._span
        sv = self._anneal(self._params.get("step_volitive_init", 0.01), self._params.get("step_volitive_final", 0.001), state) * self._span
        old_weight_sum = float(weights.sum())
        displacements = np.zeros((n, dim)); delta = np.zeros(n)
        evals = 0
        for i in range(n):
            trial = np.clip(pop[i, :-1] + np.random.uniform(-1, 1, dim) * si, self._lo, self._hi)
            fit = float(self.problem.evaluate(trial)); evals += 1
            if self._is_better(fit, pop[i, -1]):
                displacements[i] = trial - pop[i, :-1]
                delta[i] = abs(pop[i, -1] - fit)
                pop[i, :-1], pop[i, -1] = trial, fit
        if delta.max() > 0:
            weights += delta / (delta.max() + 1e-30)
            weights = np.clip(weights, float(self._params.get("min_weight", 1.0)), float(self._params.get("max_weight", 5.0)))
            instinct = np.sum(displacements * delta[:, None], axis=0) / (delta.sum() + 1e-30)
            moved = np.clip(pop[:, :-1] + instinct, self._lo, self._hi)
            fit = self._evaluate_population(moved); evals += n
            mask = self._better_mask(fit, pop[:, -1])
            pop[mask, :-1], pop[mask, -1] = moved[mask], fit[mask]
        bary = np.average(pop[:, :-1], axis=0, weights=np.maximum(weights, 1e-12))
        direction = pop[:, :-1] - bary
        norms = np.linalg.norm(direction, axis=1, keepdims=True); norms[norms == 0] = 1.0
        sign = -1.0 if weights.sum() > old_weight_sum else 1.0
        moved = np.clip(pop[:, :-1] + sign * np.random.rand(n, 1) * sv * direction / norms, self._lo, self._hi)
        fit = self._evaluate_population(moved); evals += n
        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1], pop[mask, -1] = moved[mask], fit[mask]
        return pop, evals, {"weight": weights, "school_weight": float(weights.sum())}
