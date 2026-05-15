"""pyMetaheuristic src — Variable Neighborhood Search Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._restart_common import RestartLocalSearchEngine


class VNSEngine(RestartLocalSearchEngine):
    """Variable Neighborhood Search with shaking and local improvement."""

    algorithm_id = "vns"
    algorithm_name = "Variable Neighborhood Search"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1016/S0305-0548(97)00031-2",
        "title": "Variable neighborhood search",
        "authors": "Nenad Mladenović, Pierre Hansen",
        "year": 1997,
    }
    capabilities = CapabilityProfile(
        has_population=False,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=False,
    )
    _DEFAULTS = {
        **RestartLocalSearchEngine._DEFAULTS,
        "k_max": 5,
        "shake_step_size": 0.10,
    }

    def initialize(self):
        state = super().initialize()
        state.payload["k"] = 1
        return state

    def step(self, state):
        current = np.asarray(state.payload["current"], dtype=float)
        current_fit = float(state.payload["current_fit"])
        k_max = max(1, int(self._params.get("k_max", 5)))
        k = max(1, min(k_max, int(state.payload.get("k", 1))))
        shake = float(self._params.get("shake_step_size", 0.10)) * k
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        start = self._clip(current + self._rng.normal(0.0, shake, self.problem.dimension) * self._span)
        cand, cand_fit, evals, delta = self._local_search(start, max_evaluations=remaining)
        if self._is_better(cand_fit, current_fit):
            current, current_fit = cand, float(cand_fit)
            k = 1
            stagnation = 0
        else:
            k += 1
            stagnation = int(state.payload.get("stagnation", 0)) + 1
        if self._is_better(cand_fit, state.best_fitness):
            state.best_position = cand.tolist()
            state.best_fitness = float(cand_fit)
        if k > k_max:
            extra_budget = self._remaining_evaluations(state, used=evals)
            current, current_fit, extra, delta = self._restart_current(state, max_evaluations=extra_budget)
            evals += extra
            k = 1
            stagnation = 0
        state.payload.update(current=current, current_fit=float(current_fit), delta=float(delta), k=int(k), stagnation=stagnation, last_accepted=k == 1)
        state.step += 1
        state.evaluations += int(evals)
        return state

    def observe(self, state):
        obs = super().observe(state)
        obs["k"] = int(state.payload.get("k", 1))
        return obs
