"""pyMetaheuristic src — Iterated Local Search Engine"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._restart_common import RestartLocalSearchEngine


class ILSEngine(RestartLocalSearchEngine):
    """Iterated Local Search for continuous box-constrained problems."""

    algorithm_id = "ils"
    algorithm_name = "Iterated Local Search"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1007/0-306-48056-5_11",
        "title": "Iterated Local Search",
        "authors": "Helena R. Lourenço, Olivier C. Martin, Thomas Stützle",
        "year": 2003,
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
        "perturbation_strength": 0.25,
        "acceptance_temperature": 0.05,
    }

    def step(self, state):
        current = np.asarray(state.payload["current"], dtype=float)
        current_fit = float(state.payload["current_fit"])
        strength = float(self._params.get("perturbation_strength", 0.25))
        temperature = max(1.0e-300, float(self._params.get("acceptance_temperature", 0.05)))
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        start = self._clip(current + self._rng.normal(0.0, strength, self.problem.dimension) * self._span)
        cand, cand_fit, evals, delta = self._local_search(start, max_evaluations=remaining)
        accepted = False
        if self._is_better(cand_fit, current_fit):
            accepted = True
        else:
            de = self._energy(cand_fit) - self._energy(current_fit)
            accepted = self._rng.random() < math.exp(-max(0.0, de) / temperature)
        if accepted:
            current, current_fit = cand, float(cand_fit)
        if self._is_better(cand_fit, state.best_fitness):
            state.best_position = cand.tolist()
            state.best_fitness = float(cand_fit)
            stagnation = 0
        else:
            stagnation = int(state.payload.get("stagnation", 0)) + 1
        if stagnation >= int(self._params.get("restart_stagnation_steps", 20)):
            extra_budget = self._remaining_evaluations(state, used=evals)
            current, current_fit, extra, delta = self._restart_current(state, max_evaluations=extra_budget)
            evals += extra
            stagnation = 0
        state.payload.update(current=current, current_fit=float(current_fit), delta=float(delta), stagnation=stagnation, last_accepted=bool(accepted))
        state.step += 1
        state.evaluations += int(evals)
        return state
