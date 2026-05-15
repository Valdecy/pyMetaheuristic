"""pyMetaheuristic src — Basin Hopping Engine"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._restart_common import RestartLocalSearchEngine


class BasinHoppingEngine(RestartLocalSearchEngine):
    """Basin-hopping: perturb, locally minimize, then accept/reject basin moves."""

    algorithm_id = "basin_hopping"
    algorithm_name = "Basin Hopping"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1021/jp970984n",
        "title": "Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms",
        "authors": "David J. Wales, Jonathan P. K. Doye",
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
        "temperature": 1.0,
        "hop_step_size": 0.25,
    }

    def step(self, state):
        current = np.asarray(state.payload["current"], dtype=float)
        current_fit = float(state.payload["current_fit"])
        hop_step = float(self._params.get("hop_step_size", 0.25))
        temperature = max(1.0e-300, float(self._params.get("temperature", 1.0)))
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        start = self._clip(current + self._rng.normal(0.0, hop_step, self.problem.dimension) * self._span)
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
