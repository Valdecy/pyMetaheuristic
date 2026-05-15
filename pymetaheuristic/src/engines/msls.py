"""pyMetaheuristic src — Multi-Start Local Search Engine"""
from __future__ import annotations

from .protocol import CapabilityProfile
from ._restart_common import RestartLocalSearchEngine


class MSLSEngine(RestartLocalSearchEngine):
    """Multi-start local search: independent random restarts followed by descent."""

    algorithm_id = "msls"
    algorithm_name = "Multi-Start Local Search"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1007/0-306-48056-5_12",
        "title": "Multi-Start Methods",
        "authors": "Rafael Martí",
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
        "starts_per_step": 1,
    }

    def step(self, state):
        starts = max(1, int(self._params.get("starts_per_step", 1)))
        evals = 0
        best_local_pos = None
        best_local_fit = None
        best_delta = float(state.payload.get("delta", self._params.get("step_size", 0.12)))
        for _ in range(starts):
            remaining = self._remaining_evaluations(state, used=evals)
            if remaining is not None and remaining <= 0:
                break
            pos, fit, local_evals, delta = self._new_local_optimum(max_evaluations=remaining)
            evals += local_evals
            if local_evals <= 0:
                break
            if best_local_fit is None or self._is_better(fit, best_local_fit):
                best_local_pos, best_local_fit, best_delta = pos, float(fit), float(delta)
        if best_local_pos is None or best_local_fit is None:
            return state
        if self._is_better(best_local_fit, state.best_fitness):
            state.best_position = best_local_pos.tolist()
            state.best_fitness = float(best_local_fit)
            stagnation = 0
        else:
            stagnation = int(state.payload.get("stagnation", 0)) + 1
        state.payload.update(
            current=best_local_pos,
            current_fit=float(best_local_fit),
            delta=float(best_delta),
            restarts=int(state.payload.get("restarts", 0)) + starts,
            stagnation=stagnation,
            last_accepted=True,
        )
        state.step += 1
        state.evaluations += int(evals)
        return state
