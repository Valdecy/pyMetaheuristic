"""pyMetaheuristic src — GRASP Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._restart_common import RestartLocalSearchEngine


class GRASPEngine(RestartLocalSearchEngine):
    """Greedy Randomized Adaptive Search Procedure adapted to continuous boxes."""

    algorithm_id = "grasp"
    algorithm_name = "Greedy Randomized Adaptive Search Procedure"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.1007/BF01096763",
        "title": "Greedy Randomized Adaptive Search Procedures",
        "authors": "Thomas A. Feo, Mauricio G. C. Resende",
        "year": 1995,
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
        "construction_pool_size": 12,
        "rcl_fraction": 0.35,
    }

    def _construct(self, max_evaluations: int | None = None) -> tuple[np.ndarray, float, int]:
        pool_size = max(2, int(self._params.get("construction_pool_size", 12)))
        if max_evaluations is not None:
            pool_size = min(pool_size, max(0, int(max_evaluations)))
        if pool_size <= 0:
            pos = self._random_position()
            return pos, self.problem.worst_fitness(), 0
        rcl_fraction = min(1.0, max(1.0 / pool_size, float(self._params.get("rcl_fraction", 0.35))))
        pool = self._rng.uniform(self._lo, self._hi, (pool_size, self.problem.dimension))
        fit = np.asarray([float(self.problem.evaluate(row)) for row in pool], dtype=float)
        order = np.argsort(fit) if self.problem.objective == "min" else np.argsort(fit)[::-1]
        rcl_size = max(1, int(np.ceil(rcl_fraction * pool_size)))
        chosen = int(self._rng.choice(order[:rcl_size]))
        return pool[chosen].copy(), float(fit[chosen]), pool_size

    def step(self, state):
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        start, start_fit, evals = self._construct(max_evaluations=remaining)
        extra_budget = self._remaining_evaluations(state, used=evals)
        cand, cand_fit, extra, delta = self._local_search(start, start_fit=start_fit, max_evaluations=extra_budget)
        evals += extra
        if self._is_better(cand_fit, state.best_fitness):
            state.best_position = cand.tolist()
            state.best_fitness = float(cand_fit)
            state.payload["stagnation"] = 0
        else:
            state.payload["stagnation"] = int(state.payload.get("stagnation", 0)) + 1
        state.payload.update(
            current=cand,
            current_fit=float(cand_fit),
            delta=float(delta),
            restarts=int(state.payload.get("restarts", 0)) + 1,
            last_accepted=True,
        )
        state.step += 1
        state.evaluations += int(evals)
        return state
