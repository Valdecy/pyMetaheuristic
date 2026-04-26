"""pyMetaheuristic src — Tabu Search Engine"""
from __future__ import annotations
import numpy as np
import warnings
from .protocol import CapabilityProfile
from ._ported_common import PortedTrajectoryEngine

class TSEngine(PortedTrajectoryEngine):
    """Tabu Search — neighbourhood search with tabu list preventing revisits."""
    algorithm_id   = "ts"
    algorithm_name = "Tabu Search"
    family         = "trajectory"
    _REFERENCE     = {"doi": "10.1287/ijoc.1.3.190"}
    capabilities   = CapabilityProfile(
        has_population=False,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
    )
    _DEFAULTS = dict(tabu_size=10, neighbour_size=20, perturbation_scale=0.05)

    def initialize(self):
        warnings.warn(
            "[ts] Tabu list stores continuous-valued fingerprints (rounded to 4 d.p.); "
            "exact revisit detection for floating-point spaces is approximate.",
            stacklevel=2)
        pos = np.random.uniform(self._lo, self._hi)
        fit = float(self.problem.evaluate(pos))
        return __import__('pymetaheuristic.src.engines.protocol', fromlist=['EngineState']).EngineState(
            step=0, evaluations=1,
            best_position=pos.tolist(), best_fitness=fit, initialized=True,
            payload={"current": pos, "current_fit": fit, "tabu": []})

    def step(self, state):
        from pymetaheuristic.src.engines.protocol import EngineState
        tabu_size   = int(self._params.get("tabu_size", 10))
        nb_size     = int(self._params.get("neighbour_size", 20))
        scale       = float(self._params.get("perturbation_scale", 0.05))

        current     = np.asarray(state.payload["current"], dtype=float)
        current_fit = float(state.payload["current_fit"])
        tabu        = list(state.payload.get("tabu", []))

        candidates  = np.clip(
            np.random.normal(current, scale * self._span, (nb_size, self.problem.dimension)),
            self._lo, self._hi)

        best_pos, best_fit = None, None
        evals = 0
        for cand in candidates:
            key = tuple(np.round(cand, 4))
            if key in tabu: continue
            fit = float(self.problem.evaluate(cand)); evals += 1
            if best_pos is None or self._is_better(fit, best_fit):
                best_pos, best_fit = cand.copy(), fit

        if best_pos is not None:
            current, current_fit = best_pos, best_fit
            tabu.append(tuple(np.round(current, 4)))
            if len(tabu) > tabu_size: tabu.pop(0)

        state.payload.update({"current": current, "current_fit": current_fit, "tabu": tabu})
        state.step += 1; state.evaluations += evals
        if self._is_better(current_fit, state.best_fitness):
            state.best_position = current.tolist(); state.best_fitness = current_fit
        return state
