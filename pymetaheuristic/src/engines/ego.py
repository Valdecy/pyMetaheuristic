"""
pyMetaheuristic src — Efficient Global Optimization Engine
===========================================================
Native macro-step: random sampling (fallback for Expected Improvement maximisation without Kriging)
Surrogate warning: no Kriging; uses random search within bounds.
payload keys: archive (ndarray [A, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class EGOEngine(BaseEngine):
    algorithm_id   = "ego"
    algorithm_name = "Efficient Global Optimization"
    family         = "distribution"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(init_samples=None, candidates_per_step=10)
    _REFERENCE     = dict(doi="10.1023/A:1008306431147")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        D           = problem.dimension
        n0          = p["init_samples"]
        self._n0    = max(2, int(n0) if n0 else 10 * D)
        self._cands = max(1, int(p["candidates_per_step"]))
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("surrogate",
            f"[{self.algorithm_id}] This algorithm normally delegates expensive evaluations to a surrogate model. "
            "No surrogate is registered; all evaluations use the true objective function, "
            "so the budget may be exhausted faster than intended.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n0, self.problem.dimension))
        fit = self._evaluate_population(pos)
        arc = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n0,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(archive=arc))

    def step(self, state: EngineState) -> EngineState:
        arc = np.array(state.payload["archive"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)

        # Fallback: random candidate (replaces EI maximisation)
        cands   = np.random.uniform(lo, hi, (self._cands, self.problem.dimension))
        cand_f  = self._evaluate_population(cands)
        best_i  = int(np.argmin(cand_f))
        new_pos = cands[best_i]
        new_fit = float(cand_f[best_i])

        arc = np.vstack((arc, np.hstack((new_pos[None, :], [[new_fit]]))))

        bi  = int(np.argmin(arc[:, -1]))
        bf  = float(arc[bi, -1])
        bp  = arc[bi, :-1].tolist()

        state.payload      = dict(archive=arc)
        state.evaluations += self._cands
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        arc = state.payload["archive"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness, archive_size=arc.shape[0])

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        arc = state.payload["archive"]
        return [CandidateRecord(arc[i, :-1].tolist(), float(arc[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(arc.shape[0])]
