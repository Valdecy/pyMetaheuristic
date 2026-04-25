"""
pyMetaheuristic src — RMSProp Engine
=====================================
Native macro-step: compute gradient → update running squared-gradient average → parameter step
payload keys: x (ndarray), v (ndarray [D])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


def _grad(engine, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    engine._warn_once("grad",
        f"[{engine.algorithm_id}] This algorithm requires gradient information. "
        "A numerical finite-difference approximation is used, which may be inaccurate "
        "or slow on high-dimensional problems.")
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (engine.problem.evaluate(xp) - engine.problem.evaluate(xm)) / (2 * h)
        engine._n_evals_grad += 2
    return g


class RMSPropEngine(BaseEngine):
    algorithm_id   = "rmsprop"
    algorithm_name = "RMSProp"
    family         = "math"
    capabilities   = CapabilityProfile(has_population=False, supports_candidate_injection=False)
    _DEFAULTS      = dict(alpha=1.0, rho=0.9, epsilon=1e-6)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        self._alpha = float(p["alpha"])
        self._rho   = float(p["rho"])
        self._eps   = float(p["epsilon"])
        self._warned: set[str] = set()
        self._n_evals_grad = 0
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        x   = np.random.uniform(lo, hi)
        fit = self.problem.evaluate(x)
        D   = self.problem.dimension
        return EngineState(step=0, evaluations=1,
                           best_position=x.tolist(), best_fitness=float(fit),
                           initialized=True,
                           payload=dict(x=x, v=np.zeros(D)))

    def step(self, state: EngineState) -> EngineState:
        x   = np.array(state.payload["x"])
        v   = np.array(state.payload["v"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        self._n_evals_grad = 0
        gk  = _grad(self, x)
        v   = self._rho * v + (1 - self._rho) * gk ** 2
        x1  = np.clip(x - self._alpha / np.sqrt(self._eps + v) * gk, lo, hi)
        f1  = self.problem.evaluate(x1)
        state.payload      = dict(x=x1, v=v)
        state.evaluations += 1 + self._n_evals_grad
        state.step        += 1
        if self.problem.is_better(float(f1), state.best_fitness):
            state.best_fitness  = float(f1)
            state.best_position = x1.tolist()
        return state

    def observe(self, state):
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness)

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))
