"""
pyMetaheuristic src — Adam (Adaptive Moment Estimation) Engine
==============================================================
Native macro-step: compute gradient → update m/v moments → parameter step
payload keys: x (ndarray), m (ndarray), v (ndarray), k (int)
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


def _numerical_grad(engine: "BaseEngine", x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    engine._warn_once("grad",
        f"[{engine.algorithm_id}] This algorithm requires gradient information. "
        "A numerical finite-difference approximation is used, which may be inaccurate "
        "or slow on high-dimensional problems.")
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (engine.problem.evaluate(xp) - engine.problem.evaluate(xm)) / (2 * h)
        engine._n_evals_grad = getattr(engine, "_n_evals_grad", 0) + 2
    return g


class AdamEngine(BaseEngine):
    algorithm_id   = "adam"
    algorithm_name = "Adam (Adaptive Moment Estimation)"
    family         = "math"
    capabilities   = CapabilityProfile(has_population=False, supports_candidate_injection=False)
    _DEFAULTS      = dict(alpha=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        self._alpha = float(p["alpha"])
        self._b1    = float(p["beta1"])
        self._b2    = float(p["beta2"])
        self._eps   = float(p["epsilon"])
        self._warned: set[str] = set()
        self._n_evals_grad = 0
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key: str, msg: str) -> None:
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
                           payload=dict(x=x, m=np.zeros(D), v=np.zeros(D), k=1))

    def step(self, state: EngineState) -> EngineState:
        x  = np.array(state.payload["x"])
        m  = np.array(state.payload["m"])
        v  = np.array(state.payload["v"])
        k  = int(state.payload["k"])
        lo = np.array(self.problem.min_values, dtype=float)
        hi = np.array(self.problem.max_values, dtype=float)

        self._n_evals_grad = 0
        gk = _numerical_grad(self, x)
        m  = self._b1 * m + (1 - self._b1) * gk
        v  = self._b2 * v + (1 - self._b2) * gk ** 2
        x_new = x - self._alpha * (m / (1 - self._b1 ** k)) / (np.sqrt(v / (1 - self._b2 ** k)) + self._eps)
        x_new = np.clip(x_new, lo, hi)
        fit   = self.problem.evaluate(x_new)

        state.payload      = dict(x=x_new, m=m, v=v, k=k + 1)
        state.evaluations += 1 + self._n_evals_grad
        state.step        += 1
        if self.problem.is_better(float(fit), state.best_fitness):
            state.best_fitness  = float(fit)
            state.best_position = x_new.tolist()
        return state

    def observe(self, state: EngineState) -> dict:
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness)

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(position=list(state.best_position), fitness=state.best_fitness,
                               source_algorithm=self.algorithm_id, source_step=state.step, role="best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason,
                                  capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))
