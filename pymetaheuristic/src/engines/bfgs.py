"""
pyMetaheuristic src — BFGS Quasi-Newton Engine
===============================================
Native macro-step: compute gradient → Armijo line search → BFGS Hessian update
payload keys: x (ndarray), gk (ndarray), Bk (ndarray [D,D])
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


class BFGSEngine(BaseEngine):
    algorithm_id   = "bfgs"
    algorithm_name = "BFGS Quasi-Newton Method"
    family         = "math"
    _REFERENCE     = {"doi": "10.1090/S0025-5718-1970-0274029-X"}
    capabilities   = CapabilityProfile(has_population=False, supports_candidate_injection=False)
    _DEFAULTS      = dict(beta=0.6, sigma=0.4, max_ls=20)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        self._beta  = float(p["beta"])
        self._sigma = float(p["sigma"])
        self._maxls = int(p["max_ls"])
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
        self._n_evals_grad = 0
        gk  = _grad(self, x)
        D   = self.problem.dimension
        return EngineState(step=0, evaluations=1 + self._n_evals_grad,
                           best_position=x.tolist(), best_fitness=float(fit),
                           initialized=True,
                           payload=dict(x=x, gk=gk, Bk=np.eye(D), fit=float(fit)))

    def step(self, state: EngineState) -> EngineState:
        x   = np.array(state.payload["x"])
        gk  = np.array(state.payload["gk"])
        Bk  = np.array(state.payload["Bk"])
        f0  = float(state.payload["fit"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        beta, sigma = self._beta, self._sigma

        # Search direction
        try:
            dk = -np.linalg.solve(Bk, gk)
        except np.linalg.LinAlgError:
            dk = -gk  # fallback: steepest descent

        # Armijo line search
        evals = 0
        self._n_evals_grad = 0
        x1 = x; f1 = f0
        for m in range(self._maxls):
            alpha = beta ** m
            xc    = np.clip(x + alpha * dk, lo, hi)
            fc    = self.problem.evaluate(xc)
            evals += 1
            if fc <= f0 + sigma * alpha * float(gk @ dk):
                x1 = xc; f1 = fc
                break

        gk1 = _grad(self, x1)
        sk  = x1 - x
        yk  = gk1 - gk

        # BFGS update
        sy = float(sk @ yk)
        if sy > 0:
            sBks  = float(sk @ Bk @ sk)
            Bk    = Bk - np.outer(Bk @ sk, sk @ Bk) / max(sBks, 1e-30) + np.outer(yk, yk) / sy

        state.payload      = dict(x=x1, gk=gk1, Bk=Bk, fit=f1)
        state.evaluations += evals + self._n_evals_grad
        state.step        += 1
        if self.problem.is_better(f1, state.best_fitness):
            state.best_fitness  = f1
            state.best_position = x1.tolist()
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
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))
