"""
pyMetaheuristic src — Sequential Quadratic Programming Engine
==============================================================
Native macro-step: compute gradient + constraint Jacobian → solve QP subproblem →
                   augmented Lagrangian line search → BFGS Hessian update
payload keys: x (ndarray), gk (ndarray), Bk (ndarray), lam (ndarray), fit (float)
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


class SQPEngine(BaseEngine):
    algorithm_id   = "sqp"
    algorithm_name = "Sequential Quadratic Programming"
    family         = "math"
    _REFERENCE     = {"doi": "10.1017/S0962492900002518"}
    capabilities   = CapabilityProfile(has_population=False, supports_candidate_injection=False)
    _DEFAULTS      = dict(ro=0.5, eta=0.1, sigma0=0.8, max_ls=20)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        self._ro    = float(p["ro"])
        self._eta   = float(p["eta"])
        self._sig0  = float(p["sigma0"])
        self._maxls = int(p["max_ls"])
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
        self._n_evals_grad = 0
        gk  = _grad(self, x)
        D   = self.problem.dimension
        self._warn_once("constraints",
            "[sqp] SQP requires constraint function values. No constraints are registered; "
            "the quadratic sub-problem reduces to unconstrained minimisation.")
        return EngineState(step=0, evaluations=1 + self._n_evals_grad,
                           best_position=x.tolist(), best_fitness=float(fit),
                           initialized=True,
                           payload=dict(x=x, gk=gk, Bk=np.eye(D),
                                        lam=np.zeros(0), fit=float(fit), sigma=self._sig0))

    def step(self, state: EngineState) -> EngineState:
        x    = np.array(state.payload["x"])
        gk   = np.array(state.payload["gk"])
        Bk   = np.array(state.payload["Bk"])
        f0   = float(state.payload["fit"])
        sig  = float(state.payload["sigma"])
        lo   = np.array(self.problem.min_values, dtype=float)
        hi   = np.array(self.problem.max_values, dtype=float)

        # Unconstrained QP subproblem: min 0.5 d'Bk d + gk'd  → d = -Bk^{-1} gk
        try:
            dk = -np.linalg.solve(Bk, gk)
        except np.linalg.LinAlgError:
            dk = -gk

        # Line search (Armijo)
        evals = 0
        self._n_evals_grad = 0
        ro, eta = self._ro, self._eta
        x1 = x; f1 = f0
        for mk in range(self._maxls):
            alpha = ro ** mk
            xc    = np.clip(x + alpha * dk, lo, hi)
            fc    = self.problem.evaluate(xc)
            evals += 1
            if fc - f0 < eta * alpha * float(gk @ dk):
                x1 = xc; f1 = fc
                break

        gk1 = _grad(self, x1)
        sk  = x1 - x
        yk  = gk1 - gk
        sy  = float(sk @ yk)
        sBks = float(sk @ Bk @ sk)

        if sy > 0.2 * sBks:
            omega = 1.0
        else:
            denom = sBks - sy
            omega = 0.8 * sBks / max(denom, 1e-30) if denom > 1e-30 else 1.0

        zk = omega * yk + (1 - omega) * (Bk @ sk)
        szk = float(sk @ zk)
        if szk > 1e-30:
            Bk = Bk + np.outer(zk, zk) / szk - np.outer(Bk @ sk, sk @ Bk) / max(sBks, 1e-30)

        state.payload      = dict(x=x1, gk=gk1, Bk=Bk, lam=np.zeros(0), fit=f1, sigma=sig)
        state.evaluations += evals + self._n_evals_grad
        state.step        += 1
        if self.problem.is_better(f1, state.best_fitness):
            state.best_fitness  = f1
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
