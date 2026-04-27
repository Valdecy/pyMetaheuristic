"""
pyMetaheuristic src — Gradient-Based Particle Swarm Optimization Engine
========================================================================
Native macro-step: standard PSO update → local gradient refinement on global best
payload keys: population (ndarray [N,D+1]), velocity (ndarray [N,D]), pbest (ndarray [N,D+1])
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


class GPSOEngine(BaseEngine):
    algorithm_id   = "gpso"
    algorithm_name = "Gradient-Based Particle Swarm Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, w=0.7, c1=1.5, c2=1.5, ls_steps=5, ls_alpha=0.01)
    _REFERENCE     = dict(doi="10.48550/arXiv.2312.09703")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p           = {**self._DEFAULTS, **config.params}
        self._n     = max(4, int(p["population_size"]))
        self._w     = float(p["w"])
        self._c1    = float(p["c1"])
        self._c2    = float(p["c2"])
        self._ls_steps  = int(p["ls_steps"])
        self._ls_alpha  = float(p["ls_alpha"])
        self._warned: set[str] = set()
        self._n_evals_grad = 0
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def _local_search(self, x: np.ndarray, fit: float) -> tuple[np.ndarray, float, int]:
        """Gradient descent refinement on global best."""
        lo = np.array(self.problem.min_values, dtype=float)
        hi = np.array(self.problem.max_values, dtype=float)
        evals = 0
        for _ in range(self._ls_steps):
            self._n_evals_grad = 0
            g  = _grad(self, x)
            evals += self._n_evals_grad
            xn = np.clip(x - self._ls_alpha * g, lo, hi)
            fn = self.problem.evaluate(xn)
            evals += 1
            if self.problem.is_better(fn, fit):
                x, fit = xn, fn
        return x, fit, evals

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        vel = np.zeros_like(pos)
        bi  = int(np.argmin(fit))

        # Local search on initial best
        self._n_evals_grad = 0
        gbest_pos, gbest_fit, ls_evals = self._local_search(pos[bi].copy(), float(fit[bi]))
        pop[bi, :-1] = gbest_pos
        pop[bi, -1]  = gbest_fit

        return EngineState(step=0, evaluations=self._n + ls_evals,
                           best_position=gbest_pos.tolist(), best_fitness=gbest_fit,
                           initialized=True,
                           payload=dict(population=pop, velocity=vel, pbest=pop.copy()))

    def step(self, state: EngineState) -> EngineState:
        pop   = np.array(state.payload["population"])
        vel   = np.array(state.payload["velocity"])
        pbest = np.array(state.payload["pbest"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        N, D  = pop.shape[0], self.problem.dimension

        gi      = int(np.argmin(pbest[:, -1]))
        gbest_p = pbest[gi, :-1].copy()

        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)
        vel = (self._w * vel
               + self._c1 * r1 * (pbest[:, :-1] - pop[:, :-1])
               + self._c2 * r2 * (gbest_p - pop[:, :-1]))
        new_pos = np.clip(pop[:, :-1] + vel, lo, hi)
        new_fit = self._evaluate_population(new_pos)

        better = new_fit < pbest[:, -1] if self.problem.objective == "min" else new_fit > pbest[:, -1]
        pbest[better, :-1] = new_pos[better]
        pbest[better, -1]  = new_fit[better]

        pop[:, :-1] = new_pos
        pop[:, -1]  = new_fit

        # Gradient local search on global best
        gi2 = int(np.argmin(pbest[:, -1]))
        refined_pos, refined_fit, ls_evals = self._local_search(
            pbest[gi2, :-1].copy(), float(pbest[gi2, -1]))
        pbest[gi2, :-1] = refined_pos
        pbest[gi2, -1]  = refined_fit

        bi = int(np.argmin(pbest[:, -1]))
        bf = float(pbest[bi, -1])
        bp = pbest[bi, :-1].tolist()

        state.payload      = dict(population=pop, velocity=vel, pbest=pbest)
        state.evaluations += N + ls_evals
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        pop = state.payload["population"]
        return [CandidateRecord(pop[i, :-1].tolist(), float(pop[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(pop.shape[0])]
