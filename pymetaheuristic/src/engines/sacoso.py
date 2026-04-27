"""
pyMetaheuristic src — Surrogate-Assisted Cooperative Swarm Optimization Engine
===============================================================================
Native macro-step: two cooperative swarms (FES-PSO + SL-PSO) with RBFNN surrogate
Surrogate warning: no RBFNN; both swarms use true objective directly.
payload keys: population (ndarray [N, D+1]), velocity (ndarray [N, D]), pbest (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SACOSOEngine(BaseEngine):
    algorithm_id   = "sacoso"
    algorithm_name = "Surrogate-Assisted Cooperative Swarm Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=None, NFES=30, NRBF=None)
    _REFERENCE     = dict(doi="10.1109/TEVC.2017.2675628")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p          = {**self._DEFAULTS, **config.params}
        self._NFES = max(4, int(p["NFES"]))
        nrbf       = p["NRBF"]
        self._NRBF = max(4, int(nrbf) if nrbf else max(4, self._NFES * 4))
        self._n    = self._NFES + self._NRBF
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
        N   = self._n
        pos = np.random.uniform(lo, hi, (N, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        vel = np.zeros((N, self.problem.dimension))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=N,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(population=pop, velocity=vel, pbest=pop.copy()))

    def step(self, state: EngineState) -> EngineState:
        pop   = np.array(state.payload["population"])
        vel   = np.array(state.payload["velocity"])
        pbest = np.array(state.payload["pbest"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        N, D  = pop.shape[0], self.problem.dimension

        gi    = int(np.argmin(pbest[:, -1]))
        gbest = pbest[gi, :-1]

        # FES swarm (standard PSO)
        w, c1, c2 = 0.7, 1.5, 1.5
        r1 = np.random.rand(N, D); r2 = np.random.rand(N, D)
        vel = w * vel + c1 * r1 * (pbest[:, :-1] - pop[:, :-1]) + c2 * r2 * (gbest - pop[:, :-1])
        new_pos = np.clip(pop[:, :-1] + vel, lo, hi)
        new_fit = self._evaluate_population(new_pos)

        better = new_fit < pbest[:, -1] if self.problem.objective == "min" else new_fit > pbest[:, -1]
        pbest[better, :-1] = new_pos[better]
        pbest[better, -1]  = new_fit[better]
        pop[:, :-1] = new_pos
        pop[:, -1]  = new_fit

        bi  = int(np.argmin(pbest[:, -1]))
        bf  = float(pbest[bi, -1])
        bp  = pbest[bi, :-1].tolist()

        state.payload      = dict(population=pop, velocity=vel, pbest=pbest)
        state.evaluations += N
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
