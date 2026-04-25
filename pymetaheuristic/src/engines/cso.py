"""
pyMetaheuristic src — Competitive Swarm Optimizer Engine
=========================================================
Native macro-step: random pairwise competition → losers learn from winners via velocity update
payload keys: population (ndarray [N, D+1]), velocity (ndarray [N, D])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class CSOEngine(BaseEngine):
    algorithm_id   = "cso"
    algorithm_name = "Competitive Swarm Optimizer"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, phi=0.1)
    _REFERENCE     = dict(doi="10.1109/TCYB.2014.2314537")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        if self._n % 2 != 0:
            self._n += 1
        self._phi = float(p["phi"])
        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        vel = np.zeros((self._n, self.problem.dimension))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, velocity=vel))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        vel = np.array(state.payload["velocity"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension
        N   = pop.shape[0]
        phi = self._phi

        # Random pairing
        rank   = np.random.permutation(N)
        half   = N // 2
        losers = rank[:half]
        winners= rank[half:]

        # Swap where loser actually has better fitness
        swap = pop[losers, -1] < pop[winners, -1]
        losers[swap], winners[swap] = winners[swap].copy(), losers[swap].copy()

        # Mean of all positions
        mean_pos = pop[:, :-1].mean(axis=0)

        R1 = np.random.rand(half, D)
        R2 = np.random.rand(half, D)
        R3 = np.random.rand(half, D)

        new_vel = (R1 * vel[losers]
                   + R2 * (pop[winners, :-1] - pop[losers, :-1])
                   + phi * R3 * (mean_pos - pop[losers, :-1]))
        new_pos = np.clip(pop[losers, :-1] + new_vel, lo, hi)
        new_fit = np.array([self.problem.evaluate(new_pos[i]) for i in range(half)])

        vel[losers]      = new_vel
        pop[losers, :-1] = new_pos
        pop[losers, -1]  = new_fit

        bi  = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi, -1])
        bp  = pop[bi, :-1].tolist()

        state.payload  = dict(population=pop, velocity=vel)
        state.evaluations += half
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])),
                    std_fitness=float(np.std(pop[:, -1])))

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(position=list(state.best_position),
                               fitness=state.best_fitness,
                               source_algorithm=self.algorithm_id,
                               source_step=state.step, role="best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position),
                                  best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason,
                                  capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name,
                                                elapsed_time=state.elapsed_time))

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = state.payload["population"]
        return [CandidateRecord(position=pop[i, :-1].tolist(), fitness=float(pop[i, -1]),
                                source_algorithm=self.algorithm_id,
                                source_step=state.step, role="current")
                for i in range(pop.shape[0])]
