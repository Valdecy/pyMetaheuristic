"""
pyMetaheuristic src — Optimal Foraging Algorithm Engine
========================================================
Native macro-step: foraging movement with ranked stochastic acceptance criterion
payload keys: population (ndarray [N, D+1]), gen (int)
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class OFAEngine(BaseEngine):
    algorithm_id   = "ofa"
    algorithm_name = "Optimal Foraging Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="10.1016/j.eswa.2022.117735")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["population_size"]))
        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        order = np.argsort(pop[:, -1])
        pop   = pop[order]
        bi    = 0
        return EngineState(step=0, evaluations=self._n,
                           best_position=pop[bi, :-1].tolist(), best_fitness=float(pop[bi, -1]),
                           initialized=True, payload=dict(population=pop, gen=1))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        gen = int(state.payload["gen"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N   = pop.shape[0]
        FE  = state.evaluations
        # approximate maxFE ratio
        t   = FE / max(FE, 1)

        # OFA movement: X_new = X + (FE/maxFE)*(r1-r2)*(X - X_nbr)
        # neighbour is randomly picked from rest, shifted by one (wrap)
        idx_shift = np.array([np.random.randint(1, N) for _ in range(N)])
        idx_nbr   = (np.arange(N) + idx_shift) % N

        # Use gen/N as proxy for FE/maxFE progress
        ratio  = min(1.0, gen / max(self._n, 1))
        r1     = np.random.rand(N, self.problem.dimension)
        r2     = np.random.rand(N, self.problem.dimension)
        off_pos = pop[:, :-1] + ratio * (r1 - r2) * (pop[:, :-1] - pop[idx_nbr, :-1])
        off_pos = np.clip(off_pos, lo, hi)
        off_fit = self._evaluate_population(off_pos)

        # Stochastic acceptance: λ * f_off / (1 + λ*⌈FE/N⌉) < f_pop / ⌈FE/N⌉
        lam     = np.random.rand(N)
        denom   = max(1, int(np.ceil(FE / N)))
        accept  = (lam * off_fit / (1.0 + lam * denom)) < (pop[:, -1] / denom)
        pop[accept, :-1] = off_pos[accept]
        pop[accept, -1]  = off_fit[accept]

        # Re-sort
        order = np.argsort(pop[:, -1])
        pop   = pop[order]

        bi = 0
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, gen=gen + 1)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

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
