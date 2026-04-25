"""
pyMetaheuristic src — Mountain Gazelle Optimizer Engine
=========================================================
Native macro-step: four motion strategies (TSM, MH, BMH, MSF) merged with truncation selection
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class MGOEngine(BaseEngine):
    algorithm_id   = "mgo"
    algorithm_name = "Mountain Gazelle Optimizer"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="10.1016/j.advengsoft.2022.103282")

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
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, fe=self._n))

    def step(self, state: EngineState) -> EngineState:
        pop   = np.array(state.payload["population"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        N     = pop.shape[0]
        D     = self.problem.dimension
        FE    = state.evaluations
        maxFE = max(FE * 2, 1)   # approximation; adapted each step
        t     = FE / maxFE

        # Sort: best first
        order = np.argsort(pop[:, -1])
        pop   = pop[order]

        # Coefficient matrix (4 strategies from paper)
        Cof = np.zeros((4, D))
        Cof[0] = np.full(D, -t + np.random.rand())
        Cof[1] = (-1 - t) * np.random.randn(D)
        Cof[2] = np.random.rand(D)
        N3     = np.random.randn(D)
        Cof[3] = N3 * np.random.randn(D) ** 2 * np.cos(np.random.rand() * 2 * N3)

        def randi_mat():
            return np.random.randint(1, 3, (N, D)).astype(float)

        # Backherd mean (BH): mean of random subset of ceil(N/3)
        BH = np.array([
            pop[np.random.choice(N, size=max(1, N // 3), replace=False), :-1].mean(axis=0)
            for _ in range(N)
        ])

        cof_idx = np.random.randint(4, size=N)
        Cof_N   = Cof[cof_idx]               # (N, D)

        # TSM: territory marking
        exp_term = np.exp(2 - 2 * t)
        TSM = (np.tile(pop[0, :-1], (N, 1))
               - np.abs((randi_mat() * BH - randi_mat() * pop[:, :-1])
                        * np.random.randn(N, D) * exp_term)
               * Cof_N)

        # MH: mountain herding
        r2_idx = np.random.randint(N, size=N)
        MH = (BH + Cof[np.random.randint(4, size=N)]
              + (np.random.randint(1, 3, (N, 1)) * pop[0, :-1]
                 - randi_mat() * pop[r2_idx, :-1])
              * Cof[np.random.randint(4, size=N)])

        # BMH: best mountain herding
        BMH = (pop[:, :-1]
               - (np.abs(pop[:, :-1]) + np.tile(np.abs(pop[0, :-1]), (N, 1)))
               * (2 * np.random.rand(N, D) - 1)
               + (np.random.randint(1, 3, (N, 1)) * pop[0, :-1]
                  - randi_mat() * BH)
               * Cof[np.random.randint(4, size=N)])

        # MSF: random uniform exploration
        MSF = np.random.uniform(lo, hi, (N, D))

        # Clip all
        all_off = np.clip(np.vstack([TSM, MH, BMH, MSF]), lo, hi)
        all_fit = self._evaluate_population(all_off)
        all_pop = np.hstack((all_off, all_fit[:, None]))

        combined = np.vstack((pop, all_pop))
        order2   = np.argsort(combined[:, -1])
        pop      = combined[order2[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, fe=FE + 4 * N)
        state.evaluations += 4 * N
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
