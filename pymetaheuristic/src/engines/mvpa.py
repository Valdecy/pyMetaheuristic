"""
pyMetaheuristic src — Most Valuable Player Algorithm Engine
============================================================
Native macro-step: team formation, MVP-guided updates via c1/c2 attraction, archive selection
payload keys: population (ndarray [N, D+1]), mvp_pos (ndarray [D])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class MVPAEngine(BaseEngine):
    algorithm_id   = "mvpa"
    algorithm_name = "Most Valuable Player Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s12351-017-0320-y"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, team_size=None, archive_size=None, c1=1.0, c2=2.0)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(5, int(p["population_size"]))
        ts      = p["team_size"]
        self._ts  = max(2, int(ts)) if ts else max(2, self._n // 5)
        as_     = p["archive_size"]
        self._as  = max(2, int(as_)) if as_ else max(2, self._n // 3)
        self._c1  = float(p["c1"])
        self._c2  = float(p["c2"])
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
                           initialized=True,
                           payload=dict(population=pop, mvp_pos=pop[bi, :-1].copy()))

    def step(self, state: EngineState) -> EngineState:
        pop     = np.array(state.payload["population"])
        mvp_pos = np.array(state.payload["mvp_pos"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N   = pop.shape[0]
        D   = self.problem.dimension

        # Archive: top archive_size players
        order   = np.argsort(pop[:, -1])
        archive = pop[order[:self._as], :-1]

        # Each player moves toward MVP + random archive member
        arc_pick = archive[np.random.randint(self._as, size=N)]
        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)
        new_pos = (pop[:, :-1]
                   + self._c1 * r1 * (mvp_pos - pop[:, :-1])
                   + self._c2 * r2 * (arc_pick - pop[:, :-1]))
        new_pos = np.clip(new_pos, lo, hi)
        new_fit = self._evaluate_population(new_pos)

        # Greedy replacement
        better  = new_fit < pop[:, -1] if self.problem.objective == "min" else new_fit > pop[:, -1]
        pop[better, :-1] = new_pos[better]
        pop[better, -1]  = new_fit[better]

        # Update MVP
        bi      = int(np.argmin(pop[:, -1]))
        mvp_pos = pop[bi, :-1].copy()
        bf      = float(pop[bi, -1])
        bp      = mvp_pos.tolist()

        state.payload      = dict(population=pop, mvp_pos=mvp_pos)
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
