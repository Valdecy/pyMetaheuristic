"""
pyMetaheuristic src — Sammon Mapping Assisted Differential Evolution Engine
============================================================================
Native macro-step: DE generates candidates → Sammon low-dim mapping → Kriging LCB selection → true eval
Surrogate warning: no Kriging/Sammon; candidates evaluated directly, best selected by true fitness.
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SADESammonEngine(BaseEngine):
    algorithm_id   = "sade_sammon"
    algorithm_name = "Sammon Mapping Assisted Differential Evolution"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=None, F=0.5, CR=0.5, lammda=50)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        D        = problem.dimension
        n0       = p["population_size"]
        self._n  = max(4, int(n0) if n0 else max(4, 2 * D))
        self._F  = float(p["F"])
        self._CR = float(p["CR"])
        self._lam= int(p["lammda"])
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key: str, msg: str) -> None:
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
        pos = lo + np.random.rand(N, self.problem.dimension) * (hi - lo)
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=N,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension
        F, CR = self._F, self._CR
        lam  = min(self._lam, N)

        # DE/rand/1/bin to generate lam candidates
        candidates = []
        for _ in range(lam):
            i     = np.random.randint(N)
            idxs  = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            r1, r2, r3 = pop[idxs, :-1]
            mutant= r1 + F * (r2 - r3)
            cross = np.random.rand(D) < CR
            cross[np.random.randint(D)] = True
            trial = np.where(cross, mutant, pop[i, :-1])
            candidates.append(np.clip(trial, lo, hi))

        candidates = np.array(candidates)
        cand_fit   = self._evaluate_population(candidates)
        # Best candidate (LCB → here true fitness, no surrogate)
        best_cand  = int(np.argmin(cand_fit))
        best_pos   = candidates[best_cand]
        best_fit   = float(cand_fit[best_cand])

        pop = np.vstack((pop, np.hstack((best_pos[None, :], [[best_fit]]))))

        bi  = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi, -1])
        bp  = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
        state.evaluations += lam
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    archive_size=pop.shape[0])

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(position=list(state.best_position), fitness=state.best_fitness,
                               source_algorithm=self.algorithm_id, source_step=state.step, role="best")

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
                                source_algorithm=self.algorithm_id, source_step=state.step, role="current")
                for i in range(pop.shape[0])]
