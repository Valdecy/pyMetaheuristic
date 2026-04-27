"""
pyMetaheuristic src — Surrogate-Assisted DE with Adaptive Multi-Subspace Search Engine
========================================================================================
Native macro-step: subspace decomposition → DE on surrogate → best candidate evaluated truly
Surrogate warning: no surrogate registered; uses true objective directly.
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SADEAMSSEngine(BaseEngine):
    algorithm_id   = "sade_amss"
    algorithm_name = "Surrogate-Assisted Differential Evolution with Adaptive Multi-Subspace Search"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/TEVC.2022.3226837"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=20, K=20, maxd=100, Gm=5)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._K  = int(p["K"])
        self._maxd = int(p["maxd"])
        self._Gm   = int(p["Gm"])
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
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension
        bi_g  = int(np.argmin(pop[:, -1]))
        bestX = pop[bi_g, :-1].copy()

        best_val  = float(pop[bi_g, -1])
        best_pos  = bestX.copy()

        # K subspace DE iterations (without surrogate: evaluate true fitness directly)
        evals = 0
        for _ in range(self._K):
            d     = np.random.randint(1, min(D, self._maxd) + 1)
            col   = np.random.choice(D, size=d, replace=False)
            # DE/rand/1/bin on subspace
            for _g in range(self._Gm):
                for i in range(N):
                    idxs = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                    r1, r2, r3 = pop[idxs[0], :-1], pop[idxs[1], :-1], pop[idxs[2], :-1]
                    mutant = bestX.copy()
                    mutant[col] = r1[col] + 0.5 * (r2[col] - r3[col])
                    trial = pop[i, :-1].copy()
                    cross = np.random.rand(d) < 0.9
                    cross[np.random.randint(d)] = True
                    trial[col[cross]] = mutant[col[cross]]
                    trial = np.clip(trial, lo, hi)
                    f     = self.problem.evaluate(trial)
                    evals += 1
                    if self.problem.is_better(f, pop[i, -1]):
                        pop[i, :-1] = trial
                        pop[i, -1]  = f
                    if self.problem.is_better(f, best_val):
                        best_val = f
                        best_pos = trial.copy()
                        bestX    = trial.copy()

        state.payload      = dict(population=pop)
        state.evaluations += evals
        state.step        += 1
        if self.problem.is_better(best_val, state.best_fitness):
            state.best_fitness  = best_val
            state.best_position = best_pos.tolist()
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

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
