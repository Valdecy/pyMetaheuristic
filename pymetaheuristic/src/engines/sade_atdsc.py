"""
pyMetaheuristic src — Surrogate-Assisted DE with Adaptation of Training Data Selection Criterion Engine
========================================================================================================
Native macro-step: DE trial generation → best candidate selected by surrogate → true evaluation
Surrogate warning: no surrogate; best candidate selected by random choice among DE trials.
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SADEATDSCEngine(BaseEngine):
    algorithm_id   = "sade_atdsc"
    algorithm_name = "Surrogate-Assisted DE with Adaptive Training Data Selection Criterion"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/SSCI51031.2022.10022105"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, F=0.5, CR=0.9)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._F  = float(p["F"])
        self._CR = float(p["CR"])
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
        F, CR = self._F, self._CR

        # DE/current-to-best/1/bin candidates
        bi_g  = int(np.argmin(pop[:, -1]))
        best  = pop[bi_g, :-1]
        candidates = []
        for i in range(N):
            idxs = np.random.choice([j for j in range(N) if j != i], 2, replace=False)
            r1, r2 = pop[idxs[0], :-1], pop[idxs[1], :-1]
            mutant = pop[i, :-1] + F * (best - pop[i, :-1]) + F * (r1 - r2)
            cross  = np.random.rand(D) < CR
            cross[np.random.randint(D)] = True
            trial  = np.where(cross, mutant, pop[i, :-1])
            candidates.append(np.clip(trial, lo, hi))

        # Without surrogate: evaluate all, pick best improvement
        candidates = np.array(candidates)
        cand_fit   = self._evaluate_population(candidates)

        # Greedy replacement
        better = cand_fit < pop[:, -1] if self.problem.objective == "min" else cand_fit > pop[:, -1]
        pop[better, :-1] = candidates[better]
        pop[better, -1]  = cand_fit[better]

        bi  = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi, -1])
        bp  = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
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
