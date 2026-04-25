"""
pyMetaheuristic src — Komodo Mlipir Algorithm Engine
======================================================
Native macro-step: big males move (competition), female reproduction, small males approach big males
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class KMAEngine(BaseEngine):
    algorithm_id   = "kma"
    algorithm_name = "Komodo Mlipir Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="10.1016/j.asoc.2022.108043")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(5, int(p["population_size"]))
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
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N   = pop.shape[0]
        D   = self.problem.dimension

        # Sort ascending (best first)
        order = np.argsort(pop[:, -1])
        pop   = pop[order]

        n_big    = max(1, N // 2)
        female_i = n_big          # index of female (single)
        small    = list(range(female_i + 1, N))
        big      = list(range(n_big))

        evals = 0

        # --- Big males movement ---
        big_pos  = pop[big, :-1]    # (n_big, D)
        big_fit  = pop[big, -1]     # (n_big,)
        off_big  = np.empty_like(big_pos)
        for i in range(n_big):
            signs = np.where((big_fit < big_fit[i]) | (np.random.rand(n_big) < 0.5), -1.0, 1.0)
            w     = np.random.rand(n_big)
            delta = (signs[:, None] * w[:, None] * (big_pos[i] - big_pos)).sum(axis=0)
            off_big[i] = np.clip(big_pos[i] + delta, lo, hi)

        off_big_fit = self._evaluate_population(off_big)
        evals += n_big
        combined_big     = np.vstack((pop[big], np.hstack((off_big, off_big_fit[:, None]))))
        order_big        = np.argsort(combined_big[:, -1])
        pop[big]         = combined_big[order_big[:n_big]]

        # --- Female reproduction ---
        if female_i < N:
            if np.random.rand() < 0.5:
                off_f = pop[big[0], :-1] + np.random.rand(D) * (pop[female_i, :-1] - pop[big[0], :-1])
            else:
                off_f = pop[female_i, :-1] + 0.1 * (2 * np.random.rand(D) - 1) * (hi - lo)
            off_f = np.clip(off_f, lo, hi)
            ff    = self.problem.evaluate(off_f)
            evals += 1
            if self.problem.is_better(ff, pop[female_i, -1]):
                pop[female_i, :-1] = off_f
                pop[female_i, -1]  = ff

        # --- Small males move towards big males ---
        if small:
            big_pos = pop[big, :-1]
            off_small = np.empty((len(small), D))
            for idx, i in enumerate(small):
                mask  = np.random.rand(n_big, D) < 0.5
                w     = np.random.rand(n_big, D)
                delta = (mask * w * (big_pos - pop[i, :-1])).sum(axis=0)
                off_small[idx] = np.clip(pop[i, :-1] + delta, lo, hi)
            off_small_fit = self._evaluate_population(off_small)
            evals += len(small)
            pop[small, :-1] = off_small
            pop[small, -1]  = off_small_fit

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
        state.evaluations += evals
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
