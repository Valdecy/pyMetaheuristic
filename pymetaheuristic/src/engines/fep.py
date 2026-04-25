"""
pyMetaheuristic src — Fast Evolutionary Programming Engine
===========================================================
Native macro-step: Cauchy mutation → tournament selection (q=10) on combined pop
payload keys: population (ndarray [N, D+1]), sigmas (ndarray [N, D])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class FEPEngine(BaseEngine):
    algorithm_id   = "fep"
    algorithm_name = "Fast Evolutionary Programming"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, q=10)
    _REFERENCE     = dict(doi="10.1109/4235.771163")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["population_size"]))
        self._q = int(p["q"])   # tournament size
        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize(self) -> EngineState:
        lo   = np.array(self.problem.min_values, dtype=float)
        hi   = np.array(self.problem.max_values, dtype=float)
        pos  = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit  = self._evaluate_population(pos)
        pop  = np.hstack((pos, fit[:, None]))
        # Initial strategy params: σ_i = 0.1*(hi-lo)
        sig  = np.tile(0.1 * (hi - lo), (self._n, 1))
        bi   = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, sigmas=sig))

    def step(self, state: EngineState) -> EngineState:
        pop  = np.array(state.payload["population"])
        sig  = np.array(state.payload["sigmas"])
        lo   = np.array(self.problem.min_values, dtype=float)
        hi   = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension

        # FEP mutation: Cauchy perturbation of position and log-normal mutation of sigma
        tau  = 1.0 / np.sqrt(2.0 * D)
        tau2 = 1.0 / np.sqrt(2.0 * np.sqrt(D))
        # Mutate strategy params
        sig_off = sig * np.exp(tau2 * np.random.randn(N, 1) + tau * np.random.randn(N, D))
        sig_off = np.maximum(sig_off, 1e-12)
        # Cauchy mutation of decision variables
        cauchy = np.random.standard_cauchy((N, D))
        off_pos = np.clip(pop[:, :-1] + sig_off * cauchy, lo, hi)
        off_fit = np.array([self.problem.evaluate(off_pos[i]) for i in range(N)])
        off_pop = np.hstack((off_pos, off_fit[:, None]))

        # Combine parent + offspring
        combined     = np.vstack((pop, off_pop))
        combined_fit = combined[:, -1]
        combined_sig = np.vstack((sig, sig_off))

        # Tournament selection (q random opponents)
        q    = min(self._q, 2 * N - 1)
        wins = np.zeros(2 * N, dtype=int)
        for i in range(2 * N):
            opponents = np.random.choice([j for j in range(2 * N) if j != i], size=q, replace=False)
            wins[i]   = int(np.sum(combined_fit[i] <= combined_fit[opponents]))

        rank = np.argsort(wins)[::-1][:N]
        pop  = combined[rank]
        sig  = combined_sig[rank]

        bi  = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi, -1])
        bp  = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, sigmas=sig)
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
