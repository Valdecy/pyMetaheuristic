"""
pyMetaheuristic src — Feasibility Rule with Objective Function Information Engine
==================================================================================
Native macro-step: DE offspring → FROFI environmental selection → targeted mutation
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class FROFIEngine(BaseEngine):
    algorithm_id   = "frofi"
    algorithm_name = "Feasibility Rule with Objective Function Information"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="10.1109/TCYB.2015.2493239")

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
                           initialized=True, payload=dict(population=pop))

    def _de_operator(self, pop: np.ndarray) -> np.ndarray:
        """FROFI's hybrid DE mutation (Eq. k1: current-to-rand, k2: rand-to-best with crossover)."""
        lo = np.array(self.problem.min_values, dtype=float)
        hi = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension
        CR_choices = np.array([0.1, 0.2, 1.0])
        F_choices  = np.array([0.6, 0.8, 1.0])
        CR = CR_choices[np.random.randint(3, size=N)][:, None]  # (N,1)
        F  = F_choices[np.random.randint(3, size=N)][:, None]

        # Random permutation indices for P1, P2, P3
        P  = np.argsort(np.random.rand(N, N), axis=1)
        P1 = pop[P[:, 0], :-1]
        P2 = pop[P[:, 1], :-1]
        P3 = pop[P[:, 2], :-1]
        best_idx = int(np.argmin(pop[:, -1]))
        PB = np.tile(pop[best_idx, :-1], (N, 1))

        rnd = np.random.rand(N, D)
        k1  = np.random.rand(N, 1) < 0.5  # (N,1) bool
        k2  = ~k1 & (np.random.rand(N, D) < CR)  # (N,D)
        k1_mat = np.broadcast_to(k1, (N, D))

        off = pop[:, :-1].copy()
        off[k1_mat]  = (pop[:, :-1] + rnd * (P1 - pop[:, :-1]) + F * (P2 - P3))[k1_mat]
        off[k2]      = (P1 + rnd * (PB - P1) + F * (P2 - P3))[k2]
        return np.clip(off, lo, hi)

    def _env_select(self, pop: np.ndarray, off_pos: np.ndarray, off_fit: np.ndarray) -> np.ndarray:
        """FROFI environmental selection: feasibility-rule replacement + archive-based repair."""
        # Simple greedy: replace if offspring is better (uses framework's fitness which includes penalties)
        N = pop.shape[0]
        new_pop = pop.copy()
        for i in range(off_pos.shape[0]):
            if i < N:
                if self.problem.is_better(off_fit[i], pop[i, -1]):
                    new_pop[i, :-1] = off_pos[i]
                    new_pop[i, -1]  = off_fit[i]
        return new_pop

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension
        N   = pop.shape[0]

        # DE offspring generation
        off_pos = self._de_operator(pop)
        off_fit = self._evaluate_population(off_pos)
        evals   = N

        # Environmental selection
        pop = self._env_select(pop, off_pos, off_fit)

        # Targeted mutation: if no feasible solution exists, mutate worst's single variable
        # (Framework constraint handler manages feasibility via evaluate, so we do stochastic single-gene mutation)
        worst_idx = int(np.argmax(pop[:, -1]))
        mut_pos   = pop[np.random.randint(N), :-1].copy()
        k         = np.random.randint(D)
        mut_pos[k]= np.random.uniform(lo[k], hi[k])
        mut_fit   = self.problem.evaluate(mut_pos)
        evals    += 1
        if self.problem.is_better(mut_fit, pop[worst_idx, -1]):
            pop[worst_idx, :-1] = mut_pos
            pop[worst_idx, -1]  = mut_fit

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
