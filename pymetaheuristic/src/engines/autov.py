"""
pyMetaheuristic src — Automated Design of Variation Operators Engine
====================================================================
Native macro-step: TSRI operator (translation/scale/rotation invariant) applied to population
File I/O warning: Weight.mat not loaded; weights randomly initialised.
payload keys: population (ndarray [N, D+1]), weights (ndarray [K, 4])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class AutoVEngine(BaseEngine):
    algorithm_id   = "autov"
    algorithm_name = "Automated Design of Variation Operators"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, n_operators=10)
    _REFERENCE     = dict(doi="10.1145/3712256.3726456")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._K  = max(1, int(p["n_operators"]))
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def _tsri_operator(self, pop: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply TSRI operator: weighted combination of difference vectors."""
        N, D  = pop.shape[0], self.problem.dimension
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        K     = weights.shape[0]
        # Cumulative fitness for operator selection
        fit_w = np.cumsum(np.abs(weights[:, -1]))
        if fit_w[-1] > 0:
            fit_w /= fit_w[-1]
        else:
            fit_w = np.linspace(0, 1, K)

        off = np.empty_like(pop[:, :-1])
        for i in range(N):
            # Select operator via roulette
            op_idx = np.searchsorted(fit_w, np.random.rand())
            op_idx = min(op_idx, K - 1)
            op     = weights[op_idx]
            a, b, c = op[0], op[1], op[2]  # scale, bias, rotation-like
            # Pick two random partners
            partners = np.random.choice([j for j in range(N) if j != i], 2, replace=False)
            diff     = pop[partners[0], :-1] - pop[partners[1], :-1]
            candidate= pop[i, :-1] + a * diff + b * np.random.randn(D) * (hi - lo) * c
            off[i]   = np.clip(candidate, lo, hi)
        return off

    def initialize(self) -> EngineState:
        self._warn_once("file_io",
            f"[{self.algorithm_id}] This algorithm normally loads/saves external weight files. "
            "File I/O is disabled; weights are initialised randomly each run.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        # Random operator weights: [scale∈[0,1], bias∈[0,1], rotation∈[-1,1], fitness_weight∈[1e-6,1]]
        weights = np.column_stack([
            np.random.rand(self._K),
            np.random.rand(self._K),
            2 * np.random.rand(self._K) - 1,
            np.random.rand(self._K) + 1e-6,
        ])
        bi = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, weights=weights))

    def step(self, state: EngineState) -> EngineState:
        pop     = np.array(state.payload["population"])
        weights = np.array(state.payload["weights"])
        N       = pop.shape[0]

        off     = self._tsri_operator(pop, weights)
        off_fit = self._evaluate_population(off)
        off_pop = np.hstack((off, off_fit[:, None]))

        combined = np.vstack((pop, off_pop))
        order    = np.argsort(combined[:, -1])
        pop      = combined[order[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, weights=weights)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        pop = state.payload["population"]
        return [CandidateRecord(pop[i, :-1].tolist(), float(pop[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(pop.shape[0])]
