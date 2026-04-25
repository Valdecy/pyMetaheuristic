"""
pyMetaheuristic src — Multifactorial Evolutionary Algorithm Engine
===================================================================
Native macro-step: crossover controlled by random mating probability (rmp) across skill factors
Multi-task warning: single registered problem; multi-task transfer disabled.
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class MFEAEngine(BaseEngine):
    algorithm_id   = "mfea"
    algorithm_name = "Multifactorial Evolutionary Algorithm"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, rmp=0.3)
    _REFERENCE     = dict(doi="10.1109/TEVC.2015.2458037")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._rmp= float(p["rmp"])
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("multitask",
            f"[{self.algorithm_id}] This algorithm is designed for simultaneous optimisation of multiple tasks. "
            "Only one task (the registered ProblemSpec) is active; multi-task transfer is disabled.")
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
        rmp  = self._rmp

        # Pair parents, SBX crossover with rmp control
        idx    = np.random.permutation(N)
        pa, pb = pop[idx[:N // 2], :-1], pop[idx[N // 2:N // 2 * 2], :-1]

        off = np.empty_like(pa)
        for i in range(pa.shape[0]):
            if np.random.rand() < rmp:
                # Inter-task (here same task): SBX crossover
                eta = 2.0
                u   = np.random.rand(D)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (eta + 1)),
                                (1 / (2 - 2 * u)) ** (1 / (eta + 1)))
                off[i] = 0.5 * ((1 + beta) * pa[i] + (1 - beta) * pb[i])
            else:
                # Polynomial mutation only
                off[i] = pa[i].copy()
            # Polynomial mutation
            pm = 1.0 / D
            for j in range(D):
                if np.random.rand() < pm:
                    delta = min(off[i, j] - lo[j], hi[j] - off[i, j]) / (hi[j] - lo[j] + 1e-30)
                    u2    = np.random.rand()
                    if u2 < 0.5:
                        off[i, j] += (hi[j] - lo[j]) * ((2 * u2 + (1 - 2 * u2) * (1 - delta) ** 21) ** (1 / 21) - 1)
                    else:
                        off[i, j] += (hi[j] - lo[j]) * (1 - (2 * (1 - u2) + 2 * (u2 - 0.5) * (1 - delta) ** 21) ** (1 / 21))
            off[i] = np.clip(off[i], lo, hi)

        off_fit = self._evaluate_population(off)
        off_pop = np.hstack((off, off_fit[:, None]))
        combined = np.vstack((pop, off_pop))
        order    = np.argsort(combined[:, -1])
        pop      = combined[order[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
        state.evaluations += off.shape[0]
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


class MFEA2Engine(BaseEngine):
    """MFEA-II: online estimation of random mating probability (RMP) matrix."""
    algorithm_id   = "mfea2"
    algorithm_name = "Multifactorial Evolutionary Algorithm II"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="10.1109/TEVC.2019.2904771")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("multitask",
            f"[{self.algorithm_id}] This algorithm is designed for simultaneous optimisation of multiple tasks. "
            "Only one task (the registered ProblemSpec) is active; multi-task transfer is disabled.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        # RMP starts at 0.5 (single-task degeneracy)
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, rmp=0.5))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        rmp = float(state.payload["rmp"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension

        # Online RMP update (single task: rmp → uniform)
        # Genetic operators: SBX + polynomial mutation
        idx    = np.random.permutation(N)
        pa, pb = pop[idx[:N // 2], :-1], pop[idx[N // 2:N // 2 * 2], :-1]
        off    = np.empty_like(pa)
        for i in range(pa.shape[0]):
            eta = 2.0
            u   = np.random.rand(D)
            beta = np.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2-2*u))**(1/(eta+1)))
            if np.random.rand() < rmp:
                off[i] = 0.5 * ((1+beta)*pa[i] + (1-beta)*pb[i])
            else:
                off[i] = pa[i].copy()
            pm = 1.0 / D
            for j in range(D):
                if np.random.rand() < pm:
                    delta = min(off[i,j]-lo[j], hi[j]-off[i,j]) / max(hi[j]-lo[j], 1e-30)
                    u2 = np.random.rand()
                    if u2 < 0.5:
                        off[i,j] += (hi[j]-lo[j]) * ((2*u2+(1-2*u2)*(1-delta)**21)**(1/21)-1)
                    else:
                        off[i,j] += (hi[j]-lo[j]) * (1-(2*(1-u2)+2*(u2-0.5)*(1-delta)**21)**(1/21))
            off[i] = np.clip(off[i], lo, hi)

        off_fit = self._evaluate_population(off)

        # Online RMP update: fraction that improved via crossover
        n_cross  = off.shape[0]
        n_better = int(np.sum(off_fit < pa[:, -1] if self.problem.objective == "min" else off_fit > pa[:, -1]))
        rmp      = min(0.9, max(0.1, (n_better + 0.5) / (n_cross + 1.0)))

        off_pop  = np.hstack((off, off_fit[:, None]))
        combined = np.vstack((pop, off_pop))
        order    = np.argsort(combined[:, -1])
        pop      = combined[order[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, rmp=rmp)
        state.evaluations += off.shape[0]
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])), rmp=state.payload["rmp"])

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
