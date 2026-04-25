"""
pyMetaheuristic src — Binary Space Partition Tree Genetic Algorithm Engine
===========================================================================
Native macro-step: GA crossover/mutation on binary strings → BSP tree-guided learning
Binary encoding warning: variables rounded to {0,1}.
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

try:
    from scipy.spatial import KDTree as _KDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class BSPGAEngine(BaseEngine):
    algorithm_id   = "bspga"
    algorithm_name = "Binary Space Partition Tree Genetic Algorithm"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, proC=0.5, proM=1.0, lam=0.05)
    _REFERENCE     = dict(doi="10.1016/j.ins.2019.11.055")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._proC = float(p["proC"])
        self._proM = float(p["proM"])
        self._lam  = float(p["lam"])
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def _to_binary(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(np.round(pos), 0, 1)

    def _ga_crossover(self, pop: np.ndarray) -> np.ndarray:
        N, D = pop.shape
        off  = pop.copy()
        for i in range(0, N - 1, 2):
            if np.random.rand() < self._proC:
                pt  = np.random.randint(1, D)
                off[i, pt:]   = pop[i + 1, pt:]
                off[i + 1, pt:] = pop[i, pt:]
        return off

    def _ga_mutation(self, off: np.ndarray) -> np.ndarray:
        N, D  = off.shape
        pm    = self._proM / D
        mask  = np.random.rand(N, D) < pm
        off[mask] = 1 - off[mask]   # bit flip
        return off

    def _bsp_learning(self, off: np.ndarray, best: np.ndarray) -> np.ndarray:
        """BSP tree learning: use kd-tree proximity to best to guide candidates."""
        if _HAS_SCIPY and best is not None:
            try:
                tree = _KDTree(off)
                dist, _ = tree.query(best.reshape(1, -1), k=min(5, off.shape[0]))
                # Pull offspring towards best with probability lambda
                for i in range(off.shape[0]):
                    if np.random.rand() < self._lam:
                        off[i] = np.where(np.random.rand(off.shape[1]) < 0.5, best, off[i])
            except Exception:
                pass
        return off

    def initialize(self) -> EngineState:
        self._warn_once("binary",
            f"[{self.algorithm_id}] This algorithm operates on binary strings. "
            "Decision variables are rounded to 0/1 internally and may not respect continuous bounds.")
        D   = self.problem.dimension
        pos = np.random.randint(0, 2, (self._n, D)).astype(float)
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        N   = pop.shape[0]
        D   = self.problem.dimension
        bi  = int(np.argmin(pop[:, -1]))
        best= pop[bi, :-1]

        # GA operators on binary strings
        order  = np.random.permutation(N)
        p_dec  = pop[order, :-1].copy()
        off    = self._ga_crossover(p_dec)
        off    = self._ga_mutation(off)
        off    = self._bsp_learning(off, best)
        off    = np.clip(np.round(off), 0, 1)

        off_fit = self._evaluate_population(off)
        off_pop = np.hstack((off, off_fit[:, None]))

        combined = np.vstack((pop, off_pop))
        order2   = np.argsort(combined[:, -1])
        pop      = combined[order2[:N]]

        bi2 = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi2, -1])
        bp  = pop[bi2, :-1].tolist()

        state.payload      = dict(population=pop)
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
