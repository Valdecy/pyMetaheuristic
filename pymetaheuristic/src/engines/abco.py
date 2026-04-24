"""
pyMetaheuristic src — Artificial Bee Colony Optimization Engine
==============================================================
Native macro-step: employed phase → onlooker phase → scout phase
payload keys: sources (np.ndarray pop+fitness), trial (np.ndarray counts)
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class ABCOEngine(BaseEngine):
    algorithm_id   = "abco"
    algorithm_name = "Artificial Bee Colony Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS = dict(food_sources=20, employed_bees=5, outlookers_bees=5, limit=10)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._food   = int(p["food_sources"])
        self._emp    = int(p["employed_bees"])
        self._onl    = int(p["outlookers_bees"])
        self._limit  = int(p["limit"])
        if config.seed is not None: np.random.seed(config.seed)

    @staticmethod
    def _fitness_calc(fv):
        fv_arr = np.asarray(fv, dtype=float)
        scalar = fv_arr.ndim == 0
        fv_flat = np.atleast_1d(fv_arr)

        out = np.empty_like(fv_flat, dtype=float)
        mask = fv_flat >= 0

        out[mask] = 1.0 / (1.0 + fv_flat[mask])
        out[~mask] = 1.0 + np.abs(fv_flat[~mask])

        return float(out[0]) if scalar else out

    def _fitness_fn(self, src):
        fv  = self._fitness_calc(src[:, -1])
        cs  = np.cumsum(fv); cs /= cs[-1]
        return np.column_stack((fv, cs))

    def _roulette(self, fit): return np.searchsorted(fit[:, 1], np.random.rand())

    def _employed(self, src):
        s   = np.copy(src); dim = self.problem.dimension
        lo  = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        trial = np.zeros((src.shape[0], 1))
        phi   = np.random.uniform(-1, 1, src.shape[0])
        js    = np.random.randint(dim, size=src.shape[0])
        ks    = [np.random.choice([k for k in range(src.shape[0]) if k != i]) for i in range(src.shape[0])]
        for i in range(src.shape[0]):
            j = js[i]; vij = s[i,j] + phi[i]*(s[i,j]-s[ks[i],j])
            ns = s[i,:-1].copy(); ns[j] = np.clip(vij, lo[j], hi[j])
            nf = self.problem.evaluate(ns)
            if self._fitness_calc(nf) > self._fitness_calc(s[i,-1]):
                s[i,j] = ns[j]; s[i,-1] = nf
            else: trial[i,0] += 1
        return s, trial

    def _onlooker(self, src, fit, trial):
        s = np.copy(src); dim = self.problem.dimension
        lo = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        t2 = np.copy(trial)
        for rep in range(src.shape[0]):
            i = self._roulette(fit); k = np.random.choice([x for x in range(src.shape[0]) if x!=i])
            j = np.random.randint(dim); phi = np.random.uniform(-1,1)
            vij = s[i,j] + phi*(s[i,j]-s[k,j])
            ns = s[i,:-1].copy(); ns[j] = np.clip(vij, lo[j], hi[j])
            nf = self.problem.evaluate(ns)
            if self._fitness_calc(nf) > self._fitness_calc(s[i,-1]):
                s[i,j]=ns[j]; s[i,-1]=nf; t2[i,0]=0
            else: t2[i,0]+=1
        return s, t2

    def _scout(self, src, trial):
        lo = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        for i in np.where(trial[:,0] > self._limit)[0]:
            src[i,:-1] = np.random.normal(0,1,self.problem.dimension)
            src[i,-1]  = self.problem.evaluate(src[i,:-1])
        return src

    def initialize(self) -> EngineState:
        lo = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        pos = np.random.uniform(lo, hi, (self._food, self.problem.dimension))
        fit = self._evaluate_population(pos)
        src = np.hstack((pos, fit[:,np.newaxis]))
        bi  = np.argmin(src[:,-1])
        return EngineState(step=0, evaluations=self._food,
            best_position=src[bi,:-1].tolist(), best_fitness=float(src[bi,-1]),
            initialized=True, payload=dict(sources=src, trial=np.zeros((self._food,1))))

    def step(self, state: EngineState) -> EngineState:
        src, trial = state.payload["sources"], state.payload["trial"]
        for _ in range(self._emp):   src, trial = self._employed(src)
        fit = self._fitness_fn(src)
        for _ in range(self._onl):   src, trial = self._onlooker(src, fit, trial)
        src = self._scout(src, trial)
        bi  = np.argmin(src[:,-1])
        state.evaluations += self._food*(self._emp+self._onl)+np.sum(trial[:,0]>self._limit)
        state.step += 1; state.payload = dict(sources=src, trial=trial)
        if self.problem.is_better(float(src[bi,-1]), state.best_fitness):
            state.best_fitness=float(src[bi,-1]); state.best_position=src[bi,:-1].tolist()
        return state

    def observe(self, state):
        pop      = state.payload["sources"]
        pos      = pop[:, :-1]
        fitness  = pop[:, -1]
        lo       = np.array(self.problem.min_values)
        hi       = np.array(self.problem.max_values)
        denom    = np.linalg.norm(hi - lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        return dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            mean_fitness=float(np.mean(fitness)),
            std_fitness=float(np.std(fitness)),
            diversity=diversity,
        )

    def get_best_candidate(self, state):
        return CandidateRecord(position=list(state.best_position), fitness=state.best_fitness,
            source_algorithm=self.algorithm_id, source_step=state.step, role="best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position), best_fitness=state.best_fitness,
            steps=state.step, evaluations=state.evaluations,
            termination_reason=state.termination_reason, capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        src = state.payload["sources"]
        return [CandidateRecord(position=src[i,:-1].tolist(), fitness=float(src[i,-1]),
            source_algorithm=self.algorithm_id, source_step=state.step, role="current")
            for i in range(src.shape[0])]
