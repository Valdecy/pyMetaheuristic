"""pyMetaheuristic src — Harmony Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class HSAEngine(BaseEngine):
    algorithm_id   = "hsa"
    algorithm_name = "Harmony Search Algorithm"
    family         = "trajectory"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=200, consid_rate=0.95, adjust_rate=0.7, bw=0.05)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["size"]); self._cr=float(p["consid_rate"])
        self._ar=float(p["adjust_rate"]); self._bw=float(p["bw"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        mems=self._init_pop(); bi=mems[:,- 1].argmin(); fond=mems[bi,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=fond[:-1].tolist(),best_fitness=float(fond[-1]),
            initialized=True,payload=dict(memories=mems,fond=fond))

    def step(self, state):
        mems=state.payload["memories"]; fond=state.payload["fond"]
        lo=self.problem.min_values; hi=self.problem.max_values
        ss=list(zip(lo,hi)); h=np.zeros(self.problem.dimension+1)
        for i in range(self.problem.dimension):
            if np.random.rand()<self._cr:
                v=mems[np.random.randint(0,self._n-1),i]
                if np.random.rand()<self._ar: v=v+self._bw*np.random.uniform(-1,1)
                v=max(ss[i][0],min(ss[i][1],v))
            else: v=np.random.uniform(ss[i][0],ss[i][1])
            h[i]=v
        h[-1]=self.problem.evaluate(np.clip(h[:-1],lo,hi))
        if h[-1]<fond[-1]: fond=h.copy()
        mems=np.vstack([mems,h]); mems=mems[mems[:,-1].argsort()][:self._n]
        state.step+=1; state.evaluations+=1; state.payload=dict(memories=mems,fond=fond)
        if self.problem.is_better(float(fond[-1]),state.best_fitness):
            state.best_fitness=float(fond[-1]); state.best_position=fond[:-1].tolist()
        return state

    def observe(self, state):
        pop      = state.payload["memories"]
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
    def get_best_candidate(self,state):
        return CandidateRecord(position=list(state.best_position),fitness=state.best_fitness,
            source_algorithm=self.algorithm_id,source_step=state.step,role="best")
    def finalize(self,state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),best_fitness=state.best_fitness,
            steps=state.step,evaluations=state.evaluations,
            termination_reason=state.termination_reason,capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name,elapsed_time=state.elapsed_time))
    def get_population(self,state):
        mems=state.payload["memories"]
        return [CandidateRecord(position=mems[i,:-1].tolist(),fitness=float(mems[i,-1]),
            source_algorithm=self.algorithm_id,source_step=state.step,role="current")
            for i in range(mems.shape[0])]
