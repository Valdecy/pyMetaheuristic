"""pyMetaheuristic src — Population-Based Incremental Learning Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class PBILEngine(BaseEngine):
    algorithm_id   = "pbil"
    algorithm_name = "Population-Based Incremental Learning"
    family         = "distribution"
    _REFERENCE     = {"doi": "10.1109/SSE62657.2024.00022"}
    capabilities   = CapabilityProfile(has_population=False)
    _DEFAULTS = dict(size=15, mut_factor=0.05, l_rate=0.1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["size"]); self._mf=float(p["mut_factor"]); self._lr=float(p["l_rate"])
        self._pm=1.0/len(problem.min_values)
        if config.seed is not None: np.random.seed(config.seed)

    def initialize(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        vlo=lo+(hi-lo)/2; vhi=hi.copy()
        pos=np.random.uniform(vlo,vhi,(1,self.problem.dimension))
        fit=self.problem.evaluate(pos[0])
        best=np.hstack((pos[0],fit))
        return EngineState(step=0,evaluations=1,
            best_position=pos[0].tolist(),best_fitness=float(fit),
            initialized=True,payload=dict(vlo=vlo,vhi=vhi,best=best))

    def step(self, state):
        vlo=state.payload["vlo"]; vhi=state.payload["vhi"]; best=state.payload["best"]
        cur=None; evals=0
        for _ in range(self._n):
            pos=np.random.uniform(vlo,vhi)
            fit=self.problem.evaluate(pos); evals+=1
            cand=np.hstack((pos,fit))
            if cur is None or fit<cur[-1]: cur=cand.copy()
            if fit<best[-1]: best=cand.copy()
        # update vector
        vlo=vlo*(1-self._lr)+cur[:-1]*self._lr
        vhi=vhi*(1-self._lr)+cur[:-1]*self._lr
        # mutate
        for i in range(self.problem.dimension):
            if np.random.rand()<self._pm:
                vlo[i]=vlo[i]*(1-self._mf)+np.random.uniform(0,1)*self._mf
                vhi[i]=vhi[i]*(1-self._mf)+np.random.uniform(0,1)*self._mf
        state.step+=1; state.evaluations+=evals; state.payload=dict(vlo=vlo,vhi=vhi,best=best)
        if self.problem.is_better(float(best[-1]),state.best_fitness):
            state.best_fitness=float(best[-1]); state.best_position=best[:-1].tolist()
        return state

    def observe(self,state): return dict(step=state.step,evaluations=state.evaluations,best_fitness=state.best_fitness)
    def get_best_candidate(self,state):
        return CandidateRecord(position=list(state.best_position),fitness=state.best_fitness,
            source_algorithm=self.algorithm_id,source_step=state.step,role="best")
    def finalize(self,state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),best_fitness=state.best_fitness,
            steps=state.step,evaluations=state.evaluations,
            termination_reason=state.termination_reason,capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name,elapsed_time=state.elapsed_time))
