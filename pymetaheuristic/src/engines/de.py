"""pyMetaheuristic src — Differential Evolution Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class DEEngine(BaseEngine):
    algorithm_id   = "de"
    algorithm_name = "Differential Evolution"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1023/A:1008202821328"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=3, F=0.9, Cr=0.2)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._F=float(p["F"]); self._Cr=float(p["Cr"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self, n=None):
        if n is None: n = self._n
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); bi=np.argmin(pop[:,-1])
        elite=pop[bi,:].copy()

        return EngineState(step=0,evaluations=self._n,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,elite=elite))

    def step(self, state):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        evals=0
        best=pop[pop[:,-1].argsort()][0,:]
        for i in range(self._n):
            k1=np.random.randint(self._n); k2=np.random.randint(self._n)
            while k1==k2: k1=np.random.randint(self._n)
            v=np.copy(best)
            for j in range(self.problem.dimension):
                ri=np.random.rand()
                v[j]=np.clip(best[j]+self._F*(pop[k1,j]-pop[k2,j]) if ri<=self._Cr else pop[i,j],lo[j],hi[j])
            v[-1]=self.problem.evaluate(v[:-1]); evals+=1
            if v[-1]<=pop[i,-1]: pop[i,:]=v
        best=pop[pop[:,-1].argsort()][0,:]
        bi=np.argmin(pop[:,-1])
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite)
        if self.problem.is_better(float(elite[-1]),state.best_fitness):
            state.best_fitness=float(elite[-1]); state.best_position=elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        fitness = pop[:, -1]
        pos = pop[:, :-1]
        lo = np.array(self.problem.min_values)
        hi = np.array(self.problem.max_values)
        denom = np.linalg.norm(hi - lo) or 1.0
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
        pop=state.payload["population"]
        return [CandidateRecord(position=pop[i,:-1].tolist(),fitness=float(pop[i,-1]),
            source_algorithm=self.algorithm_id,source_step=state.step,role="current")
            for i in range(pop.shape[0])]
