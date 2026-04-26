"""pyMetaheuristic src — Cockroach Swarm Optimization Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class COCKROACH_SOEngine(BaseEngine):
    algorithm_id   = "cockroach_so"
    algorithm_name = "Cockroach Swarm Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ICCET.2010.5485993"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=10, step=2)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._step=float(p["step"])
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
        old=np.copy(pop); g=np.copy(pop); cut=self._n
        joe=elite.copy()
        for i in range(cut):
            if np.random.uniform()<0.1:
                pop[i,:-1]=np.random.uniform(lo,hi); pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1; continue
            for k in range(cut):
                if pop[k,-1]<pop[i,-1]:
                    r=np.random.uniform()
                    pop[i,:-1]=np.clip(pop[i,:-1]+self._step*r*(pop[k,:-1]-pop[i,:-1]),lo,hi)
            for k in range(cut):
                r=np.random.uniform()
                g[i,:-1]=np.clip(g[i,:-1]+self._step*r*(joe[:-1]-g[i,:-1]),lo,hi)
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
            g[i,-1]=self.problem.evaluate(g[i,:-1]); evals+=1
        combined=np.vstack([old,g,pop])
        combined=combined[combined[:,-1].argsort()]; pop=combined[:cut,:]
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<joe[-1]: joe=pop[bi,:].copy()
        elite=joe
        bi=np.argmin(pop[:,-1])
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite)
        if self.problem.is_better(float(elite[-1]),state.best_fitness):
            state.best_fitness=float(elite[-1]); state.best_position=elite[:-1].tolist()
        return state

    def observe(self, state):
        pop      = state.payload["population"]
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
        pop=state.payload["population"]
        return [CandidateRecord(position=pop[i,:-1].tolist(),fitness=float(pop[i,-1]),
            source_algorithm=self.algorithm_id,source_step=state.step,role="current")
            for i in range(pop.shape[0])]
