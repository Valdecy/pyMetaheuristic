"""pyMetaheuristic src — Biogeography-Based Optimization Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class BBOEngine(BaseEngine):
    algorithm_id   = "bbo"
    algorithm_name = "Biogeography-Based Optimization"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/TEVC.2008.919004"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=50, mutation_rate=0.1, elite=1, eta=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._mr=float(p["mutation_rate"]); self._elite=int(p["elite"]); self._eta=float(p["eta"])
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
        n=self._n
        mu=[(n+1-i)/(n+1) for i in range(n)]; lbd=[1-mu[i] for i in range(n)]
        island=np.zeros_like(pop)
        for k in range(n):
            if np.random.rand()<lbd[k]:
                cs=np.cumsum(mu); idx=np.searchsorted(cs,np.random.rand()*cs[-1])
                island[k,:-1]=pop[idx,:-1]
            else: island[k,:-1]=pop[k,:-1]
            island[k,-1]=self.problem.evaluate(island[k,:-1]); evals+=1
        prob=np.random.rand(n,self.problem.dimension); rnd=np.random.rand(n,self.problem.dimension)
        rnd_d=np.random.rand(n,self.problem.dimension)
        dm=np.where(rnd<=0.5,2*rnd_d**(1/(self._eta+1))-1,1-2*(1-rnd_d)**(1/(self._eta+1)))
        mut=prob<self._mr
        island[self._elite:,:-1]=np.where(mut[self._elite:],np.clip(island[self._elite:,:-1]+dm[self._elite:],lo,hi),island[self._elite:,:-1])
        for i in range(self._elite,n):
            if np.any(mut[i]): island[i,-1]=self.problem.evaluate(island[i,:-1]); evals+=1
        pop=np.vstack([pop,island]); pop=pop[pop[:,-1].argsort()][:n,:]
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
