"""pyMetaheuristic src — Hunting Search Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class HUSEngine(BaseEngine):
    algorithm_id   = "hus"
    algorithm_name = "Hunting Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ICSCCW.2009.5379451"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=15, alpha=0.5, beta=0.01, mml=0.5, c_rate=0.5, min_radius=0.5, max_radius=2.0)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._al=float(p["alpha"]); self._be=float(p["beta"])
        self._mml=float(p["mml"]); self._cr=float(p["c_rate"])
        self._minr=float(p["min_radius"]); self._maxr=float(p["max_radius"])
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
        import random
        T=self.config.max_steps or 1; t=state.step
        idx=int(np.argmin(pop[:,-1])); best=pop[idx,:].copy()
        ep=max(1,T//self._n); old=np.copy(pop)
        for i in range(self._n):
            if t%ep!=0:
                if random.random()<self._cr:
                    d=random.randint(0,self._n-1); pop[i,:-1]=pop[d,:-1]
                else: pop[i,:-1]=pop[i,:-1]+self._mml*(pop[idx,:-1]-pop[i,:-1])
                for j in range(self.problem.dimension):
                    pop[i,j]+=2*(random.random()-0.5)*self._minr*(pop[:,j].max()-pop[:,j].min())*np.exp(np.log(self._maxr/self._minr)*t/T)
            else:
                tt=t//ep
                for j in range(self.problem.dimension):
                    pop[i,j]=best[j]+2*(random.random()-0.5)*(hi[j]-lo[j])*self._al*np.exp(-self._be*tt)
            pop[i,:-1]=np.clip(pop[i,:-1],lo,hi)
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        combined=np.vstack([np.unique(old,axis=0),np.unique(pop,axis=0)])
        combined=combined[combined[:,-1].argsort()]; pop=combined[:self._n,:]
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<best[-1]: best=pop[bi,:].copy(); idx=bi
        elite=best
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
