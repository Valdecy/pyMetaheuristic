"""pyMetaheuristic src — Whale Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class WOAEngine(BaseEngine):
    algorithm_id   = "woa"
    algorithm_name = "Whale Optimization Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(hunting_party=25, spiral_param=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["hunting_party"])
        self._sp=float(p["spiral_param"])
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
        T=self.config.max_steps or 1; t=state.step
        al=2-t*(2/T); bl=-1+t*(-1/T)
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<elite[-1]: elite=pop[bi,:].copy()
        for i in range(self._n):
            r1=np.random.rand(); r2=np.random.rand(); A=2*al*r1-al; C=2*r2; p=np.random.rand()
            for j in range(self.problem.dimension):
                if p<0.5:
                    if abs(A)>=1:
                        ri=np.random.randint(self._n); xr=pop[ri,:]
                        pop[i,j]=np.clip(xr[j]-A*abs(C*xr[j]-pop[i,j]),lo[j],hi[j])
                    else:
                        pop[i,j]=np.clip(elite[j]-A*abs(C*elite[j]-pop[i,j]),lo[j],hi[j])
                else:
                    d=abs(elite[j]-pop[i,j]); r=np.random.rand()
                    m=(bl-1)*r+1
                    pop[i,j]=np.clip(d*np.exp(self._sp*m)*np.cos(m*2*np.pi)+elite[j],lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
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
