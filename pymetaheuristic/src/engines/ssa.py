"""pyMetaheuristic src — Salp Swarm Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class SSAEngine(BaseEngine):
    algorithm_id   = "ssa"
    algorithm_name = "Salp Swarm Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2017.07.002"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(swarm_size=5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["swarm_size"])

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
        c1=2*np.exp(-(4*(t/T))**2)
        food=state.payload.get("food",elite.copy())
        better=pop[pop[:,-1]<food[-1]]
        if len(better)>0: food=better[np.argmin(better[:,-1]),:].copy()
        hs=pop.shape[0]//2
        rng=np.array(self.problem.max_values)-np.array(self.problem.min_values)
        c2=np.random.rand(hs,self.problem.dimension); c3=np.random.rand(hs,self.problem.dimension)
        rm=c3[:hs]>=0.5
        pr=c1*(rng*c2+lo); fm=np.tile(food[:-1],(hs,1))
        pop[:hs,:-1]=np.where(rm,np.clip(fm+pr,lo,hi),np.clip(fm-pr,lo,hi))
        if hs>0 and hs+1<self._n:
            fl=pop[:hs-1,:-1] if hs>1 else pop[:1,:-1]
            fl2=pop[1:hs,:-1] if hs>1 else pop[:1,:-1]
            n2=min(fl.shape[0],fl2.shape[0])
            pop[hs:hs+n2,:-1]=np.clip((fl[:n2]+fl2[:n2])/2,lo,hi)
        pop[:,-1]=self._evaluate_population(pop[:,:-1]); evals+=self._n
        state.payload["food"]=food
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
