"""pyMetaheuristic src — Arithmetic Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class AOAEngine(BaseEngine):
    algorithm_id   = "aoa"
    algorithm_name = "Arithmetic Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.cma.2020.113609"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=250, alpha=0.5, mu=5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["size"]); self._alpha=float(p["alpha"]); self._mu=float(p["mu"])
        if config.seed is not None: np.random.seed(config.seed)

    def initialize(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        pop=np.hstack((pos,fit[:,np.newaxis])); elite=pop[pop[:,-1].argsort()][0,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,elite=elite))

    def step(self, state):
        T=self.config.max_steps or 1; t=state.step
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        moa=0.2+t*((1-0.2)/T); mop=1-((t**(1/self._alpha))/(T**(1/self._alpha)))
        e=2.2204e-16
        p=np.copy(pop); r1=np.random.rand(self._n,self.problem.dimension)
        r2=np.random.rand(self._n,self.problem.dimension); r3=np.random.rand(self._n,self.problem.dimension)
        u1=np.where(r1>moa,elite[:-1]/(mop+e)*((hi-lo)*self._mu+lo),elite[:-1])
        u2=np.where(r2<=0.5,u1*mop,u1-mop)
        u3=np.where(r3>0.5,u2-((hi-lo)*self._mu+lo),u2+((hi-lo)*self._mu+lo))
        up=np.clip(u3,lo,hi)
        for i in range(self._n):
            nf=self.problem.evaluate(up[i,:]); 
            if nf<pop[i,-1]: p[i,:-1]=up[i,:]; p[i,-1]=nf
        bi=p[p[:,-1].argsort()][0,:]
        if bi[-1]<elite[-1]: elite=bi.copy()
        state.step+=1; state.evaluations+=self._n; state.payload=dict(population=p,elite=elite)
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
