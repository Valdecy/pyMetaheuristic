"""pyMetaheuristic src — Monarch Butterfly Optimization Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class MBOEngine(BaseEngine):
    algorithm_id   = "mbo"
    algorithm_name = "Monarch Butterfly Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-015-1923-y"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=15, ratio=0.4167, phi=1.2, adj_rate=0.4167, walk_size=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._ratio=float(p["ratio"]); self._phi=float(p["phi"]); self._ar=float(p["adj_rate"]); self._ws=float(p["walk_size"])
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
        div=int(np.ceil(self._n*5/12)); sp=np.argsort(pop[:,-1])
        r1=pop[sp[:div],:].copy(); r2=pop[sp[div:],:].copy()
        # migration
        nc1,nc2=r1.shape[0],r1.shape[1]-1
        mask=(np.random.rand(nc1,nc2)<self._ratio*self._phi).flatten()
        kidx=np.random.randint(0,r2.shape[0],nc1*nc2)
        rf=r1[:,:nc2].flatten(); r2f=r2[:,:nc2].flatten()
        tmp=np.random.randint(0,max(1,r2.shape[0]),nc1*nc2)
        full=r2[:,: nc2].flatten()
        full2=full[tmp]; rf[mask]=full2[mask]; r1[:,:nc2]=rf.reshape(nc1,nc2)
        # adjustment
        al=self._ws/((t+1)**2); stp=int(np.ceil(np.random.exponential(2*T)))
        nr2,nc2b=r2.shape[0],r2.shape[1]-1
        r2f2=r2[:,:nc2b].flatten(); rm=(np.random.rand(nr2,nc2b)<self._ratio).flatten()
        for i2 in range(nr2*nc2b):
            if rm[i2]: r2f2[i2]=elite[i2%nc2b]
        for i2 in range(nr2):
            for j2 in range(nc2b):
                if not rm[i2*nc2b+j2]:
                    k2=np.random.randint(nr2); r2f2[i2*nc2b+j2]=r2[k2,j2]
                    if np.random.rand()>self._ar:
                        r2f2[i2*nc2b+j2]+=al*(np.tan(np.pi*np.random.rand())-0.5)*stp
        r2[:,:nc2b]=r2f2.reshape(nr2,nc2b)
        pop=np.vstack((r1,r2))
        pop[:,-1]=self._evaluate_population(pop[:,:-1]); evals+=self._n
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
