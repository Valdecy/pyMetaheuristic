"""pyMetaheuristic src — Cat Swarm Optimization Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CAT_SOEngine(BaseEngine):
    algorithm_id   = "cat_so"
    algorithm_name = "Cat Swarm Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/978-3-540-36668-3_94"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=15, mixture_ratio=0.2, seeking_range=0.2, dim_change=2, c1=0.5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._mr=float(p["mixture_ratio"]); self._sr=float(p["seeking_range"])
        self._dc=int(p["dim_change"]); self._c1=float(p["c1"])
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
        dim=self.problem.dimension
        v=self._init_pop()  # velocities (reuse pop shape)
        c_sm=int((1-self._mr)*self._n)
        flag=[0 if i<c_sm else 1 for i in range(self._n)]; np.random.shuffle(flag)
        bi=np.argmin(pop[:,-1]); best_cat=pop[bi,:].copy()
        for i in range(self._n):
            if flag[i]==0:
                dm=np.random.choice([0,1],size=(self._n,dim),p=[1-self._dc/dim,self._dc/dim])
                sc=(2*np.random.rand(self._n,dim)-1)*self._sr
                copies=np.clip(pop[:,:-1]+dm*sc*pop[:,:-1],lo,hi)
                cf=self._evaluate_population(copies); evals+=self._n
                ic=np.argmin(cf)
                if cf[ic]<pop[i,-1]: pop[i,:-1]=copies[ic,:]; pop[i,-1]=cf[ic]
            else:
                v[i,:-1]=np.clip(v[i,:-1]+np.random.rand(dim)*self._c1*(best_cat[:-1]-pop[i,:-1]),-np.array(self.problem.max_values)*2,np.array(self.problem.max_values)*2)
                pop[i,:-1]=np.clip(pop[i,:-1]+v[i,:-1],lo,hi)
                pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        bi2=np.argmin(pop[:,-1])
        if pop[bi2,-1]<best_cat[-1]: best_cat=pop[bi2,:].copy()
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
