"""pyMetaheuristic src — Gravitational Search Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class GSAEngine(BaseEngine):
    algorithm_id   = "gsa"
    algorithm_name = "Gravitational Search Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.ins.2009.03.004"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(swarm_size=200)

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
        vel=state.payload.get("vel",np.zeros((self._n,self.problem.dimension)))
        T=self.config.max_steps or 1; t=state.step
        G=100*np.exp(-20*(t/T))
        bt=pop[pop[:,-1].argsort()][0,:]; wt=pop[pop[:,-1].argsort()][-1,:]
        mass=(pop[:,-1]-wt[-1])/(bt[-1]-wt[-1]+1e-16)
        mass/=(np.sum(mass)+1e-16)
        kb=max(1,int(self._n-(self._n*(t/T))))
        a=pop[:,:-1]; b=a.reshape(np.prod(a.shape[:-1]),1,a.shape[-1])
        rij=np.sqrt(np.einsum("ijk,ijk->ij",b-a,b-a)).squeeze()
        fi=np.zeros((self._n,self.problem.dimension))
        for d in range(self.problem.dimension):
            fij=np.zeros((self._n,self._n))
            for i in range(self._n):
                for j in range(self._n):
                    if i!=j: fij[i,j]=G*((mass[i]*mass[j])/(rij[i,j]+2.2e-16))*(pop[j,d]-pop[i,d])
            for i in range(self._n):
                for j in range(kb): fi[i,d]+=np.random.rand()*fij[i,j]
        acc=np.nan_to_num(fi/(mass[:,np.newaxis]+1e-16))
        vel=np.random.rand(*vel.shape)*vel+acc
        old=np.copy(pop); npos=np.clip(pop[:,:-1]+vel,lo,hi)
        nfit=self._evaluate_population(npos); evals+=self._n
        imp=nfit<pop[:,-1]; pop[imp,:-1]=npos[imp]; pop[imp,-1]=nfit[imp]
        combined=np.vstack([pop,old]); combined=combined[combined[:,-1].argsort()]; pop=combined[:self._n,:]
        state.payload["vel"]=vel
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
