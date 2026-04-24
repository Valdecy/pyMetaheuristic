"""pyMetaheuristic src — Grasshopper Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class GOAEngine(BaseEngine):
    algorithm_id   = "goa"
    algorithm_name = "Grasshopper Optimization Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(grasshoppers=25, c_min=0.00004, c_max=1.0, F=0.5, L=1.5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["grasshoppers"])
        self._cmin=float(p["c_min"]); self._cmax=float(p["c_max"]); self._F=float(p["F"]); self._L=float(p["L"])
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
        C=self._cmax-t*((self._cmax-self._cmin)/T)
        a=pop[:,:-1]; b=a.reshape(np.prod(a.shape[:-1]),1,a.shape[-1])
        dm=np.sqrt(np.einsum("ijk,ijk->ij",b-a,b-a)).squeeze()
        dm2=2*(dm-np.min(dm))/(np.ptp(dm)+1e-8)+1; np.fill_diagonal(dm2,0)
        for j in range(self.problem.dimension):
            sg=np.zeros(self._n)
            for i in range(self._n):
                sv=self._F*np.exp(-dm2[:,i]/self._L)-np.exp(-dm2[:,i])
                dn=np.where(dm2[:,i]==0,1,dm2[:,i])
                sg[i]=np.sum(C*((self.problem.max_values[j]-self.problem.min_values[j])/2)*sv*(pop[:,j]-pop[i,j])/dn)
            pop[:,j]=np.clip(C*sg+elite[j],lo[j],hi[j])
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
