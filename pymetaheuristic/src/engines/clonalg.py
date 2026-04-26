"""pyMetaheuristic src — Clonal Selection Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CLONALGEngine(BaseEngine):
    algorithm_id   = "clonalg"
    algorithm_name = "Clonal Selection Algorithm"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/TEVC.2002.1011539"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=200, clone_factor=0.1, num_rand=2)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._cf=float(p["clone_factor"]); self._nr=int(p["num_rand"])
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
        nc=int(self._n*self._cf); mr=np.exp(-2.5*pop[:,-1])
        clones=np.zeros((self._n*nc,pop.shape[1]))
        for i,ab in enumerate(pop):
            s=i*nc; e2=s+nc
            clones[s:e2,:-1]=np.tile(ab[:-1],(nc,1))
            mm=np.random.rand(nc,self.problem.dimension)<mr[i]
            rm=np.random.normal(0,1,(nc,self.problem.dimension))
            clones[s:e2,:-1]=np.clip(clones[s:e2,:-1]+mm*rm,lo,hi)
        clones[:,-1]=self._evaluate_population(clones[:,:-1]); evals+=self._n*nc
        pop=np.vstack([pop,clones]); pop=pop[np.argsort(pop[:,-1])][:self._n]
        if self._nr>0:
            nr=np.random.uniform(lo,hi,(self._nr,self.problem.dimension))
            nf=self._evaluate_population(nr); evals+=self._nr
            ni=np.hstack((nr,nf.reshape(-1,1))); pop=np.vstack([pop,ni])
            pop=pop[np.argsort(pop[:,-1])][:self._n]
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
