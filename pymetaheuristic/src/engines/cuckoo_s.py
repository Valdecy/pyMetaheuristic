"""pyMetaheuristic src — Cuckoo Search Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CUCKOO_SEngine(BaseEngine):
    algorithm_id   = "cuckoo_s"
    algorithm_name = "Cuckoo Search"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/NABIC.2009.5393690"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(birds=3, discovery_rate=0.25, alpha_value=0.01, lambda_value=1.5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["birds"])
        self._dr=float(p["discovery_rate"]); self._av=float(p["alpha_value"]); self._lv=float(p["lambda_value"])
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
        mean=self._lv
        for i in range(self._n):
            rb=np.random.randint(self._n)
            u1=np.random.uniform(-0.5*np.pi,0.5*np.pi); u2=np.random.uniform(-0.5*np.pi,0.5*np.pi)
            v=np.random.uniform(); x1=np.sin((mean-1)*u1)/((np.cos(u1))**(1/(mean-1)))
            x2=(np.cos((2-mean)*u2)/(-np.log(v)))**((2-mean)/(mean-1)); lv=x1*x2
            ns=np.clip(pop[rb,:-1]+self._av*lv*pop[rb,:-1]*np.random.rand(self.problem.dimension),lo,hi)
            nf=self.problem.evaluate(ns); evals+=1
            if nf<pop[rb,-1]: pop[rb,:-1]=ns; pop[rb,-1]=nf
        # abandon worst nests
        ab=int(np.ceil(self._dr*self._n))+1
        nlist=np.argsort(pop[:,-1])[-ab:]
        bj=np.random.choice(self._n); bk=np.random.choice(self._n)
        while bk==bj: bk=np.random.choice(self._n)
        for i in nlist:
            r=np.random.rand(self.problem.dimension)
            if np.random.rand()>self._dr:
                pop[i,:-1]=np.clip(pop[i,:-1]+r*(pop[bj,:-1]-pop[bk,:-1]),lo,hi)
                pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        combined=np.vstack([pop,pop]); combined=combined[combined[:,-1].argsort()]
        pop=combined[:self._n,:]
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
