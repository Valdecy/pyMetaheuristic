"""pyMetaheuristic src — Symbiotic Organisms Search Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class SOSEngine(BaseEngine):
    algorithm_id   = "sos"
    algorithm_name = "Symbiotic Organisms Search"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.compstruc.2014.03.007"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(population_size=5, eta=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])
        self._eta=float(p["eta"])
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
        # mutualism
        xl=[i for i in range(self._n)]; pi=np.copy(pop); pj=np.copy(pop)
        for i in range(self._n):
            xl.remove(i); j=np.random.choice(xl); r1=np.random.rand(); r2=np.random.rand()
            mv=(pop[i,:]+pop[j,:])/2; b1=np.random.choice([1,2]); b2=np.random.choice([1,2])
            pi[i,:]=pop[i,:]+r1*(elite-mv*b1); pj[i,:]=pop[j,:]+r2*(elite-mv*b2)
            pi[i,-1]=self.problem.evaluate(pi[i,:-1]); pj[i,-1]=self.problem.evaluate(pj[i,:-1]); evals+=2
            xl.append(i)
        combined=np.vstack([pop,pi,pj]); combined=combined[combined[:,-1].argsort()]
        pop=combined[:self._n,:]
        # comensalism
        xl2=[i for i in range(self._n)]; pc=np.copy(pop)
        for i in range(self._n):
            xl2.remove(i); j=np.random.choice(xl2); r=np.random.uniform(-1,1)
            pc[i,:]=pop[i,:]+r*(elite-pop[j,:]); pc[i,-1]=self.problem.evaluate(pc[i,:-1]); evals+=1
            xl2.append(i)
        combined=np.vstack([pop,pc]); combined=combined[combined[:,-1].argsort()]; pop=combined[:self._n,:]
        # parasitism
        pp=np.copy(pop)
        for i in range(self._n):
            for j in range(self.problem.dimension):
                if np.random.rand()<1.0:
                    r=np.random.rand(); rd=np.random.rand()
                    d=2*rd**(1/(self._eta+1))-1 if r<=0.5 else 1-2*(1-rd)**(1/(self._eta+1))
                    pp[i,j]=np.clip(pp[i,j]+d,lo[j],hi[j])
            pp[i,-1]=self.problem.evaluate(pp[i,:-1]); evals+=1
        combined=np.vstack([pop,pp]); combined=combined[combined[:,-1].argsort()]; pop=combined[:self._n,:]
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
