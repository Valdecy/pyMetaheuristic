"""pyMetaheuristic src — Flower Pollination Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class FPAEngine(BaseEngine):
    algorithm_id   = "fpa"
    algorithm_name = "Flower Pollination Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(flowers=25, gama=0.5, lamb=1.4, p=0.8)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["flowers"])
        self._gama=float(p["gama"]); self._lamb=float(p["lamb"]); self._p=float(p["p"])
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
        from scipy.special import gamma as gfunc
        r1=np.random.rand(self.problem.dimension); r2=np.random.rand(self.problem.dimension)
        sn=gfunc(1+self._lamb)*np.sin(np.pi*self._lamb/2)
        sd=gfunc((1+self._lamb)/2)*self._lamb*2**((self._lamb-1)/2)
        sigma=(sn/sd)**(1/self._lamb)
        levy=(0.01*r1*sigma)/(np.abs(r2)**(1/self._lamb))
        for i in range(self._n):
            nb1=np.random.randint(self._n); nb2=np.random.randint(self._n)
            while nb1==nb2: nb1=np.random.randint(self._n)
            r=np.random.rand()
            if r<self._p:
                x=np.copy(elite); x[:-1]=np.clip(pop[i,:-1]+self._gama*levy*(pop[i,:-1]-elite[:-1]),lo,hi); x[-1]=self.problem.evaluate(x[:-1]); evals+=1
            else:
                rr=np.random.rand(self.problem.dimension)
                x=np.copy(elite); x[:-1]=np.clip(pop[i,:-1]+rr*(pop[nb1,:-1]-pop[nb2,:-1]),lo,hi); x[-1]=self.problem.evaluate(x[:-1]); evals+=1
            if x[-1]<=pop[i,-1]: pop[i,:]=x
            val=pop[pop[:,-1].argsort()][0,:]
            if elite[-1]>val[-1]: elite=val.copy()
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
