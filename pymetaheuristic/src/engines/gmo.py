"""pyMetaheuristic src — Geometric Mean Optimizer Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class GMOEngine(BaseEngine):
    algorithm_id   = "gmo"
    algorithm_name = "Geometric Mean Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00500-023-08202-z"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=15, epsilon=0.001)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._eps=float(p["epsilon"])
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
        pb=state.payload.get("pb",np.copy(pop))
        pv=state.payload.get("pv",np.zeros((self._n,self.problem.dimension)))
        T=self.config.max_steps or 1; t=state.step; wt=1-(t/T)
        max_vel=0.1*(np.array(self.problem.max_values)-lo); min_vel=-max_vel
        # generate guide
        ave=np.mean(pb[:,-1]); std=np.std(pb[:,-1])
        nb=int(np.round(self._n-(self._n-2)*(t/T))); dual=np.zeros(self._n)
        for i in range(self._n):
            div1=std*np.sqrt(np.e); div2=pb[:,-1]-ave; div3=1-np.eye(self._n)[i]
            exp1=np.exp(np.clip(-4/(div1*div2+1e-16),-5,5))
            dual[i]=np.prod(1/(1+exp1*div3))
        idx2=np.argsort(dual)[::-1]; ss=np.sum(dual[idx2[:nb]])
        pg=np.zeros((self._n,self.problem.dimension))
        for i in range(self._n):
            for k in idx2[:nb]:
                if k!=i: pg[i]+=dual[k]/(ss+self._eps)*pb[k,:-1]
            pg[i]=np.clip(pg[i],lo,hi)
        # improve guide
        gf=self._evaluate_population(pg); evals+=self._n
        pg2=np.hstack((pg,gf[:,np.newaxis]))
        pg2=np.vstack([pg2,pop,pb]); pg2=pg2[pg2[:,-1].argsort()][:self._n,:-1]
        # update
        for i in range(self._n):
            mut=pg2[i]+wt*np.random.randn(self.problem.dimension)*(np.max(pg2,axis=0)-np.min(pg2,axis=0))
            pv[i]=wt*pv[i]+(1+(2*np.random.rand(self.problem.dimension)-1)*wt)*(np.clip(mut,lo,hi)-pop[i,:-1])
            pv[i]=np.clip(pv[i],min_vel,max_vel); pop[i,:-1]=np.clip(pop[i,:-1]+pv[i],lo,hi)
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
            if pop[i,-1]<pb[i,-1]: pb[i,:]=pop[i,:]
        state.payload["pb"]=pb; state.payload["pv"]=pv
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
