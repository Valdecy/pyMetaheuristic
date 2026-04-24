"""pyMetaheuristic src — Dynamic Virtual Bats Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class DVBAEngine(BaseEngine):
    algorithm_id   = "dvba"
    algorithm_name = "Dynamic Virtual Bats Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=20, wave_vectors=5, search_points=6, bats=20, beta=100)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._wv=int(p["wave_vectors"]); self._sp=int(p["search_points"]); self._bats=int(p["bats"]); self._beta=float(p["beta"])
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
        rho=(np.array(self.problem.max_values)-lo)/self._beta
        freq=state.payload.get("freq",np.random.uniform(0,2,(self._n,self.problem.dimension))*rho)
        v=state.payload.get("v",np.random.uniform(-1,1,(self._n,self.problem.dimension)))
        lam=state.payload.get("lam",np.random.uniform(0.01,10.0,(self._n,self.problem.dimension))*rho)
        rm=np.random.rand(self._n)<0.25
        if np.any(rm):
            pop[rm,:-1]=np.random.uniform(lo,hi,(rm.sum(),self.problem.dimension))
            pop[rm,-1]=self._evaluate_population(pop[rm,:-1]); evals+=rm.sum()
        for i in range(self._n):
            if rm[i]:
                for z in range(self._wv):
                    for _ in range(self._sp):
                        u=-1+2*np.random.rand()
                        A=self._bats*u/np.clip(freq[i],1e-9,np.inf)
                        V=v[i]+A; V=(V-lo)/(np.array(self.problem.max_values)-lo+1e-9)
                        H=np.clip(pop[i,:-1]+(z+1)*V*lam[i],lo,hi)
                        fH=self.problem.evaluate(H); evals+=1
                        if fH<pop[i,-1]:
                            pop[i,:-1]=H; pop[i,-1]=fH
                            nm=np.clip(np.linalg.norm(H-pop[i,:-1]),1e-9,np.inf)
                            v[i]=(H-pop[i,:-1])/nm; lam[i]-=(np.array(self.problem.max_values)-lo)*0.01; freq[i]+=(np.array(self.problem.max_values)-lo)*0.01
                        if fH>elite[-1]:
                            v[i]=-1+2*np.random.rand(); lam[i]+=(np.array(self.problem.max_values)-lo)*0.01; freq[i]-=(np.array(self.problem.max_values)-lo)*0.01
                        if fH<=pop[i,-1] and fH<elite[-1]:
                            pop[i,:-1]=H; lam[i]=lo; freq[i]=np.array(self.problem.max_values); v[i]=-1+2*np.random.rand()
        combined=np.vstack([pop,pop]); combined=combined[np.argsort(combined[:,-1])]; pop=combined[:self._n,:]
        state.payload["freq"]=freq; state.payload["v"]=v; state.payload["lam"]=lam
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
