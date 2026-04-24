"""pyMetaheuristic src — Pathfinder Algorithm Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class PFAEngine(BaseEngine):
    algorithm_id   = "pfa"
    algorithm_name = "Pathfinder Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(population_size=15)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])

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
        pf=state.payload.get("pathfinder",elite.copy())
        # update pathfinder
        u2=np.random.uniform(-1,1); r3=np.random.rand()
        A=u2*np.exp((-2*t)/T); lb=pop[np.argmin(pop[:,-1])]
        npf=pf.copy(); npf[:-1]=np.clip(pf[:-1]+2*r3*(pf[:-1]-lb[:-1])+A,lo,hi)
        npf[-1]=self.problem.evaluate(npf[:-1]); evals+=1
        if pf[-1]>npf[-1]: pf=npf.copy()
        # update positions
        a2=pop[:,:-1]; b2=a2.reshape(np.prod(a2.shape[:-1]),1,a2.shape[-1])
        dist=np.sqrt(np.einsum("ijk,ijk->ij",b2-a2,b2-a2)).squeeze()
        u1=np.random.uniform(-1,1)
        r1=np.random.rand(self._n**2,1); r2=np.random.rand(self._n**2,1)
        al=np.random.uniform(1,2,self._n**2).reshape(-1,1)
        be=np.random.uniform(1,2,self._n**2).reshape(-1,1)
        e=(1-(t/T))*u1*dist.flatten()
        pfe=np.tile(pf[:-1],(self._n**2,1)); pe=np.repeat(pop[:,:-1],self._n,axis=0)
        np2=pe+al*r1*(pe-pe)+be*r2*(pfe-pe)+e.reshape(-1,1)
        np2=np.clip(np2,lo,hi); nf=self._evaluate_population(np2); evals+=self._n**2
        np3=np.hstack((np2,nf.reshape(-1,1))); np3=np.vstack([pop,np3])
        np3=np3[np.argsort(np3[:,-1])][:self._n,:]; pop=np3
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<pf[-1]: pf=pop[bi,:].copy()
        state.payload["pathfinder"]=pf
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
