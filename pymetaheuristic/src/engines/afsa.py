"""pyMetaheuristic src — Artificial Fish Swarm Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class AFSAEngine(BaseEngine):
    algorithm_id   = "afsa"
    algorithm_name = "Artificial Fish Swarm Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-012-9342-2"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(school_size=25, attempts=100, visual=0.3, step=0.5, delta=0.5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n=int(p["school_size"]); self._att=int(p["attempts"])
        self._vis=float(p["visual"]); self._stp=float(p["step"]); self._delta=float(p["delta"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n, self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); bi=np.argmin(pop[:,-1])
        return EngineState(step=0,evaluations=self._n,
            best_position=pop[bi,:-1].tolist(),best_fitness=float(pop[bi,-1]),
            initialized=True,payload=dict(population=pop))

    def _prey(self,pop):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop2=np.copy(pop); evals=0
        for i in range(pop.shape[0]):
            for _ in range(self._att):
                k=np.random.choice([c for c in range(pop.shape[0]) if c!=i])
                rnd=np.random.uniform(-1,1,self.problem.dimension)
                np_=np.clip(pop[k,:-1]+self._vis*rnd,lo,hi)
                nf=self.problem.evaluate(np_); evals+=1
                if nf<=pop[i,-1]:
                    pop2[i,:-1]=np_; pop2[i,-1]=nf; break
        combined=np.vstack([pop,pop2]); combined=combined[combined[:,-1].argsort()]
        return combined[:pop.shape[0],:], evals

    def _swarm(self,pop):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        a=pop[:,:-1]; b=a.reshape(np.prod(a.shape[:-1]),1,a.shape[-1])
        dm=np.sqrt(np.einsum('ijk,ijk->ij',b-a,b-a)).squeeze()
        pop2=np.copy(pop); evals=0
        for i in range(pop.shape[0]):
            vis=np.where((dm[i,:]<=self._vis)&(np.arange(dm.shape[1])!=i))[0]
            if len(vis)==0: vis=np.array([c for c in range(pop.shape[0]) if c!=i])
            k=np.random.choice(vis)
            rnd=np.random.uniform(-1,1,self.problem.dimension)
            np_=np.clip(pop[k,:-1]+self._vis*rnd,lo,hi)
            nf=self.problem.evaluate(np_); evals+=1
            if nf<=pop[i,-1]: pop2[i,:-1]=np_; pop2[i,-1]=nf
        combined=np.vstack([pop,pop2]); combined=combined[combined[:,-1].argsort()]
        return combined[:pop.shape[0],:], evals

    def _follow(self,pop,best):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop2=np.copy(pop); evals=0
        rnd=np.random.uniform(-1,1,(pop.shape[0],self.problem.dimension))
        nps=np.clip(best[:-1]+self._vis*rnd,lo,hi)
        for i in range(pop.shape[0]):
            nf=self.problem.evaluate(nps[i]); evals+=1
            if nf<=pop[i,-1]: pop2[i,:-1]=nps[i]; pop2[i,-1]=nf
        combined=np.vstack([pop,pop2]); combined=combined[combined[:,-1].argsort()]
        return combined[:pop.shape[0],:], evals

    def _leap(self,pop):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        rnd=np.random.uniform(-1,1,(pop.shape[0],self.problem.dimension))
        nps=np.clip(pop[0,:-1]+self._vis*rnd,lo,hi)
        for i in range(pop.shape[0]):
            pop[i,:-1]=nps[i]; pop[i,-1]=self.problem.evaluate(nps[i])
        return pop, pop.shape[0]

    def step(self, state):
        pop=state.payload["population"]; best=pop[np.argmin(pop[:,-1])]
        evs=0
        pop,e=self._prey(pop);  evs+=e
        pop,e=self._swarm(pop); evs+=e
        pop,e=self._follow(pop,best); evs+=e
        pop,e=self._leap(pop); evs+=e
        if abs(pop[:,-1].mean()-best[-1])<=self._delta:
            pop,e=self._leap(pop); evs+=e
        bi=np.argmin(pop[:,-1])
        state.step+=1; state.evaluations+=evs; state.payload=dict(population=pop)
        if self.problem.is_better(float(pop[bi,-1]),state.best_fitness):
            state.best_fitness=float(pop[bi,-1]); state.best_position=pop[bi,:-1].tolist()
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
