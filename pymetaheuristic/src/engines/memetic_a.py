"""pyMetaheuristic src — Memetic Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class MEMETIC_AEngine(BaseEngine):
    algorithm_id   = "memetic_a"
    algorithm_name = "Memetic Algorithm"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1162/evco.1991.1.1.67"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(population_size=5, mutation_rate=0.1, elite=1, eta=1, mu=1, std=0.1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["population_size"]); self._mr=float(p["mutation_rate"])
        self._elite=int(p["elite"]); self._eta=float(p["eta"]); self._mu=float(p["mu"]); self._std=float(p["std"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self, seed=None):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        if seed is not None:
            s=np.atleast_2d(seed)
            nr=self._n-s.shape[0]
            if nr>0: rows=np.random.uniform(lo,hi,(nr,self.problem.dimension)); pos=np.vstack((s[:,:self.problem.dimension],rows))
            else: pos=s[:self._n,:self.problem.dimension]
        else: pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def _fitness_fn(self,pop):
        fc=1/(1+pop[:,-1]+abs(pop[:,-1].min())); cs=np.cumsum(fc); cs/=cs[-1]
        return np.column_stack((fc,cs))
    def _roulette(self,fit):
        r=np.random.rand()
        for i in range(fit.shape[0]):
            if r<=fit[i,1]: return i
        return fit.shape[0]-1

    def _breed(self,pop,fit):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        off=np.copy(pop); evals=0
        if self._elite>0:
            prs=pop[pop[:,-1].argsort()]
            for i in range(self._elite): off[i,:]=prs[i,:]
        for i in range(self._elite,self._n):
            p1=self._roulette(fit); p2=self._roulette(fit)
            while p1==p2: p2=np.random.choice(self._n-1)
            for j in range(self.problem.dimension):
                r=np.random.rand(); rb=np.random.rand(); rc=np.random.rand()
                b=2*rb**(1/(self._mu+1)) if r<=0.5 else (1/(2*(1-rb)))**(1/(self._mu+1))
                if rc>=0.5: off[i,j]=np.clip(((1+b)*pop[p1,j]+(1-b)*pop[p2,j])/2,lo[j],hi[j])
                else: off[i,j]=np.clip(((1-b)*pop[p1,j]+(1+b)*pop[p2,j])/2,lo[j],hi[j])
            off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
        return off,evals

    def _mutate(self,off):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values); evals=0
        for i in range(self._n):
            ch=False
            for j in range(self.problem.dimension):
                if np.random.rand()<self._mr:
                    r=np.random.rand(); rd=np.random.rand()
                    d=2*rd**(1/(self._eta+1))-1 if r<=0.5 else 1-2*(1-rd)**(1/(self._eta+1))
                    off[i,j]=np.clip(off[i,j]+d,lo[j],hi[j]); ch=True
            if ch: off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
        return off,evals

    def _xhc(self,off,fit):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values); evals=0
        for _ in range(off.shape[0]):
            p1=self._roulette(fit); p2=self._roulette(fit)
            while p1==p2: p2=np.random.choice(self._n-1)
            xo=np.zeros((2,self.problem.dimension+1))
            for j in range(self.problem.dimension):
                r=np.random.rand(); rb=np.random.rand()
                b=2*rb**(1/(self._mu+1)) if r<=0.5 else (1/(2*(1-rb)))**(1/(self._mu+1))
                xo[0,j]=np.clip(((1+b)*off[p1,j]+(1-b)*off[p2,j])/2,lo[j],hi[j])
                xo[1,j]=np.clip(((1-b)*off[p1,j]+(1+b)*off[p2,j])/2,lo[j],hi[j])
            xo[0,-1]=self.problem.evaluate(xo[0,:-1]); xo[1,-1]=self.problem.evaluate(xo[1,:-1]); evals+=2
            if xo[1,-1]<xo[0,-1]: xo[0,:]=xo[1,:]
            if off[p1,-1]<off[p2,-1]:
                if xo[0,-1]<off[p1,-1]: off[p1,:]=xo[0,:]
            elif xo[0,-1]<off[p2,-1]: off[p2,:]=xo[0,:]
        return off,evals

    def initialize(self):
        pop=self._init_pop(); bi=np.argmin(pop[:,-1])
        return EngineState(step=0,evaluations=self._n,
            best_position=pop[bi,:-1].tolist(),best_fitness=float(pop[bi,-1]),
            initialized=True,payload=dict(population=pop,elite=pop[bi,:].copy()))

    def step(self, state):
        pop=state.payload["population"]; elite=state.payload["elite"]
        fit=self._fitness_fn(pop)
        off,e1=self._breed(pop,fit); pop,e2=self._mutate(off); pop,e3=self._xhc(pop,fit)
        if (pop[:,:-1].std()/len(self.problem.min_values))<self._std:
            pop=self._init_pop(seed=elite); e2+=self._n
        fit=self._fitness_fn(pop); bi=pop[pop[:,-1].argsort()][0,:]
        if bi[-1]<elite[-1]: elite=bi.copy()
        state.step+=1; state.evaluations+=e1+e2+e3; state.payload=dict(population=pop,elite=elite)
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
