"""pyMetaheuristic src — Improved Whale Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class I_WOAEngine(BaseEngine):
    algorithm_id   = "i_woa"
    algorithm_name = "Improved Whale Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.jcde.2019.02.002"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(hunting_party=25, spiral_param=1, mu=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["hunting_party"]); self._sp=float(p["spiral_param"]); self._mu=float(p["mu"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
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
        off=np.copy(pop); lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values); evals=0
        for i in range(self._n):
            p1=self._roulette(fit); p2=self._roulette(fit)
            while p1==p2: p2=np.random.choice(self._n-1)
            for j in range(self.problem.dimension):
                r=np.random.rand(); rb=np.random.rand(); rc=np.random.rand()
                b=2*rb**(1/(self._mu+1)) if r<=0.5 else (1/(2*(1-rb)))**(1/(self._mu+1))
                if rc>=0.5: off[i,j]=np.clip(((1+b)*pop[p1,j]+(1-b)*pop[p2,j])/2,lo[j],hi[j])
                else: off[i,j]=np.clip(((1-b)*pop[p1,j]+(1+b)*pop[p2,j])/2,lo[j],hi[j])
            off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
        return off,evals

    def initialize(self):
        pop=self._init_pop(); leader=pop[pop[:,-1].argsort()][0,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=leader[:-1].tolist(),best_fitness=float(leader[-1]),
            initialized=True,payload=dict(population=pop,leader=leader))

    def step(self, state):
        pop=state.payload["population"]; leader=state.payload["leader"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        T=self.config.max_steps or 1; t=state.step
        al=2-t*(2/T); bl=-1+t*(-1/T)
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<leader[-1]: leader=pop[bi,:].copy()
        for i in range(self._n):
            r1=np.random.rand(); r2=np.random.rand(); A=2*al*r1-al; C=2*r2; p=np.random.rand()
            for j in range(self.problem.dimension):
                if p<0.5:
                    if abs(A)>=1:
                        ri=np.random.randint(self._n); xr=pop[ri,:]
                        pop[i,j]=np.clip(xr[j]-A*abs(C*xr[j]-pop[i,j]),lo[j],hi[j])
                    else:
                        pop[i,j]=np.clip(leader[j]-A*abs(C*leader[j]-pop[i,j]),lo[j],hi[j])
                else:
                    d=abs(leader[j]-pop[i,j]); r=np.random.rand()
                    m=(bl-1)*r+1
                    pop[i,j]=np.clip(d*np.exp(self._sp*m)*np.cos(m*2*np.pi)+leader[j],lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1])
        fit=self._fitness_fn(pop); pop,e=self._breed(pop,fit)
        bi2=np.argmin(pop[:,-1])
        if pop[bi2,-1]<leader[-1]: leader=pop[bi2,:].copy()
        state.step+=1; state.evaluations+=self._n+e; state.payload=dict(population=pop,leader=leader)
        if self.problem.is_better(float(leader[-1]),state.best_fitness):
            state.best_fitness=float(leader[-1]); state.best_position=leader[:-1].tolist()
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
