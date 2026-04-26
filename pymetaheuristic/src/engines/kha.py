"""pyMetaheuristic src — Krill Herd Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class KHAEngine(BaseEngine):
    algorithm_id   = "kha"
    algorithm_name = "Krill Herd Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2016.08.041"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(population_size=15, c_t=1.0)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["population_size"]); self._ct=np.clip(float(p["c_t"]),0,2)
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

    def initialize(self):
        pop=self._init_pop(); best=pop[np.argmin(pop[:,-1]),:].copy()
        worst=pop[np.argmax(pop[:,-1]),:].copy()
        pn=np.zeros_like(pop); pf=np.copy(pop); pd=np.zeros_like(pop)
        return EngineState(step=0,evaluations=self._n,
            best_position=best[:-1].tolist(),best_fitness=float(best[-1]),
            initialized=True,payload=dict(population=pop,best=best,worst=worst,pn=pn,pf=pf,pd=pd))

    def step(self, state):
        pl=state.payload; pop=pl["population"]; best=pl["best"]; worst=pl["worst"]
        pn=pl["pn"]; pf=pl["pf"]; pd=pl["pd"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        T=self.config.max_steps or 1; c=state.step/T; dim=self.problem.dimension; evals=0
        # motion induced
        a2=pop[:,:-1]; b2=a2.reshape(np.prod(a2.shape[:-1]),1,a2.shape[-1])
        dtm=np.sqrt(np.einsum("ijk,ijk->ij",b2-a2,b2-a2)).squeeze()
        np.fill_diagonal(dtm,float("+inf"))
        n_max=0.01
        for i in range(self._n):
            idx=np.argpartition(dtm[i,:],int(dim/5))[:int(dim/5)]
            kij=np.sum((pn[i,-1]-pn[idx,-1])/(-best[-1]+worst[-1]+1e-16))
            xij=np.sum((pn[idx,:-1]-pn[i,:-1])/(dtm[i,idx][:,np.newaxis]+1e-16),axis=0)
            aiL=kij*xij; r=np.random.rand()
            ai=aiL+2*(r+c)*best[-1]*best[:-1]
            wv=np.random.rand(dim)
            pn[i,:-1]=np.clip(ai*n_max+wv*pn[i,:-1],lo,hi)
            pn[i,-1]=self.problem.evaluate(pn[i,:-1]); evals+=1
        # foraging
        xf=best[:-1]*(1/(best[-1]+1e-16))/(1/(best[-1]+1e-16)); kf=self.problem.evaluate(xf); evals+=1
        cf2=2*(1-c); bi2=cf2*kf*xf+best[-1]*best[:-1]; vf=0.02
        for i in range(self._n):
            wv=np.random.rand(dim)
            pf[i,:-1]=np.clip(vf*bi2+wv*pf[i,:-1],lo,hi)
            pf[i,-1]=self.problem.evaluate(pf[i,:-1]); evals+=1
        # diffusion
        dm=np.random.uniform(0.002,0.010,(self._n,1)); rd=np.random.uniform(-1,1,(self._n,dim))
        pd[:,:-1]=np.clip(dm*rd*(1-c),lo,hi)
        for i in range(self._n): pd[i,-1]=self.problem.evaluate(pd[i,:-1]); evals+=1
        # position
        dt=self._ct*np.sum(hi-lo); old=np.copy(pop)
        pop[:,:-1]=np.clip(pop[:,:-1]+dt*(pn[:,:-1]+pf[:,:-1]+pd[:,:-1]),lo,hi)
        pop[:,-1]=self._evaluate_population(pop[:,:-1]); evals+=self._n
        combined=np.vstack([old,pop,pn,pf,pd]); combined=combined[combined[:,-1].argsort()]
        pop=combined[:self._n,:]
        # GA step 20%
        if np.random.rand()<=0.2:
            fit=self._fitness_fn(pop); off=np.copy(pop); mu=1; el=1
            preserve=pop[pop[:,-1].argsort()]
            for i in range(el): off[i,:]=preserve[i,:]
            for i in range(el,self._n):
                p1=self._roulette(fit); p2=self._roulette(fit)
                while p1==p2: p2=np.random.choice(self._n-1)
                for j in range(dim):
                    r=np.random.rand(); rb=np.random.rand(); rc=np.random.rand()
                    b=2*rb**(1/(mu+1)) if r<=0.5 else (1/(2*(1-rb)))**(1/(mu+1))
                    if rc>=0.5: off[i,j]=np.clip(((1+b)*pop[p1,j]+(1-b)*pop[p2,j])/2,lo[j],hi[j])
                    else: off[i,j]=np.clip(((1-b)*pop[p1,j]+(1+b)*pop[p2,j])/2,lo[j],hi[j])
                off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
            for i in range(self._n):
                for j in range(dim):
                    if np.random.rand()<0.05:
                        r=np.random.rand(); rd2=np.random.rand()
                        d=2*rd2**(1/(1+1))-1 if r<=0.5 else 1-2*(1-rd2)**(1/(1+1))
                        off[i,j]=np.clip(off[i,j]+d,lo[j],hi[j])
                off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
            pop=off
        bl=pop[np.argmin(pop[:,-1]),:]; wl=pop[np.argmax(pop[:,-1]),:]
        if bl[-1]<best[-1]: best=bl.copy()
        state.step+=1; state.evaluations+=evals
        state.payload=dict(population=pop,best=best,worst=wl,pn=pn,pf=pf,pd=pd)
        if self.problem.is_better(float(best[-1]),state.best_fitness):
            state.best_fitness=float(best[-1]); state.best_position=best[:-1].tolist()
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
