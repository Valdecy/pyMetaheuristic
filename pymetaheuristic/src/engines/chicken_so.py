"""pyMetaheuristic src — Chicken Swarm Optimization Engine"""
from __future__ import annotations
import random, numpy as np
from scipy.stats import norm
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CHICKEN_SOEngine(BaseEngine):
    algorithm_id   = "chicken_so"
    algorithm_name = "Chicken Swarm Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/978-3-319-11857-4_10"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=10, g=5, rooster_ratio=0.2, hen_ratio=0.6)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["size"]); self._g=int(p["g"])
        self._rr=float(p["rooster_ratio"]); self._hr=float(p["hen_ratio"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); bi=np.argmin(pop[:,-1]); best=pop[bi,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=best[:-1].tolist(),best_fitness=float(best[-1]),
            initialized=True,payload=dict(population=pop,best=best,
                roosters=pop[:1,:],hens=pop[:1,:],chicks=pop[:1,:],
                h_rooster={},c_hen={}))

    def step(self, state):
        pl=state.payload; pop=pl["population"]; best=pl["best"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        t=state.step; evals=0
        if t%self._g==0:
            pop=pop[pop[:,-1].argsort()]
            nr=int(self._n*self._rr); nh=int(self._n*self._hr)
            roosters=pop[:nr,:].copy(); hens=pop[nr:nr+nh,:].copy(); chicks=pop[nr+nh:,:].copy()
            h_rooster={nr+h:random.choice(range(nr)) for h in range(nh)}
            c_hen={nr+nh+c:random.choice(range(nr,nr+nh)) for c in range(len(chicks))}
        else:
            roosters=pl["roosters"]; hens=pl["hens"]; chicks=pl["chicks"]
            h_rooster=pl["h_rooster"]; c_hen=pl["c_hen"]
        e=1e-9; old=np.copy(pop); cut=self._n
        for i in range(cut):
            if random.random()<0.1:
                pop[i,:-1]=np.random.uniform(lo,hi); pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1; continue
            row=pop[i,:]
            in_roosters=any(np.all(row[:-1]==r[:-1]) for r in roosters)
            in_hens=any(np.all(row[:-1]==h[:-1]) for h in hens)
            for j in range(self.problem.dimension):
                if in_roosters:
                    delta=best[-1]-pop[i,-1]
                    ss=np.exp(delta/(abs(pop[i,-1])+e)) if pop[i,-1]<best[-1] and abs(delta/(abs(pop[i,-1])+e))<100 else 1
                    pop[i,j]=np.clip(pop[i,j]+norm.rvs(scale=ss),lo[j],hi[j])
                elif in_hens:
                    r1=h_rooster.get(i,random.choice(list(h_rooster.values()) if h_rooster else [0]))
                    r2=random.randint(0,cut-1)
                    delta=pop[i,-1]-pop[r1,-1]
                    s1=np.exp(delta/(abs(pop[i,-1])+e)) if abs(delta/(abs(pop[i,-1])+e))<100 else 0
                    pop[i,j]=np.clip(pop[i,j]+s1*random.random()*(pop[r1,j]-pop[i,j])+random.random()*(pop[r2,j]-pop[i,j]),lo[j],hi[j])
                else:
                    m=c_hen.get(i,random.choice(list(c_hen.values()) if c_hen else [0]))
                    pop[i,j]=np.clip(pop[i,j]+random.uniform(0.5,0.9)*(pop[m,j]-pop[i,j]),lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        combined=np.vstack([np.unique(old,axis=0),np.unique(pop,axis=0)])
        combined=combined[combined[:,-1].argsort()][:cut,:]
        bi=np.argmin(combined[:,-1])
        if combined[bi,-1]<best[-1]: best=combined[bi,:].copy()
        state.step+=1; state.evaluations+=evals
        state.payload=dict(population=combined,best=best,roosters=roosters,hens=hens,
            chicks=chicks,h_rooster=h_rooster,c_hen=c_hen)
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
