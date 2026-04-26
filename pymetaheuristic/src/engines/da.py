"""pyMetaheuristic src — Dragonfly Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class DAEngine(BaseEngine):
    algorithm_id   = "da"
    algorithm_name = "Dragonfly Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-015-1920-1"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=3)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["size"])
        beta=3/2
        sig_num=gamma(1+beta)*np.sin(np.pi*beta/2)
        sig_den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
        self._sigma=(sig_num/sig_den)**(1/beta); self._beta=beta
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self,n=None):
        if n is None: n=self._n
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); dl=self._init_pop()
        food=self._init_pop(1); enemy=self._init_pop(1)
        best=food[np.argmin(food[:,-1]),:].copy()
        return EngineState(step=0,evaluations=self._n*2+2,
            best_position=best[:-1].tolist(),best_fitness=float(best[-1]),
            initialized=True,payload=dict(population=pop,deltaflies=dl,food=food,enemy=enemy))

    def step(self, state):
        pl=state.payload; pop=pl["population"]; dl=pl["deltaflies"]
        food=pl["food"]; enemy=pl["enemy"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        T=self.config.max_steps or 1; t=state.step
        r=(hi-lo)/4+((hi-lo)*t/(T*2))
        w=0.9-t*((0.9-0.4)/T)
        mc=max(0.1-t*((0.1-0)/(T/2)),0)
        s=2*np.random.rand()*mc; a=2*np.random.rand()*mc; c=2*np.random.rand()*mc
        f=2*np.random.rand(); e=mc
        delta_max=(hi-lo)/10
        # update food & enemy
        bi=np.argmin(pop[:,-1]); wi=np.argmax(pop[:,-1])
        if pop[bi,-1]<food[0,-1]: food[0,:]=pop[bi,:].copy()
        if pop[wi,-1]>enemy[0,-1]: enemy[0,:]=pop[wi,:].copy()
        evals=0
        for i in range(self._n):
            nbs_d=[]; nbs_p=[]
            for j in range(self._n):
                d=np.sqrt(np.sum((pop[i,:-1]-pop[j,:-1])**2))
                if d>0 and (d<=r).all():
                    nbs_d.append(dl[j,:-1]); nbs_p.append(pop[j,:-1])
            A=np.mean(nbs_d,axis=0) if nbs_d else dl[i,:-1]
            C=np.mean(nbs_p,axis=0)-pop[i,:-1] if nbs_p else np.zeros(self.problem.dimension)
            S=-np.sum(np.array(nbs_p)-pop[i,:-1],axis=0) if nbs_p else np.zeros(self.problem.dimension)
            df=np.sqrt(np.sum((pop[i,:-1]-food[0,:-1])**2))
            de=np.sqrt(np.sum((pop[i,:-1]-enemy[0,:-1])**2))
            F=food[0,:-1]-pop[i,:-1] if (df<=r).all() else np.zeros(self.problem.dimension)
            E=enemy[0,:-1] if (de<=r).all() else np.zeros(self.problem.dimension)
            for k in range(self.problem.dimension):
                if (df>r).all():
                    if len(nbs_p)>1:
                        dl[i,k]=w*dl[i,k]+np.random.rand()*(a*A[k]+c*C[k]+s*S[k])
                    else:
                        lf=0.01*np.random.rand()*self._sigma/abs(np.random.rand())**(1/self._beta)
                        pop[i,:-1]+=lf*pop[i,:-1]
                        dl[i,k]=np.clip(dl[i,k],lo[k],hi[k]); break
                else:
                    dl[i,k]=(a*A[k]+c*C[k]+s*S[k]+f*F[k]+e*E[k])+w*dl[i,k]
                dl[i,k]=np.clip(dl[i,k],-delta_max[k],delta_max[k])
                pop[i,k]=np.clip(pop[i,k]+dl[i,k],lo[k],hi[k])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        bi2=np.argmin(pop[:,-1])
        if pop[bi2,-1]<food[0,-1]: food[0,:]=pop[bi2,:].copy()
        best=food[0,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,deltaflies=dl,food=food,enemy=enemy)
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


    def inject_candidates(self, state, candidates, policy="native"):
        pop = state.payload["population"]
        delta = state.payload["deltaflies"]
        worst = np.argsort(pop[:, -1])[::-1]
        for j, cand in enumerate(candidates):
            wi = int(worst[j % len(worst)])
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            pop[wi, :-1] = pos; pop[wi, -1] = fit
            delta[wi, :-1] = 0.0
            delta[wi, -1] = fit
            state.evaluations += 1
        ranked = pop[pop[:, -1].argsort()]
        food = ranked[0:1, :].copy()
        enemy = ranked[-1:, :].copy()
        state.payload.update(dict(population=pop, deltaflies=delta, food=food, enemy=enemy))
        if self.problem.is_better(float(food[0, -1]), state.best_fitness):
            state.best_fitness = float(food[0, -1]); state.best_position = food[0, :-1].tolist()
        return state
