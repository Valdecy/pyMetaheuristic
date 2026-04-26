"""pyMetaheuristic src — Bat Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class BATAEngine(BaseEngine):
    algorithm_id   = "bat_a"
    algorithm_name = "Bat Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/978-3-642-12538-6_6"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(swarm_size=50, alpha=0.9, gama=0.9, fmin=0.0, fmax=10.0)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["swarm_size"]); self._alpha=float(p["alpha"])
        self._gama=float(p["gama"]); self._fmin=float(p["fmin"]); self._fmax=float(p["fmax"])
        if config.seed is not None: np.random.seed(config.seed)

    def initialize(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        pop=np.hstack((pos,fit[:,np.newaxis]))
        vel=np.zeros((self._n,self.problem.dimension))
        freq=np.zeros((self._n,1)); rate=np.random.rand(self._n,1)
        loud=np.random.uniform(1,2,(self._n,1))
        best=pop[pop[:,-1].argsort()][0,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=best[:-1].tolist(),best_fitness=float(best[-1]),
            initialized=True,payload=dict(population=pop,velocity=vel,frequency=freq,rate=rate,loudness=loud,best=best))

    def step(self, state):
        pl=state.payload; pop=pl["population"]; vel=pl["velocity"]
        freq=pl["frequency"]; rate=pl["rate"]; loud=pl["loudness"]; best=pl["best"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        t=state.step; pop2=np.zeros_like(pop)
        beta=np.random.rand(self._n)
        rand=np.random.rand(self._n); rand_pu=np.random.rand(self._n)
        freq[:,0]=self._fmin+(self._fmax-self._fmin)*beta
        vel=vel+(pop[:,:-1]-best[:-1])*freq
        pop2[:,:-1]=np.clip(pop[:,:-1]+vel,lo,hi)
        for i in range(self._n):
            pop2[i,-1]=self.problem.evaluate(pop2[i,:-1])
            if rand[i]>rate[i,0]:
                lm=loud.mean()
                rs=np.random.uniform(-1,1,self.problem.dimension)*lm
                pop2[i,:-1]=np.clip(best[:-1]+rs,lo,hi)
                pop2[i,-1]=self.problem.evaluate(pop2[i,:-1])
            else:
                pop2[i,:-1]=np.random.uniform(lo,hi,self.problem.dimension); pop2[i,-1]=self.problem.evaluate(pop2[i,:-1])
            if rand_pu[i]<loud[i,0] and pop2[i,-1]<=pop[i,-1]:
                pop[i,:]=pop2[i,:]
                rate[i,0]=np.random.rand()*(1-np.exp(-self._gama*t))
                loud[i,0]*=self._alpha
        combined=np.vstack([pop,pop2]); combined=combined[combined[:,-1].argsort()]
        pop=combined[:self._n,:]
        bi=np.argmin(pop[:,-1])
        if best[-1]>pop[bi,-1]: best=pop[bi,:].copy()
        state.step+=1; state.evaluations+=self._n*2
        state.payload=dict(population=pop,velocity=vel,frequency=freq,rate=rate,loudness=loud,best=best)
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
        vel = state.payload["velocity"]
        freq = state.payload["frequency"]
        rate = state.payload["rate"]
        loud = state.payload["loudness"]
        worst = np.argsort(pop[:, -1])[::-1]
        mean_rate = float(np.mean(rate)) if rate.size else 0.5
        mean_loud = float(np.mean(loud)) if loud.size else 1.0
        for j, cand in enumerate(candidates):
            wi = int(worst[j % len(worst)])
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            pop[wi, :-1] = pos; pop[wi, -1] = fit
            vel[wi, :] = 0.0
            freq[wi, 0] = self._fmin + np.random.rand() * (self._fmax - self._fmin)
            rate[wi, 0] = mean_rate
            loud[wi, 0] = mean_loud
            state.evaluations += 1
        best = pop[pop[:, -1].argsort()][0, :].copy()
        state.payload.update(dict(population=pop, velocity=vel, frequency=freq, rate=rate, loudness=loud, best=best))
        if self.problem.is_better(float(best[-1]), state.best_fitness):
            state.best_fitness = float(best[-1]); state.best_position = best[:-1].tolist()
        return state
