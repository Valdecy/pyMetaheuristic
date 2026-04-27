"""pyMetaheuristic src — Adaptive Random Search Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class ARSEngine(BaseEngine):
    algorithm_id   = "ars"
    algorithm_name = "Adaptive Random Search"
    family         = "trajectory"
    _REFERENCE     = {"doi": "10.1002/nav.20422"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(solutions=25, step_size_factor=0.05, factor_1=3.0, factor_2=1.5,
                     large_step_threshold=10, improvement_threshold=25)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["solutions"]); self._ssf=float(p["step_size_factor"])
        self._f1=float(p["factor_1"]); self._f2=float(p["factor_2"])
        self._lst=int(p["large_step_threshold"]); self._imt=int(p["improvement_threshold"])
        if config.seed is not None: np.random.seed(config.seed)

    def initialize(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        pop=np.hstack((pos,fit[:,np.newaxis]))
        ss=np.tile((hi-lo)*self._ssf,(self._n,1))
        bi=np.argmin(pop[:,-1])
        return EngineState(step=0,evaluations=self._n,
            best_position=pop[bi,:-1].tolist(),best_fitness=float(pop[bi,-1]),
            initialized=True,payload=dict(population=pop,step_size=ss,threshold=np.zeros(self._n)))

    def step(self, state):
        pop=state.payload["population"]; ss=state.payload["step_size"]; thr=state.payload["threshold"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        t=state.step
        # small step
        rand=np.random.rand(self._n,self.problem.dimension)
        mn=np.clip(pop[:,:-1]-ss,lo,hi); mx=np.clip(pop[:,:-1]+ss,lo,hi)
        ps=np.clip(mn+(mx-mn)*rand,lo,hi)
        fs=self._evaluate_population(ps)
        pop_step=np.hstack((ps,fs[:,np.newaxis]))
        # large step
        factor=self._f1 if t>0 and t%self._lst==0 else self._f2
        adj=ss*factor
        rand2=np.random.rand(self._n,self.problem.dimension)
        mn2=np.clip(pop[:,:-1]-adj,lo,hi); mx2=np.clip(pop[:,:-1]+adj,lo,hi)
        pl=np.clip(mn2+(mx2-mn2)*rand2,lo,hi)
        fl=self._evaluate_population(pl)
        pop_large=np.hstack((pl,fl[:,np.newaxis]))
        # update
        for i in range(self._n):
            if pop_step[i,-1]<pop[i,-1] or pop_large[i,-1]<pop[i,-1]:
                if pop_large[i,-1]<pop_step[i,-1]:
                    pop[i,:]=pop_large[i,:]; ss[i,:]=adj[i,:]
                else:
                    pop[i,:]=pop_step[i,:]
                thr[i]=0
            else:
                thr[i]+=1
                if thr[i]>=self._imt: thr[i]=0; ss[i,:]/=self._f2
        bi=np.argmin(pop[:,-1])
        state.step+=1; state.evaluations+=self._n*2
        state.payload=dict(population=pop,step_size=ss,threshold=thr)
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


    def inject_candidates(self, state, candidates, policy="native"):
        pop = state.payload["population"]
        ss = state.payload["step_size"]
        thr = state.payload["threshold"]
        worst = np.argsort(pop[:, -1])[::-1]
        ss_template = np.median(ss, axis=0) if ss.size else (np.array(self.problem.max_values) - np.array(self.problem.min_values)) * self._ssf
        thr_template = float(np.median(thr)) if thr.size else 0.0
        for j, cand in enumerate(candidates):
            wi = int(worst[j % len(worst)])
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            pop[wi, :-1] = pos; pop[wi, -1] = fit
            ss[wi, :] = ss_template
            thr[wi] = thr_template
            state.evaluations += 1
        bi = np.argmin(pop[:, -1])
        state.payload["population"] = pop
        state.payload["step_size"] = ss
        state.payload["threshold"] = thr
        if self.problem.is_better(float(pop[bi, -1]), state.best_fitness):
            state.best_fitness = float(pop[bi, -1]); state.best_position = pop[bi, :-1].tolist()
        return state
