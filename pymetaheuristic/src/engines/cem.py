"""pyMetaheuristic src — Cross Entropy Method Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CEMEngine(BaseEngine):
    algorithm_id   = "cem"
    algorithm_name = "Cross Entropy Method"
    family         = "distribution"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(size=5, learning_rate=0.7, k_samples=2)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["size"])
        self._lr=float(p["learning_rate"]); self._ks=int(p["k_samples"])
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
        mn=np.mean(pop[:,:-1],axis=0).reshape(1,-1)
        sd=np.std(pop[:,:-1],axis=0).reshape(1,-1)
        return EngineState(step=0,evaluations=self._n,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,elite=elite,mean=mn,std=sd))

    def step(self, state):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        evals=0
        mn=state.payload["mean"]; sd=state.payload["std"]
        guess=state.payload["population"]
        gs=guess[guess[:,-1].argsort()].copy()
        ns=np.random.normal(mn,sd,(self._n-self._ks,self.problem.dimension))
        ns=np.clip(ns,lo,hi)
        gs[self._ks:,:-1]=ns
        gs[:,-1]=self._evaluate_population(gs[:,:-1]); evals+=self._n
        top=gs[np.argsort(gs[:,-1])[:self._ks],:-1]
        mn=self._lr*mn+(1-self._lr)*np.mean(top,axis=0)
        sd=self._lr*sd+(1-self._lr)*np.std(top,axis=0); sd[sd<0.005]=3
        pop=gs; state.payload["mean"]=mn; state.payload["std"]=sd
        bi=np.argmin(pop[:,-1])
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite,mean=mn,std=sd)
        if self.problem.is_better(float(elite[-1]),state.best_fitness):
            state.best_fitness=float(elite[-1]); state.best_position=elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        fitness = pop[:, -1]
        pos = pop[:, :-1]
        lo = np.array(self.problem.min_values)
        hi = np.array(self.problem.max_values)
        denom = np.linalg.norm(hi - lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        return dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            mean_fitness=float(np.mean(fitness)),
            std_fitness=float(np.std(fitness)),
            diversity=diversity,
            cem_mean=state.payload["mean"].tolist() if hasattr(state.payload.get("mean", None), "tolist") else None,
            cem_std=state.payload["std"].tolist() if hasattr(state.payload.get("std", None), "tolist") else None,
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
        worst = np.argsort(pop[:, -1])[::-1]
        for j, cand in enumerate(candidates):
            wi = int(worst[j % len(worst)])
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            pop[wi, :-1] = pos; pop[wi, -1] = fit
            state.evaluations += 1
        ranked = pop[pop[:, -1].argsort()]
        elite = ranked[0, :].copy()
        elite_set = ranked[:max(1, min(self._ks, ranked.shape[0])), :-1]
        mean = elite_set.mean(axis=0, keepdims=True)
        std = np.maximum(elite_set.std(axis=0, keepdims=True), 1e-12)
        state.payload.update(dict(population=pop, elite=elite, mean=mean, std=std))
        if self.problem.is_better(float(elite[-1]), state.best_fitness):
            state.best_fitness = float(elite[-1]); state.best_position = elite[:-1].tolist()
        return state
