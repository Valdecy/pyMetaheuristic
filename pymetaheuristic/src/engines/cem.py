"""pyMetaheuristic src — Cross Entropy Method Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class CEMEngine(BaseEngine):
    algorithm_id   = "cem"
    algorithm_name = "Cross Entropy Method"
    family         = "distribution"
    _REFERENCE     = {"doi": "10.1007/978-1-4757-4321-0"}
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

    def _objective_improvement(self, before, after):
        """Return objective-consistent improvement for EvoMapX telemetry."""
        if before is None or after is None:
            return 0.0
        before = float(before)
        after = float(after)
        if self.problem.objective == "max":
            return max(0.0, after - before)
        return max(0.0, before - after)

    def step(self, state):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        evals=0
        mn=state.payload["mean"]; sd=state.payload["std"]
        prev_best = None if state.best_fitness is None else float(state.best_fitness)
        prev_mean = np.array(mn, dtype=float).copy()
        prev_std = np.array(sd, dtype=float).copy()
        guess=state.payload["population"]
        gs=guess[guess[:,-1].argsort()].copy()

        # CEM native operators: preserve elites, sample from the current
        # distribution, select the elite set, then update the distribution.
        ns=np.random.normal(mn,sd,(self._n-self._ks,self.problem.dimension))
        ns=np.clip(ns,lo,hi)
        if self._n > self._ks:
            gs[self._ks:,:-1]=ns
        gs[:,-1]=self._evaluate_population(gs[:,:-1]); evals+=self._n

        sorted_idx = np.argsort(gs[:,-1]) if self.problem.objective == "min" else np.argsort(gs[:,-1])[::-1]
        top=gs[sorted_idx[:self._ks],:-1]
        best_idx = int(sorted_idx[0])
        best_iter_fitness = float(gs[best_idx, -1])
        if self._n > self._ks:
            sampled_fitness = gs[self._ks:, -1]
            best_sampled = float(np.min(sampled_fitness) if self.problem.objective == "min" else np.max(sampled_fitness))
        else:
            best_sampled = None

        mn=self._lr*mn+(1-self._lr)*np.mean(top,axis=0)
        sd=self._lr*sd+(1-self._lr)*np.std(top,axis=0); sd[sd<0.005]=3

        pop=gs; state.payload["mean"]=mn; state.payload["std"]=sd
        bi = int(np.argmin(pop[:,-1]) if self.problem.objective == "min" else np.argmax(pop[:,-1]))
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()

        range_norm = float(np.linalg.norm(hi - lo)) or 1.0
        mean_shift = float(np.linalg.norm(np.asarray(mn) - prev_mean) / range_norm)
        std_before = float(np.linalg.norm(prev_std) / range_norm)
        std_after = float(np.linalg.norm(np.asarray(sd)) / range_norm)
        std_contraction = max(0.0, std_before - std_after)
        operator_contributions = {
            "cem_sampling": self._objective_improvement(prev_best, best_sampled),
            "cem_elite_selection": self._objective_improvement(prev_best, best_iter_fitness),
            # Distribution update is a state-space adaptation rather than a direct
            # new objective evaluation. It is reported separately in metadata and
            # kept as zero contribution in OAM/CDS to avoid mixing fitness gains
            # with scale-change diagnostics.
            "cem_distribution_update": 0.0,
        }
        evomapx = {
            "operator_contributions": operator_contributions,
            "operator": max(operator_contributions, key=operator_contributions.get),
            "cem_best_sampled_fitness": best_sampled,
            "cem_best_iteration_fitness": best_iter_fitness,
            "cem_mean_shift": mean_shift,
            "cem_std_contraction": std_contraction,
            "cem_elite_size": int(self._ks),
            "cem_sampled_size": int(max(0, self._n - self._ks)),
        }

        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite,mean=mn,std=sd,evomapx=evomapx)
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
            **dict(state.payload.get("evomapx", {}) or {}),
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
