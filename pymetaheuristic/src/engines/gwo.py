"""pyMetaheuristic src — Grey Wolf Optimizer Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class GWOEngine(BaseEngine):
    algorithm_id   = "gwo"
    algorithm_name = "Grey Wolf Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2013.12.007"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(pack_size=5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["pack_size"])

        if config.seed is not None: np.random.seed(config.seed)

    def _objective_improvement(self, before, after):
        """Objective-consistent positive improvement used by EvoMapX."""
        if before is None or after is None:
            return 0.0
        before = float(before); after = float(after)
        if self.problem.objective == "max":
            return max(0.0, after - before)
        return max(0.0, before - after)

    def _ranked_indices(self, fitness):
        idx = np.argsort(fitness)
        if self.problem.objective == "max":
            idx = idx[::-1]
        return idx

    def _init_pop(self, n=None):
        if n is None: n = self._n
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); bi=self._ranked_indices(pop[:,-1])[0]
        elite=pop[bi,:].copy()

        return EngineState(step=0,evaluations=self._n,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,elite=elite,evomapx={}))

    def step(self, state):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        old_fit = pop[:,-1].copy()
        evals=0
        T=self.config.max_steps or 1; t=state.step; a=2-t*(2/T)
        idx=self._ranked_indices(pop[:,-1])
        alpha=pop[idx[0],:].copy(); beta=pop[idx[1],:].copy() if self._n>1 else alpha.copy()
        delta=pop[idx[2],:].copy() if self._n>2 else beta.copy()
        dim=self.problem.dimension
        r1=np.random.rand(self._n,dim); r2=np.random.rand(self._n,dim)
        A=2*a*r1-a; C=2*r2
        da=np.abs(C*alpha[:dim]-pop[:,:dim]); db=np.abs(C*beta[:dim]-pop[:,:dim]); dd=np.abs(C*delta[:dim]-pop[:,:dim])
        x1=np.clip(alpha[:dim]-A*da,lo,hi); x2=np.clip(beta[:dim]-A*db,lo,hi); x3=np.clip(delta[:dim]-A*dd,lo,hi)
        f1=self._evaluate_population(x1); evals+=self._n
        f2=self._evaluate_population(x2); evals+=self._n
        f3=self._evaluate_population(x3); evals+=self._n
        np_=np.clip((x1+x2+x3)/3,lo,hi)
        nf=self._evaluate_population(np_); evals+=self._n
        operator_contributions = {
            "gwo_alpha_guidance": float(sum(self._objective_improvement(b, a) for b, a in zip(old_fit, f1))),
            "gwo_beta_guidance": float(sum(self._objective_improvement(b, a) for b, a in zip(old_fit, f2))),
            "gwo_delta_guidance": float(sum(self._objective_improvement(b, a) for b, a in zip(old_fit, f3))),
            "gwo_position_averaging": float(sum(self._objective_improvement(b, a) for b, a in zip(old_fit, nf))),
        }
        new=np.hstack((np_,nf[:,np.newaxis]))
        combined=np.vstack([pop,new,np.hstack((x1,f1[:,np.newaxis])),np.hstack((x2,f2[:,np.newaxis])),np.hstack((x3,f3[:,np.newaxis]))])
        combined=combined[self._ranked_indices(combined[:,-1])]; pop=combined[:self._n,:]
        bi=self._ranked_indices(pop[:,-1])[0]
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()
        evomapx = {
            "operator_contributions": operator_contributions,
            "operator": max(operator_contributions, key=operator_contributions.get),
            "gwo_a": float(a),
            "gwo_alpha_fitness": float(alpha[-1]),
            "gwo_beta_fitness": float(beta[-1]),
            "gwo_delta_fitness": float(delta[-1]),
        }
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite,evomapx=evomapx)
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
