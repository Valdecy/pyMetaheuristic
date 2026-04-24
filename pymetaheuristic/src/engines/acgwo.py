"""
pyMetaheuristic src — Adaptive Chaotic Grey Wolf Optimizer Engine
================================================================
Native macro-step: one full pack position update with adaptive+chaotic initialisation.
payload keys: population (ndarray), alpha/beta/delta (1-D arrays), a_lin (float)
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class ACGWOEngine(BaseEngine):
    algorithm_id   = "acgwo"
    algorithm_name = "Adaptive Chaotic Grey Wolf Optimizer"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS = dict(pack_size=15, lmbda=0.5)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["pack_size"]); self._lmbda = float(p["lmbda"])
        if config.seed is not None: np.random.seed(config.seed)

    def _chaotic(self, x): return np.where(x < self._lmbda, x/self._lmbda, (1-x)/(1-self._lmbda))

    def initialize(self):
        lo = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        rv = np.random.uniform(0,1,(self._n, self.problem.dimension))
        pos = lo + self._chaotic(rv)*(hi-lo); pos = np.clip(pos,lo,hi)
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:,np.newaxis]))
        idx = np.argsort(pop[:,-1])
        alpha=pop[idx[0],:].copy(); beta=pop[idx[1],:].copy() if self._n>1 else alpha.copy()
        delta=pop[idx[2],:].copy() if self._n>2 else beta.copy()
        return EngineState(step=0, evaluations=self._n,
            best_position=alpha[:-1].tolist(), best_fitness=float(alpha[-1]),
            initialized=True, payload=dict(population=pop, alpha=alpha, beta=beta, delta=delta))

    def step(self, state):
        pl = state.payload; pop = pl["population"]
        alpha,beta,delta = pl["alpha"],pl["beta"],pl["delta"]
        lo = np.array(self.problem.min_values); hi = np.array(self.problem.max_values)
        T = self.config.max_steps or 1
        a = 2 - 2*((np.exp(state.step/T)-1)/(np.exp(1)-1))
        idx = np.argsort(pop[:,-1])
        alpha=pop[idx[0],:].copy(); beta=pop[idx[1],:].copy() if self._n>1 else alpha.copy()
        delta=pop[idx[2],:].copy() if self._n>2 else beta.copy()
        dim = self.problem.dimension
        r1=np.random.rand(self._n,dim); r2=np.random.rand(self._n,dim)
        A=2*a*r1-a; C=2*r2
        da=np.abs(C*alpha[:dim]-pop[:,:dim]); db=np.abs(C*beta[:dim]-pop[:,:dim]); dd=np.abs(C*delta[:dim]-pop[:,:dim])
        x1=np.clip(alpha[:dim]-A*da,lo,hi); x2=np.clip(beta[:dim]-A*db,lo,hi); x3=np.clip(delta[:dim]-A*dd,lo,hi)
        f1=self._evaluate_population(x1)
        f2=self._evaluate_population(x2)
        f3=self._evaluate_population(x3)
        wa=alpha[-1]/(beta[-1]+1e-16); wb=beta[-1]/(delta[-1]+1e-16)
        new_pos=np.clip((x1*wa+x2*wb+x3)/3,lo,hi)
        new_fit=self._evaluate_population(new_pos)
        new_pop=np.hstack((new_pos,new_fit[:,np.newaxis]))
        combined=np.vstack([pop,new_pop,np.hstack((x1,f1[:,np.newaxis])),np.hstack((x2,f2[:,np.newaxis])),np.hstack((x3,f3[:,np.newaxis]))])
        combined=combined[combined[:,-1].argsort()][:self._n,:]
        idx=np.argsort(combined[:,-1])
        alpha=combined[idx[0],:].copy(); beta=combined[idx[1],:].copy() if self._n>1 else alpha.copy()
        delta=combined[idx[2],:].copy() if self._n>2 else beta.copy()
        state.step+=1; state.evaluations+=self._n*4
        state.payload=dict(population=combined,alpha=alpha,beta=beta,delta=delta)
        if self.problem.is_better(float(alpha[-1]),state.best_fitness):
            state.best_fitness=float(alpha[-1]); state.best_position=alpha[:-1].tolist()
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

    def get_best_candidate(self, state):
        return CandidateRecord(position=list(state.best_position), fitness=state.best_fitness,
            source_algorithm=self.algorithm_id, source_step=state.step, role="best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position), best_fitness=state.best_fitness,
            steps=state.step, evaluations=state.evaluations,
            termination_reason=state.termination_reason, capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        pop=state.payload["population"]
        return [CandidateRecord(position=pop[i,:-1].tolist(), fitness=float(pop[i,-1]),
            source_algorithm=self.algorithm_id, source_step=state.step, role="current")
            for i in range(pop.shape[0])]
