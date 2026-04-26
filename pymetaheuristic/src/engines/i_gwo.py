"""pyMetaheuristic src — Improved Grey Wolf Optimizer Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class I_GWOEngine(BaseEngine):
    algorithm_id   = "i_gwo"
    algorithm_name = "Improved Grey Wolf Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2020.113917"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(pack_size=25)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["pack_size"])

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

        return EngineState(step=0,evaluations=self._n,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,elite=elite))

    def step(self, state):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pop=state.payload["population"]; elite=state.payload["elite"]
        evals=0
        T=self.config.max_steps or 1; t=state.step; a=2-t*(2/T)
        idx=np.argsort(pop[:,-1])
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
        upd=np.clip((x1+x2+x3)/3,lo,hi)
        uf=self._evaluate_population(upd); evals+=self._n
        up=np.hstack((upd,uf[:,np.newaxis]))
        combined=np.vstack([pop,up,np.hstack((x1,f1[:,np.newaxis])),np.hstack((x2,f2[:,np.newaxis])),np.hstack((x3,f3[:,np.newaxis]))])
        combined=combined[combined[:,-1].argsort()]; upd2=combined[:self._n,:]
        # improve step
        a2=pop[:,:-1]; b2=a2.reshape(np.prod(a2.shape[:-1]),1,a2.shape[-1])
        dm=np.sqrt(np.einsum("ijk,ijk->ij",b2-a2,b2-a2)).squeeze()
        ip=np.copy(pop)
        for i in range(self._n):
            dist=np.sqrt(np.sum((pop[i,:-1]-upd2[i,:-1])**2))
            idx2=np.where(dm[i,:]<=dist)[0]
            for j in range(dim):
                r=np.random.rand(); ix1=np.random.choice(idx2); ix2=np.random.randint(self._n)
                ip[i,j]=np.clip(ip[i,j]+r*(pop[ix1,j]-pop[ix2,j]),lo[j],hi[j])
            ip[i,-1]=self.problem.evaluate(ip[i,:-1]); evals+=1
            mf=min(pop[i,-1],upd2[i,-1],ip[i,-1])
            if upd2[i,-1]==mf: ip[i,:]=upd2[i,:]
            elif pop[i,-1]==mf: ip[i,:]=pop[i,:]
        pop=ip
        bi=np.argmin(pop[:,-1])
        if self.problem.is_better(float(pop[bi,-1]),float(elite[-1])): elite=pop[bi,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,elite=elite)
        if self.problem.is_better(float(elite[-1]),state.best_fitness):
            state.best_fitness=float(elite[-1]); state.best_position=elite[:-1].tolist()
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
