"""pyMetaheuristic src — Student Psychology Based Optimization Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class SPBOEngine(BaseEngine):
    algorithm_id   = "spbo"
    algorithm_name = "Student Psychology Based Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(population_size=50)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])

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
        # classify
        n=len(pop)-1; segs=[]
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    for L in range(k+1,n+(i>0)):
                        segs.append((i,j,k,L))
        if segs: y=segs[np.random.randint(len(segs))]; ii,jj,kk,LL=y
        else: ii,jj,kk,LL=0,1,2,3
        p_lst=list(range(self._n)); ga=p_lst[:ii]; gb=p_lst[ii:jj]; av=p_lst[jj:LL]; rd=p_lst[LL:]
        # update best student
        idx2=list(range(self._n)); idx3=np.random.choice(idx2)
        k2=np.random.choice([1,2]); nb=np.copy(elite)
        for j in range(self.problem.dimension):
            nb[j]=np.clip(nb[j]+((-1)**k2)*np.random.rand()*(nb[j]-pop[idx3,j]),lo[j],hi[j])
        nb[-1]=self.problem.evaluate(nb[:-1]); evals+=1
        if nb[-1]<elite[-1]: elite=nb.copy()
        # update groups
        for i in ga:
            for j in range(self.problem.dimension): pop[i,j]=np.clip(pop[i,j]+np.random.rand()*(elite[j]-pop[i,j]),lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        if gb:
            gm=pop[gb,:-1].mean(axis=0) if len(gb)>0 else elite[:-1]
            for i in gb:
                for j in range(self.problem.dimension):
                    r1=np.random.rand(); r2=np.random.rand()
                    pop[i,j]=np.clip(pop[i,j]+r1*(elite[j]-pop[i,j])+r2*(pop[i,j]-gm[j]),lo[j],hi[j])
                pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        if av:
            am=pop[av,:-1].mean(axis=0) if len(av)>0 else elite[:-1]
            for i in av:
                for j in range(self.problem.dimension): pop[i,j]=np.clip(pop[i,j]+np.random.rand()*(am[j]-pop[i,j]),lo[j],hi[j])
                pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        for i in rd:
            for j in range(self.problem.dimension): pop[i,j]=np.clip(pop[:,j].min()+np.random.rand()*(pop[:,j].max()-pop[:,j].min()),lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
        combined=np.vstack([pop,np.atleast_2d(elite)]); combined=combined[combined[:,-1].argsort()]
        pop=combined[:self._n,:]
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
