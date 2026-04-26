"""pyMetaheuristic src — Ant Lion Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class ALOEngine(BaseEngine):
    algorithm_id   = "alo"
    algorithm_name = "Ant Lion Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2015.01.010"}
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(colony_size=500)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n=int(p["colony_size"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    @staticmethod
    def _fit_calc(fv): return np.where(fv>=0,1/(1+fv),1+np.abs(fv))
    def _fit_fn(self,src):
        fv=self._fit_calc(src[:,-1]); cs=np.cumsum(fv); cs/=cs[-1]
        return np.column_stack((fv,cs))
    def _roulette(self,fit): return np.searchsorted(fit[:,1],np.random.rand())
    def _rw(self,iters): s=np.cumsum(np.where(np.random.rand(iters)>0.5,1,-1)); return np.insert(s,0,0)

    def initialize(self):
        pop=self._init_pop(); al=self._init_pop()
        elite=al[al[:,-1].argsort()][0,:].copy()
        return EngineState(step=0,evaluations=self._n*2,
            best_position=elite[:-1].tolist(),best_fitness=float(elite[-1]),
            initialized=True,payload=dict(population=pop,antlions=al,elite=elite))

    def step(self, state):
        pl=state.payload; pop=pl["population"]; al=pl["antlions"]; elite=pl["elite"]
        T=self.config.max_steps or 1; t=state.step
        i_ratio=1
        if t>0.1*T:
            w=min(2+int((t/T-0.1)/0.15),6); i_ratio=(10**w)*(t/T)
        sal=al[al[:,-1].argsort()]; best_al=sal[0,:-1]; worst_al=sal[-1,:-1]
        fit=self._fit_fn(al)
        xrw=self._rw(T); erw=self._rw(T)
        mx=xrw.max(); mnx=xrw.min(); me=erw.max(); mne=erw.min()
        lo=self.problem.min_values; hi=self.problem.max_values
        for i in range(pop.shape[0]):
            ant=self._roulette(fit)
            for j in range(len(lo)):
                r=np.random.rand()
                minc=(best_al[j]/i_ratio)+(al[ant,j] if r<0.5 else -best_al[j]/i_ratio)
                maxd=(worst_al[j]/i_ratio)+(al[ant,j] if r>=0.5 else -worst_al[j]/i_ratio)
                xw=(((xrw[t]-mnx)*(maxd-minc))/(mx-mnx+1e-16))+minc
                ew=(((erw[t]-mne)*(maxd-minc))/(me-mne+1e-16))+minc
                pop[i,j]=np.clip((xw+ew)/2,lo[j],hi[j])
            pop[i,-1]=self.problem.evaluate(pop[i,:-1])
        # combine
        comb=np.vstack([pop,al]); comb=comb[comb[:,-1].argsort()]
        new_pop=comb[:self._n,:]; new_al=comb[self._n:,:]
        val=new_al[new_al[:,-1].argsort()][0,:]
        if elite[-1]>val[-1]: elite=val.copy()
        else: new_al[new_al[:,-1].argsort()][0,:]=elite.copy()
        state.step+=1; state.evaluations+=self._n
        state.payload=dict(population=new_pop,antlions=new_al,elite=elite)
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


    def inject_candidates(self, state, candidates, policy="native"):
        pop = state.payload["population"]
        antlions = state.payload["antlions"]
        worst_pop = np.argsort(pop[:, -1])[::-1]
        worst_al = np.argsort(antlions[:, -1])[::-1]
        replaced = []
        for j, cand in enumerate(candidates):
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            wi = int(worst_pop[j % len(worst_pop)])
            ai = int(worst_al[j % len(worst_al)])
            pop[wi, :-1] = pos; pop[wi, -1] = fit
            antlions[ai, :-1] = pos; antlions[ai, -1] = fit
            state.evaluations += 1
            replaced.append((wi, ai))
        elite = antlions[antlions[:, -1].argsort()][0, :].copy()
        state.payload["population"] = pop
        state.payload["antlions"] = antlions
        state.payload["elite"] = elite
        if self.problem.is_better(float(elite[-1]), state.best_fitness):
            state.best_fitness = float(elite[-1]); state.best_position = elite[:-1].tolist()
        return state
