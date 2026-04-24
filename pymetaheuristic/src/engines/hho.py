"""pyMetaheuristic src — Harris Hawks Optimization Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class HHOEngine(BaseEngine):
    algorithm_id   = "hho"
    algorithm_name = "Harris Hawks Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(hawks=50)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}; self._n=int(p["hawks"])
        if config.seed is not None: np.random.seed(config.seed)

    def _levy(self, dim, beta=1.5):
        r1=np.random.rand(dim); r2=np.random.rand(dim)
        sn=gamma(1+beta)*np.sin(np.pi*beta/2); sd=gamma((1+beta)/2)*beta*2**((beta-1)/2)
        sigma=(sn/sd)**(1/beta); return (0.01*r1*sigma)/(np.abs(r2)**(1/beta))

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def initialize(self):
        pop=self._init_pop(); rabbit=pop[0,:].copy()
        return EngineState(step=0,evaluations=self._n,
            best_position=rabbit[:-1].tolist(),best_fitness=float(rabbit[-1]),
            initialized=True,payload=dict(population=pop,rabbit=rabbit))

    def step(self, state):
        pop=state.payload["population"]; rabbit=state.payload["rabbit"]
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        T=self.config.max_steps or 1; t=state.step; evals=0
        bi=np.argmin(pop[:,-1])
        if pop[bi,-1]<rabbit[-1]: rabbit=pop[bi,:].copy()
        er=2*(1-t/T)
        for i in range(self._n):
            ee=er*(2*np.random.rand()-1); ae=abs(ee)
            if ae>=1:
                r1=np.random.rand(); idx=np.random.randint(self._n); hawk=pop[idx,:]
                if r1<0.5:
                    a=np.random.rand(); b=np.random.rand()
                    pop[i,:-1]=hawk[:-1]-a*abs(hawk[:-1]-2*b*pop[i,:-1])
                else:
                    c=np.random.rand(); d=np.random.rand()
                    pop[i,:-1]=(rabbit[:-1]-pop[i,:-1].mean())-(c*(hi-lo)*d+lo)
                pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
            else:
                r2=np.random.rand()
                if r2>=0.5 and ae<0.5:
                    pop[i,:-1]=rabbit[:-1]-ee*abs(rabbit[:-1]-pop[i,:-1])
                    pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
                elif r2>=0.5 and ae>=0.5:
                    e=np.random.rand(); js=2*(1-e)
                    pop[i,:-1]=(rabbit[:-1]-pop[i,:-1])-ee*abs(js*rabbit[:-1]-pop[i,:-1])
                    pop[i,-1]=self.problem.evaluate(pop[i,:-1]); evals+=1
                elif r2<0.5 and ae>=0.5:
                    f=np.random.rand(); js=2*(1-f)
                    x1=rabbit[:-1]-ee*abs(js*rabbit[:-1]-pop[i,:-1])
                    x1c=np.clip(x1,lo,hi); fx1=self.problem.evaluate(x1c); evals+=1
                    if fx1<pop[i,-1]: pop[i,:-1]=x1c; pop[i,-1]=fx1
                    else:
                        x2=rabbit[:-1]-ee*abs(js*rabbit[:-1]-pop[i,:-1])+np.random.randn(self.problem.dimension)*self._levy(self.problem.dimension)
                        x2c=np.clip(x2,lo,hi); fx2=self.problem.evaluate(x2c); evals+=1
                        if fx2<pop[i,-1]: pop[i,:-1]=x2c; pop[i,-1]=fx2
                else:
                    g=np.random.rand(); js=2*(1-g)
                    x1=rabbit[:-1]-ee*abs(js*rabbit[:-1]-pop[i,:-1].mean())
                    x1c=np.clip(x1,lo,hi); fx1=self.problem.evaluate(x1c); evals+=1
                    if fx1<pop[i,-1]: pop[i,:-1]=x1c; pop[i,-1]=fx1
                    else:
                        x2=rabbit[:-1]-ee*abs(js*rabbit[:-1]-pop[i,:-1].mean())+np.random.randn(self.problem.dimension)*self._levy(self.problem.dimension)
                        x2c=np.clip(x2,lo,hi); fx2=self.problem.evaluate(x2c); evals+=1
                        if fx2<pop[i,-1]: pop[i,:-1]=x2c; pop[i,-1]=fx2
        bi2=np.argmin(pop[:,-1])
        if pop[bi2,-1]<rabbit[-1]: rabbit=pop[bi2,:].copy()
        state.step+=1; state.evaluations+=evals; state.payload=dict(population=pop,rabbit=rabbit)
        if self.problem.is_better(float(rabbit[-1]),state.best_fitness):
            state.best_fitness=float(rabbit[-1]); state.best_position=rabbit[:-1].tolist()
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
