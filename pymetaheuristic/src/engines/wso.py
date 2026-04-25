"""pyMetaheuristic src — White Shark Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WSOEngine(PortedPopulationEngine):
    algorithm_id   = "wso"
    algorithm_name = "White Shark Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2022.108457"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"v": np.zeros_like(pop[:,:-1]), "wbest": pop[:,:-1].copy(),
                "wfit": pop[:,-1].copy(), "gbest": pop[self._best_index(pop[:,-1]),:-1].copy(),
                "fmin0": pop[self._best_index(pop[:,-1]),-1]}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        v=state.payload["v"]; wbest=state.payload["wbest"]
        wfit=state.payload["wfit"]; gbest=state.payload["gbest"]; fmin0=state.payload["fmin0"]
        fmax=0.75; fmin=0.07; tau=4.11
        mu=2/abs(2-tau-np.sqrt(tau**2-4*tau))
        pmin,pmax=0.5,1.5; a0,a1_,a2_=6.25,100,0.0005
        mv=1/(a0+np.exp((max_iter/2-t)/a1_))
        s_s=abs(1-np.exp(-a2_*t/max_iter))
        p1_=pmax+(pmax-pmin)*np.exp(-(4*t/max_iter)**2)
        nu=np.random.randint(0,n,n)
        for i in range(n):
            rmin,rmax=1.0,3.0; rr=rmin+np.random.random()*(rmax-rmin)
            wr=abs((2*np.random.random()-np.random.random()-np.random.random())/rr)
            v[i]=mu*v[i]+wr*(wbest[nu[i]]-pop[i,:-1])
        for i in range(n):
            f_=fmin+(fmax-fmin)/(fmax+fmin)
            a_=pop[i,:-1]>hi; b_=pop[i,:-1]<lo; wo=~(a_&b_)
            if np.random.random()<mv:
                pop[i,:-1]=pop[i,:-1]*~wo+hi*a_+lo*b_
            else:
                pop[i,:-1]+=v[i]/f_
        for i in range(n):
            for j in range(d):
                if np.random.random()<s_s:
                    Dist=abs(np.random.random()*(gbest[j]-pop[i,j]))
                    if i==0:
                        pop[i,j]=gbest[j]+np.random.random()*Dist*np.sign(np.random.random()-0.5)
                    else:
                        prev=pop[i-1,j]
                        tmp=gbest[j]+np.random.random()*Dist*np.sign(np.random.random()-0.5)
                        pop[i,j]=(tmp+prev)/2*np.random.random()
        for i in range(n):
            if np.all(pop[i,:-1]>=lo) and np.all(pop[i,:-1]<=hi):
                new_fit=float(self._evaluate_population(pop[i,:-1][None])[0]); evals+=1
                pop[i,-1]=new_fit
                if self._is_better(new_fit,wfit[i]):
                    wbest[i]=pop[i,:-1].copy(); wfit[i]=new_fit
                if self._is_better(wfit[i],fmin0):
                    fmin0=wfit[i]; gbest=wbest[i].copy()
        state.payload.update({"v":v,"wbest":wbest,"wfit":wfit,"gbest":gbest,"fmin0":fmin0})
        return pop, evals, {}
