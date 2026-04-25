"""pyMetaheuristic src — Termite Life Cycle Optimizer Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class TLCOEngine(PortedPopulationEngine):
    algorithm_id   = "tlco"
    algorithm_name = "Termite Life Cycle Optimizer"
    family         = "bio"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2022.119211"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"tw": np.zeros(pop.shape[0]), "ts": np.zeros(pop.shape[0])}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        tw=state.payload["tw"]; ts=state.payload["ts"]
        beta=0.5/max_iter*t+1.5
        sigma=1/(0.1*max_iter)
        lw=1-1/(1+np.exp(-sigma*(t-0.5*max_iter)))
        ls=1/(1+np.exp(-sigma*(t-0.5*max_iter)))
        gbest=pop[self._best_index(pop[:,-1])]; gbest_pos=gbest[:-1].copy(); gbest_fit=gbest[-1]
        nw=int(round(0.7*n)); rl=_levy(n,d,beta)
        for i in range(nw):
            sign=np.random.choice([-1,1])
            new_pos=np.clip(pop[i,:-1]+sign*(np.random.random(d)+rl[i])*np.abs(gbest_pos-pop[i,:-1]),lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(new_pos,new_fit); tw[i]=0
                if self._is_better(new_fit,gbest_fit): gbest_pos=new_pos.copy(); gbest_fit=new_fit
            else:
                tw[i]+=1
                if tw[i]/max_iter>lw:
                    tw[i]=0
                    r1,r2=np.random.randint(n),np.random.randint(n)
                    if self._is_better(pop[r2,-1],pop[r1,-1]):
                        rp=pop[r1,:-1]+np.random.random(d)*(pop[r2,:-1]-pop[r1,:-1])
                    else:
                        rp=pop[r1,:-1]-np.random.random(d)*(pop[r2,:-1]-pop[r1,:-1])
                    rp=np.clip(rp,lo,hi)
                    rf=float(self._evaluate_population(rp[None])[0]); evals+=1
                    if self._is_better(rf,pop[i,-1]):
                        pop[i]=np.append(rp,rf)
                        if self._is_better(rf,gbest_fit): gbest_pos=rp.copy(); gbest_fit=rf
        for i in range(nw,n):
            rl_s=_levy(1,d,beta)
            sign=np.random.choice([-1,1])
            new_pos=np.clip(2*np.random.random()*gbest_pos+sign*np.abs(pop[i,:-1]-rl_s[0]*gbest_pos),lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(new_pos,new_fit); ts[i]=0
                if self._is_better(new_fit,gbest_fit): gbest_pos=new_pos.copy(); gbest_fit=new_fit
            else:
                ts[i]+=1
                if ts[i]/max_iter>ls:
                    ts[i]=0
                    r1,r2=np.random.randint(n),np.random.randint(n)
                    if self._is_better(pop[r2,-1],pop[r1,-1]):
                        rp=pop[r1,:-1]+np.random.random(d)*(pop[r2,:-1]-pop[r1,:-1])
                    else:
                        rp=pop[r1,:-1]-np.random.random(d)*(pop[r2,:-1]-pop[r1,:-1])
                    rp=np.clip(rp,lo,hi)
                    rf=float(self._evaluate_population(rp[None])[0]); evals+=1
                    if self._is_better(rf,pop[i,-1]):
                        pop[i]=np.append(rp,rf)
                        if self._is_better(rf,gbest_fit): gbest_pos=rp.copy(); gbest_fit=rf
        state.payload.update({"tw":tw,"ts":ts})
        return pop, evals, {}
