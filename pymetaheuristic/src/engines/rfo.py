"""pyMetaheuristic src — Rüppell's Fox Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RFOEngine(PortedPopulationEngine):
    algorithm_id   = "rfo"
    algorithm_name = "Rüppell's Fox Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10586-024-04823-3"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"pbest": pop[:,:-1].copy(), "pfit": pop[:,-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        pbest=state.payload["pbest"]; pfit=state.payload["pfit"]
        fmin0=pfit[self._best_index(pfit)]; fbest=pbest[self._best_index(pfit)].copy()
        beta=1e-10; L=100.0
        h=1/(1+np.exp((t-max_iter/2)/L)); s=1/(1+np.exp((max_iter/2-t)/L))
        tmp=2/(1+np.exp((max_iter/2-t)/100)); smell=0.1/abs(np.arccos(np.clip(tmp,-1,1))+1e-300)
        x=pop[:,:-1].copy(); fpos=x.copy(); fit=pop[:,-1].copy()
        vec_flag=[-1,1]; ds=np.random.randint(0,n,n)
        if np.random.random()>=0.5:
            for i in range(n):
                if s>=h:
                    if np.random.random()>=0.25:
                        ri=ds[i]; r=np.random.random()
                        fpos[i]=x[i]+r*(pbest[ri]-x[i])+r*(fbest-x[i])
                        fpos[i]=_rotate(fpos[i],x[i],np.random.random()*260*np.pi/180,d)
                    else:
                        Flag=np.random.choice(vec_flag)
                        fpos[i]=pbest[ds[i]]+beta*np.random.randn(d)*Flag
                else:
                    if np.random.random()>=0.75:
                        ri=ds[i]; r=np.random.random()
                        fpos[i]=x[i]+r*(pbest[ri]-x[i])+r*(fbest-x[i])
                        fpos[i]=_rotate(fpos[i],x[i],np.random.random()*150*np.pi/180,d)
                    else:
                        Flag=np.random.choice(vec_flag)
                        fpos[i]=pbest[ds[i]]+beta*np.random.randn(d)*Flag
        else:
            dr=np.random.randint(0,n,n)
            for i in range(n):
                if h>=s:
                    if np.random.random()>=0.25:
                        ri=dr[i]; r=np.random.random()
                        fpos[i]=x[i]+r*(pbest[ri]-x[i])+r*(fbest-x[i])
                        fpos[i]=_rotate(fpos[i],x[i],np.random.random()*150*np.pi/180,d)
                    else:
                        Flag=np.random.choice(vec_flag)
                        fpos[i]=pbest[dr[i]]+beta*np.random.randn(d)*Flag
                else:
                    if np.random.random()>=0.75:
                        ri=dr[i]; r=np.random.random()
                        fpos[i]=x[i]+r*(pbest[ri]-x[i])+r*(fbest-x[i])
                        fpos[i]=_rotate(fpos[i],x[i],np.random.random()*260*np.pi/180,d)
                    else:
                        Flag=np.random.choice(vec_flag)
                        fpos[i]=pbest[dr[i]]+beta*np.random.randn(d)*Flag
        # smell
        ss=np.random.randint(0,n,n)
        for i in range(n):
            if np.random.random()>=smell:
                xr=2+np.random.random()*(4-2)
                eps=abs(4*np.random.random()-(np.random.random()+np.random.random()))/xr
                fpos[i]=x[i]+eps*(pbest[ss[i]]-x[i])+eps*(fbest-x[i])
            else:
                Flag=np.random.choice(vec_flag)
                fpos[i]=pbest[ss[i]]+beta*np.random.randn(d)*Flag
        fpos=np.clip(fpos,lo,hi)
        new_fits=self._evaluate_population(fpos); evals+=n
        mask=self._better_mask(new_fits,pfit)
        pbest[mask]=fpos[mask]; pfit[mask]=new_fits[mask]
        state.payload.update({"pbest":pbest,"pfit":pfit})
        return np.hstack([fpos,new_fits[:,None]]), evals, {}

def _rotate(pos,center,theta,d):
    if d<2: return pos
    diff=pos-center
    c=np.cos(theta); s_=np.sin(theta)
    rot=np.zeros_like(diff)
    rot[0]=c*diff[0]-s_*diff[1 if d>1 else 0]
    rot[1]=s_*diff[0]+c*diff[1 if d>1 else 0]
    if d>2: rot[2:]=diff[2:]
    return center+rot
