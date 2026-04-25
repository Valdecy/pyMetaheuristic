"""pyMetaheuristic src — Quadratic Interpolation Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _interpolation(xi,xj,xk,fi,fj,fk,l,u):
    a=(xj**2-xk**2)*fi+(xk**2-xi**2)*fj+(xi**2-xj**2)*fk
    b=2*((xj-xk)*fi+(xk-xi)*fj+(xi-xj)*fk)
    L_xmin=a/(b+1e-300)
    if np.isnan(L_xmin) or np.isinf(L_xmin) or L_xmin>u or L_xmin<l:
        L_xmin=np.random.random()*(u-l)+l
    return L_xmin

class QIOEngine(PortedPopulationEngine):
    algorithm_id   = "qio"
    algorithm_name = "Quadratic Interpolation Optimization"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.cma.2023.116446"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        for i in range(n):
            new_pos=np.zeros(d)
            others=[k for k in range(n) if k!=i]
            if np.random.random()>0.5 and len(others)>=3:
                K1,K2,K3=np.random.choice(others,3,replace=False)
                f1,f2,f3=pop[i,-1],pop[K1,-1],pop[K2,-1]
                for j in range(d):
                    x1,x2,x3=pop[i,j],pop[K1,j],pop[K2,j]
                    new_pos[j]=_interpolation(x1,x2,x3,f1,f2,f3,lo[j],hi[j])
                a_c=np.cos(np.pi/2*t/max_iter)
                b_c=0.7*a_c+0.15*a_c*(np.cos(5*np.pi*t/max_iter)+1)
                w1=3*b_c*np.random.randn()
                new_pos+=w1*(pop[K3,:-1]-new_pos)+round(0.5*(0.05+np.random.random()))*np.log(np.random.random()/max(np.random.random(),1e-300))
            else:
                if len(others)>=2:
                    K1,K2=np.random.choice(others,2,replace=False)
                else:
                    K1=K2=i
                f1,f2=pop[i,-1],pop[K1,-1]
                for j in range(d):
                    x1,x2=pop[i,j],pop[K1,j]
                    x3=2*x1-x2
                    new_pos[j]=_interpolation(x1,x2,x3,f1,f2,f1,lo[j],hi[j])
                a_c=np.cos(np.pi/2*t/max_iter)
                b_c=0.7*a_c+0.15*a_c*(np.cos(5*np.pi*t/max_iter)+1)
                w1=3*b_c*np.random.randn()
                new_pos+=w1*(pop[K2,:-1]-new_pos)
            new_pos=np.clip(new_pos,lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(new_pos,new_fit)
        return pop, evals, {}
