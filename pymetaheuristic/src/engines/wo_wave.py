"""pyMetaheuristic src — Wave Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class WaveOptEngine(PortedPopulationEngine):
    algorithm_id   = "wo_wave"
    algorithm_name = "Wave Optimization Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.cor.2014.10.008"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, P=0.4)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        P=float(self._params.get("P",0.4))
        F_number=max(1,round(n*P)); M_number=F_number; C_number=n-F_number-M_number
        Alpha=1-t/max_iter; Beta=1-1/(1+np.exp((0.5*max_iter-t)/max_iter*10))
        A=2*Alpha; r1=np.random.random(); R=2*r1-1; Danger=A*R
        Safety=np.random.random()
        for i in range(n):
            ub_check=pop[i,:-1]>hi; lb_check=pop[i,:-1]<lo
            pop[i,:-1]=(pop[i,:-1]*~(ub_check|lb_check))+hi*ub_check+lo*lb_check
        best_idx=self._best_index(pop[:,-1]); best_pos=pop[best_idx,:-1].copy(); best_sc=pop[best_idx,-1]
        sec_sc=np.inf; sec_pos=best_pos.copy()
        for i in range(n):
            if pop[i,-1]>best_sc and pop[i,-1]<sec_sc:
                sec_sc=pop[i,-1]; sec_pos=pop[i,:-1].copy()
        GBestX=np.tile(best_pos,(n,1))
        if abs(Danger)>=1:
            r3=np.random.random(); p1=np.random.permutation(n); p2=np.random.permutation(n)
            mig=(Beta*r3**2)*(pop[p1,:-1]-pop[p2,:-1])
            pop[:,:-1]+=mig
        elif abs(Danger)<1:
            if Safety>=0.5:
                for i in range(M_number):
                    base=7; result=0.0; f=1/base; idx=i+1
                    while idx>0:
                        result+=f*int(idx%base); idx=idx//base; f/=base
                    m1=lo+result*(hi-lo); pop[i,:-1]=m1
                for j in range(M_number,M_number+F_number):
                    pop[j,:-1]+=Alpha*(pop[M_number-1,:-1]-pop[j,:-1])+(1-Alpha)*(GBestX[j]-pop[j,:-1])
                for i in range(max(0,n-C_number),n):
                    o=GBestX[i]+pop[i,:-1]*_levy1d(d)
                    P_coef=np.random.random()
                    pop[i,:-1]=P_coef*(o-pop[i,:-1])
            if Safety<0.5 and abs(Danger)>=0.5:
                r4=np.random.random()
                pop[:,:-1]=pop[:,:-1]*R-np.abs(GBestX-pop[:,:-1])*r4**2
            if Safety<0.5 and abs(Danger)<0.5:
                for i in range(n):
                    for j in range(d):
                        t1=np.random.random(); a1=Beta*np.random.random()-Beta
                        b1=np.tan(t1*np.pi)
                        X1=best_pos[j]-a1*b1*abs(best_pos[j]-pop[i,j])
                        t2=np.random.random(); a2=Beta*np.random.random()-Beta
                        b2=np.tan(t2*np.pi)
                        X2=sec_pos[j]-a2*b2*abs(sec_pos[j]-pop[i,j])
                        pop[i,j]=(X1+X2)/2
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=new_fits
        return pop, evals, {}
