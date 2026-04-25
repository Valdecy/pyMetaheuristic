"""pyMetaheuristic src — Young's Double-Slit Experiment Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class YDSEEngine(PortedPopulationEngine):
    algorithm_id   = "ydse"
    algorithm_name = "Young's Double-Slit Experiment Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.cma.2022.115652"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000); t_eff=max(t,1)
        L=1; ds=5e-3; I=0.01; Lambda=5e-6; Delta=0.38
        q=t_eff/max_iter; Int_max0=1e-20; Int_max=Int_max0*q
        order=self._order(pop[:,-1]); pop=pop[order]
        best_pos=pop[0,:-1].copy()
        X_new=np.zeros((n,d))
        for i in range(n):
            a=t_eff**(2*np.random.random()-1)
            H=2*np.random.random(d)-1
            z=a/np.where(np.abs(H)<1e-300,1e-300,H)
            r1=np.random.random()
            if i==0:
                beta_=q*np.cosh(np.pi/t_eff)
                A_bright=2/(1+np.sqrt(abs(1-beta_**2)))
                rand_a=np.random.choice([j for j in range(n) if j%2==0],1)[0]
                X_new[i]=best_pos+Int_max*A_bright*pop[i,:-1]-r1*z*pop[rand_a,:-1]
            elif i%2==1:
                beta_=q*np.cosh(np.pi/t_eff)
                A_bright=2/(1+np.sqrt(abs(1-beta_**2)))
                m_=(i-1)
                y_bright=Lambda*L*m_/ds
                Int_bright=Int_max*np.cos((np.pi*ds)/(Lambda*L)*y_bright)**2
                s_=np.random.choice(n,2,replace=False)
                g=2*np.random.random()-1
                Y=pop[s_[1],:-1]-pop[s_[0],:-1]
                X_new[i]=pop[i,:-1]-((1-g)*A_bright*Int_bright*pop[i,:-1]+g*Y)
            else:
                A_dark=Delta*np.arctanh(-(q)+1)
                m_=(i-1)
                y_dark=Lambda*L*(m_+0.5)/ds
                Int_dark=Int_max*np.cos((np.pi*ds)/(Lambda*L)*y_dark)**2
                X_new[i]=pop[i,:-1]-(r1*A_dark*Int_dark*pop[i,:-1]-z*best_pos)
            X_new[i]=np.clip(X_new[i],lo,hi)
        new_fits=self._evaluate_population(X_new); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([X_new,new_fits[:,None]])[mask]
        return pop, evals, {}
