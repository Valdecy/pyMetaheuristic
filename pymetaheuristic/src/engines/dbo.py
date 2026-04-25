"""pyMetaheuristic src — Dung Beetle Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class DBOEngine(PortedPopulationEngine):
    algorithm_id   = "dbo"
    algorithm_name = "Dung Beetle Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s11227-022-04959-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, P_percent=0.2)

    def _initialize_payload(self, pop):
        return {"pX": pop[:, :-1].copy(), "pFit": pop[:, -1].copy(), "XX": pop[:, :-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        P_percent = float(self._params.get("P_percent", 0.2))
        pNum = max(1, round(n * P_percent))
        pX=state.payload["pX"]; pFit=state.payload["pFit"]; XX=state.payload["XX"]
        x=pop[:, :-1].copy(); fit=pop[:, -1].copy()
        worst_idx=self._worst_index(fit)
        worse=x[worst_idx].copy()
        r2=np.random.random()
        for i in range(pNum):
            if r2<0.9:
                r1=np.random.random()
                a=1 if np.random.random()>0.1 else -1
                x[i]=pX[i]+0.3*np.abs(pX[i]-worse)+a*0.1*XX[i]
            else:
                theta=np.random.choice([0,90,180])*np.pi/180
                x[i]=pX[i]+np.tan(theta)*np.abs(pX[i]-XX[i])
            x[i]=np.clip(x[i],lo,hi)
        new_fits_p=self._evaluate_population(x[:pNum]); evals+=pNum
        fit[:pNum]=new_fits_p
        best_idx=self._best_index(fit)
        bestXX=x[best_idx].copy()
        R=1-t/max_iter
        Xnew1=np.clip(bestXX*(1-R),lo,hi); Xnew2=np.clip(bestXX*(1+R),lo,hi)
        bX=pX[self._best_index(pFit)].copy()
        Xnew11=np.clip(bX*(1-R),lo,hi); Xnew22=np.clip(bX*(1+R),lo,hi)
        mid1=min(pNum,n); mid2=min(12,n); mid3=min(19,n)
        for i in range(mid1, mid2):
            x[i]=bestXX+np.random.random(d)*(pX[i]-Xnew1)+np.random.random(d)*(pX[i]-Xnew2)
            x[i]=np.clip(x[i],Xnew1,Xnew2)
        for i in range(mid2, mid3):
            x[i]=pX[i]+np.random.randn()*(pX[i]-Xnew11)+np.random.random(d)*(pX[i]-Xnew22)
            x[i]=np.clip(x[i],lo,hi)
        for i in range(mid3, n):
            x[i]=bX+np.random.randn(d)*((np.abs(pX[i]-bestXX)+np.abs(pX[i]-bX))/2)
            x[i]=np.clip(x[i],lo,hi)
        rest_fits=self._evaluate_population(x[pNum:]); evals+=n-pNum
        fit[pNum:]=rest_fits
        XX=pX.copy()
        mask=self._better_mask(fit,pFit)
        pFit[mask]=fit[mask]; pX[mask]=x[mask]
        state.payload.update({"pX":pX,"pFit":pFit,"XX":XX})
        return np.hstack([x,fit[:,None]]), evals, {}
