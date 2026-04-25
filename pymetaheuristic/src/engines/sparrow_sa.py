"""pyMetaheuristic src — Sparrow Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SparrowSAEngine(PortedPopulationEngine):
    algorithm_id   = "sparrow_sa"
    algorithm_name = "Sparrow Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1080/21642583.2019.1708830"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, P_percent=0.2)

    def _initialize_payload(self, pop):
        return {"pFit": pop[:,-1].copy(), "pX": pop[:,:-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        P_percent=float(self._params.get("P_percent",0.2))
        pNum=max(1,round(n*P_percent))
        pFit=state.payload["pFit"]; pX=state.payload["pX"]
        x=pop[:,:-1].copy(); fit=pop[:,-1].copy()
        sorted_idx=self._order(fit); worst_idx=self._worst_index(fit)
        worst=x[worst_idx].copy()
        best_idx=sorted_idx[0]; bestX=pX[self._best_index(pFit)].copy(); fMin=pFit[self._best_index(pFit)]
        r2=np.random.random()
        if r2<0.8:
            for i in range(pNum):
                si=sorted_idx[i]; r1=np.random.random()
                x[si]=pX[si]*np.exp(-si/(r1*max_iter))
                x[si]=np.clip(x[si],lo,hi)
        else:
            for i in range(pNum):
                si=sorted_idx[i]
                x[si]=pX[si]+np.random.randn(d)
                x[si]=np.clip(x[si],lo,hi)
        fits_p=self._evaluate_population(x[sorted_idx[:pNum]]); evals+=pNum
        bestXX_idx=np.argmin(fits_p); bestXX=x[sorted_idx[bestXX_idx]].copy()
        for i in range(pNum, n):
            si=sorted_idx[i]; A=2*np.floor(np.random.random(d)*2)-1
            if i>n//2:
                x[si]=np.random.randn(d)*np.exp((worst-pX[si])/(i+1)**2)
            else:
                x[si]=bestXX+np.abs(pX[si]-bestXX)*A
            x[si]=np.clip(x[si],lo,hi)
        fits_rest=self._evaluate_population(x[sorted_idx[pNum:]]); evals+=n-pNum
        # awareness
        aware=sorted_idx[:min(20,n)]
        for si in aware:
            if self._is_better(pFit[si],fMin):
                x[si]=bestX+np.random.randn(d)*np.abs(pX[si]-bestX)
            else:
                denom=(pFit[si]-max(pFit)+1e-300)
                x[si]=pX[si]+(2*np.random.random()-1)*np.abs(pX[si]-worst)/denom
            x[si]=np.clip(x[si],lo,hi)
        all_fits=self._evaluate_population(x); evals+=n
        mask=self._better_mask(all_fits,pFit)
        pFit[mask]=all_fits[mask]; pX[mask]=x[mask]
        state.payload.update({"pFit":pFit,"pX":pX})
        return np.hstack([x,all_fits[:,None]]), evals, {}
