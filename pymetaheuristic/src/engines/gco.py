"""pyMetaheuristic src — Germinal Center Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GCOEngine(PortedPopulationEngine):
    """Germinal Center Optimization — DE-like mutation weighted by adaptive life-signal."""
    algorithm_id = "gco"; algorithm_name = "Germinal Center Optimization"; family = "human"
    _REFERENCE   = {"doi": "10.1016/j.ifacol.2018.07.300"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, wf=0.1, cr=0.9)

    def _initialize_payload(self, pop):
        n=pop.shape[0]
        return {"life_signal":np.zeros(n), "cell_counter":np.ones(n,dtype=int)}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        wf=float(self._params.get("wf",0.1)); cr=float(self._params.get("cr",0.9)); EPS=1e-30
        ls=np.asarray(state.payload.get("life_signal",np.zeros(n)), dtype=float)
        cc=np.asarray(state.payload.get("cell_counter",np.ones(n,int)), dtype=int)
        evals=0
        # Dark zone
        for i in range(n):
            if np.random.uniform(0,100)<ls[i]: cc[i]+=1
            elif cc[i]>1: cc[i]-=1
            p=cc/float(cc.sum()+EPS)
            r1,r2,r3=np.random.choice(n,3,replace=False,p=p)
            pos=pop[r1,:-1]+wf*(pop[r2,:-1]-pop[r3,:-1])
            cond=np.random.random(dim)<cr
            pos=np.clip(np.where(cond,pos,pop[i,:-1]),self._lo,self._hi)
            fit=float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit,float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit; ls[i]+=10
        # Light zone
        ls-=10
        fits=pop[:,-1]
        fmax,fmin=float(fits.max()),float(fits.min())
        f_norm=(fits-fmax)/(fmin-fmax+EPS)
        if self.problem.objective!="min": f_norm=1-f_norm
        ls+=10*f_norm
        return pop, evals, {"life_signal":ls,"cell_counter":cc}
