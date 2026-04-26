"""pyMetaheuristic src — Egret Swarm Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ESOAEngine(PortedPopulationEngine):
    """Egret Swarm Optimization Algorithm — gradient-estimated individual+group direction flight."""
    algorithm_id = "esoa"; algorithm_name = "Egret Swarm Optimization Algorithm"; family = "swarm"
    _REFERENCE     = {"doi": "10.3390/biomimetics7040144"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, beta1=0.9, beta2=0.99)

    def _initialize_payload(self, pop):
        n,dim=pop.shape[0],self.problem.dimension
        return {"local_best":pop.copy(), "g":np.zeros((n,dim)), "m":np.zeros((n,dim)), "v":np.zeros((n,dim))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        b1=float(self._params.get("beta1",0.9)); b2=float(self._params.get("beta2",0.99)); EPS=1e-30
        lb  = np.asarray(state.payload.get("local_best",pop.copy()), dtype=float)
        g   = np.asarray(state.payload.get("g",np.zeros((n,dim))), dtype=float)
        m   = np.asarray(state.payload.get("m",np.zeros((n,dim))), dtype=float)
        v   = np.asarray(state.payload.get("v",np.zeros((n,dim))), dtype=float)
        order = self._order(pop[:,-1]); best_pos=pop[order[0],:-1].copy(); best_fit=float(pop[order[0],-1])
        evals=0
        for i in range(n):
            pd=(lb[i,:-1]-pop[i,:-1])*(float(lb[i,-1])-float(pop[i,-1])+EPS)
            pd=pd/(np.sum(pd)**2+EPS)
            dp=pd+g[i]
            cd=(best_pos-pop[i,:-1])*(best_fit-float(pop[i,-1])+EPS)
            cd=cd/(np.sum(cd)**2+EPS)
            dg=cd+g[order[0]]
            r1,r2=np.random.random(dim),np.random.random(dim)
            g[i]=(1-r1-r2)*g[i]+r1*dp+r2*dg
            g[i]=g[i]/(np.sum(g[i])+EPS)
            m[i]=b1*m[i]+(1-b1)*g[i]
            v[i]=b2*v[i]+(1-b2)*g[i]**2
            hop=self._span; x0=np.clip(pop[i,:-1]+np.exp(-1/(0.1*T))*0.1*hop*g[i],self._lo,self._hi)
            fit0=float(self.problem.evaluate(x0)); evals+=1
            if self._is_better(fit0,float(pop[i,-1])): pop[i,:-1]=x0; pop[i,-1]=fit0
            r3=np.random.uniform(-np.pi/2,np.pi/2,dim)
            xn=np.clip(pop[i,:-1]+np.tan(r3)*hop/t*0.5, self._lo, self._hi)
            fitn=float(self.problem.evaluate(xn)); evals+=1
            if self._is_better(fitn,float(pop[i,-1])): pop[i,:-1]=xn; pop[i,-1]=fitn
            if self._is_better(float(pop[i,-1]),float(lb[i,-1])): lb[i]=pop[i]
        return pop, evals, {"local_best":lb,"g":g,"m":m,"v":v}
