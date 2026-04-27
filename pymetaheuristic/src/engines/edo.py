"""pyMetaheuristic src — Exponential Distribution Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EDOEngine(PortedPopulationEngine):
    algorithm_id   = "edo"
    algorithm_name = "Exponential Distribution Optimizer"
    family         = "math"
    _REFERENCE     = {"doi": "10.1007/s10462-023-10403-9"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"memory": pop[:, :-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        memory = state.payload["memory"]
        order=self._order(pop[:,-1]); pop=pop[order]; memory=memory[order]
        d_val=(1-t/max_iter)
        f=2*np.random.random()-1
        a=f**10; b=f**5; c=d_val*f
        X_guide=np.mean(pop[:3,:-1],axis=0)
        V=np.zeros((n,d))
        for i in range(n):
            alpha=np.random.random()
            if alpha<0.5:
                if np.allclose(memory[i],pop[i,:-1]):
                    Mu=(X_guide+memory[i])/2
                    ExPrate=1/(Mu+1e-300); variance=1/(ExPrate**2)
                    V[i]=a*(memory[i]-variance)+b*X_guide
                else:
                    Mu=(X_guide+memory[i])/2
                    ExPrate=1/(Mu+1e-300); variance=1/(ExPrate**2)
                    phi=np.random.random()
                    V[i]=b*(memory[i]-variance)+np.log(max(phi,1e-300))*pop[i,:-1]
            else:
                M=np.mean(pop[:,:-1],axis=0)
                s=np.random.choice(n,2,replace=False)
                D1=M-pop[s[0],:-1]; D2=M-pop[s[1],:-1]
                Z1=M-D1+D2; Z2=M-D2+D1
                V[i]=pop[i,:-1]+c*Z1+(1-c)*Z2-M
            V[i]=np.clip(V[i],lo,hi)
        memory=V.copy()
        new_fits=self._evaluate_population(V); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([V,new_fits[:,None]])[mask]
        state.payload["memory"]=memory
        return pop, evals, {}
