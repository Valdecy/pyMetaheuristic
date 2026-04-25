"""pyMetaheuristic src — Snow Ablation Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SnowOAEngine(PortedPopulationEngine):
    algorithm_id   = "snow_oa"
    algorithm_name = "Snow Ablation Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2023.120069"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        n = pop.shape[0]; N1=max(1,n//2)
        order=self._order(pop[:,-1])
        ep=np.array([pop[order[i],:-1] for i in range(min(4,n))])
        return {"Na":N1,"Nb":N1,"elite":ep,"index":list(range(n))}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        N1=max(1,n//2); Na=state.payload["Na"]; Nb=state.payload["Nb"]
        elite=state.payload["elite"]; idx_all=list(range(n))
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        T=np.exp(-t/max_iter); k=1
        DDF=0.35*(1+(5/7)*(np.exp(t/max_iter)-1)**k/(np.exp(1)-1)**k)
        M=DDF*T; X_centroid=np.mean(pop[:,:-1],axis=0)
        RB=np.random.randn(n,d)
        index1=np.random.choice(n,Na,replace=False)
        index2=np.array([i for i in idx_all if i not in index1])
        for i in range(Na):
            r1=np.random.random(); k1=np.random.randint(min(4,len(elite)))
            for j in range(d):
                pop[index1[i],j]=elite[k1,j]+RB[index1[i],j]*(r1*(best_pos[j]-pop[index1[i],j])+(1-r1)*(X_centroid[j]-pop[index1[i],j]))
        if Na<n: Na=min(Na+1,n); Nb=max(Nb-1,0)
        if Nb>=1 and len(index2)>0:
            for i in range(min(Nb,len(index2))):
                r2=2*np.random.random()-1
                for j in range(d):
                    pop[index2[i],j]=M*best_pos[j]+RB[index2[i],j]*(r2*(best_pos[j]-pop[index2[i],j])+(1-r2)*(X_centroid[j]-pop[index2[i],j]))
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=new_fits
        order=self._order(pop[:,-1])
        e4=[pop[order[i],:-1].copy() for i in range(min(4,n))]
        elite=np.array(e4)
        state.payload.update({"Na":Na,"Nb":Nb,"elite":elite})
        return pop, evals, {}
