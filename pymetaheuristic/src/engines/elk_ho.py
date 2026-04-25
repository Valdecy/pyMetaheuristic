"""pyMetaheuristic src — Elk Herd Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ElkHOEngine(PortedPopulationEngine):
    algorithm_id   = "elk_ho"
    algorithm_name = "Elk Herd Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-023-10680-4"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, MalesRate=0.2)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        MalesRate=float(self._params.get("MalesRate",0.2))
        No_Males=max(1,round(n*MalesRate))
        order=self._order(pop[:,-1]); sorted_fit=pop[order,-1]
        BestBull=pop[order[0],:-1].copy()
        # assign families (females to males by fitness proportional)
        TF=np.array([1/(sorted_fit[i]+1e-300) for i in range(No_Males)])
        TF_sum=TF.sum()
        Families=np.zeros(n,dtype=int)
        for i in range(No_Males,n):
            fi=order[i]; r=np.random.random()
            cs=0
            for j in range(No_Males):
                cs+=TF[j]/TF_sum
                if cs>r: Families[fi]=order[j]; break
        new_pop=pop.copy()
        for i in range(n):
            if Families[i]==0 and i not in order[:No_Males]:
                h=np.random.randint(n)
                new_pop[i,:-1]=np.clip(pop[i,:-1]+np.random.random(d)*(pop[h,:-1]-pop[i,:-1]),lo,hi)
            elif i not in order[:No_Males]:
                males_in_family=[k for k in range(n) if Families[k]==Families[i]]
                if len(males_in_family)>1:
                    h=males_in_family[np.random.randint(len(males_in_family))]
                else:
                    h=np.random.randint(n)
                rd=-2+4*np.random.random()
                new_pos=pop[i,:-1]+(pop[Families[i],:-1]-pop[i,:-1])+rd*(pop[h,:-1]-pop[i,:-1])
                new_pop[i,:-1]=np.clip(new_pos,lo,hi)
            else:
                h=np.random.randint(n)
                new_pop[i,:-1]=np.clip(pop[i,:-1]+np.random.random(d)*(pop[h,:-1]-pop[i,:-1]),lo,hi)
        new_fits=self._evaluate_population(new_pop[:,:-1]); evals+=n
        all_pos=np.vstack([pop[:,:-1],new_pop[:,:-1]])
        all_fits=np.concatenate([pop[:,-1],new_fits])
        best_order=np.argsort(all_fits)[:n]
        pop=np.hstack([all_pos[best_order,:],all_fits[best_order,None]])
        return pop, evals, {}
