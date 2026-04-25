"""pyMetaheuristic src — Growth Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GOGrowthEngine(PortedPopulationEngine):
    algorithm_id   = "go_growth"
    algorithm_name = "Growth Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2022.110206"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, P1=5, P2=0.001, P3=0.3)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        P1=int(self._params.get("P1",5)); P2=float(self._params.get("P2",0.001))
        P3=float(self._params.get("P3",0.3))
        order=self._order(pop[:,-1]); best=pop[order[0],:-1].copy()
        for i in range(n):
            worst_candidates=order[max(0,n-P1):]
            worst=pop[np.random.choice(worst_candidates),:-1]
            better_candidates=order[1:min(P1+1,n)]
            if len(better_candidates)>0:
                better=pop[np.random.choice(better_candidates),:-1]
            else:
                better=best.copy()
            others=[k for k in range(n) if k!=i]
            if len(others)>=2:
                L1,L2=np.random.choice(others,2,replace=False)
            else:
                L1=L2=i
            G1=best-better; G2=best-worst; G3=better-worst
            G4=pop[L1,:-1]-pop[L2,:-1]
            dist_sum=sum(np.linalg.norm(G) for G in [G1,G2,G3,G4])+1e-300
            LF=[np.linalg.norm(G)/dist_sum for G in [G1,G2,G3,G4]]
            SF=pop[i,-1]/(max(pop[:,-1])+1e-300)
            KA=[LF[k]*SF*[G1,G2,G3,G4][k] for k in range(4)]
            newx=np.clip(pop[i,:-1]+sum(KA),lo,hi)
            new_fit=float(self._evaluate_population(newx[None])[0]); evals+=1
            if pop[i,-1]>new_fit or np.random.random()<P2:
                pop[i]=np.append(newx,new_fit)
        order=self._order(pop[:,-1])
        for i in range(n):
            newx=pop[i,:-1].copy()
            for j in range(d):
                if np.random.random()<P3:
                    R=pop[order[np.random.randint(min(P1,n))],:-1]
                    newx[j]=pop[i,j]+(R[j]-pop[i,j])*np.random.random()
                    AF=0.01+(0.1-0.01)*(1-evals/(n*self._params.get("max_iterations",1000)))
                    if np.random.random()<AF:
                        newx[j]=lo[j]+(hi[j]-lo[j])*np.random.random()
            newx=np.clip(newx,lo,hi)
            new_fit=float(self._evaluate_population(newx[None])[0]); evals+=1
            if pop[i,-1]>new_fit or np.random.random()<P2:
                pop[i]=np.append(newx,new_fit)
        return pop, evals, {}
