"""pyMetaheuristic src — Dynamic Differential Annealed Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class DDAOEngine(PortedPopulationEngine):
    algorithm_id   = "ddao"
    algorithm_name = "Dynamic Differential Annealed Optimization"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2020.106392"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, T0=2000.0, alpha=0.995, MaxSubIt=50)

    def _initialize_payload(self, pop):
        return {"T": float(self._params.get("T0",2000.0))}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        alpha=float(self._params.get("alpha",0.995))
        MaxSubIt=int(self._params.get("MaxSubIt",50))
        T=state.payload["T"]
        sub_positions=np.random.uniform(lo,hi,(MaxSubIt,d))
        sub_fits=self._evaluate_population(sub_positions); evals+=MaxSubIt
        best_sub_idx=np.argmin(sub_fits); bnew_pos=sub_positions[best_sub_idx]
        kk=np.random.randint(n); bb=np.random.randint(n)
        t=state.step
        if t%2==1:
            Mnew_pos=pop[kk,:-1]-pop[bb,:-1]+bnew_pos
        else:
            Mnew_pos=pop[kk,:-1]-pop[bb,:-1]+bnew_pos*np.random.random()
        Mnew_pos=np.clip(Mnew_pos,lo,hi)
        Mnew_fit=float(self._evaluate_population(Mnew_pos[None])[0]); evals+=1
        for i in range(n):
            if self._is_better(Mnew_fit,pop[i,-1]):
                pop[i]=np.append(Mnew_pos,Mnew_fit)
            else:
                DELTA=Mnew_fit-pop[i,-1]
                P=np.exp(-DELTA/(T+1e-300))
                if np.random.random()<=P:
                    pop[-1]=np.append(Mnew_pos,Mnew_fit)
        T*=alpha; state.payload["T"]=T
        return pop, evals, {}
