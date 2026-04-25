"""pyMetaheuristic src — Sinh Cosh Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SCHOEngine(PortedPopulationEngine):
    algorithm_id   = "scho"
    algorithm_name = "Sinh Cosh Optimizer"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2023.111081"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        n, d = pop.shape[0], self.problem.dimension
        return {"dest2": pop[self._order(pop[:,-1])[1] if len(pop)>1 else 0, :-1].copy(),
                "BS_next": int(np.floor(self._params.get("max_iterations",1000)/1.55)+1),
                "T": int(np.floor(1.2+self._params.get("max_iterations",1000)/2.25)),
                "lb2": self._lo.copy(), "ub2": self._hi.copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        dest2=state.payload["dest2"]; T_thresh=state.payload["T"]
        BS_next=state.payload["BS_next"]; lb2=state.payload["lb2"]; ub2=state.payload["ub2"]
        u=0.388; m=0.45; nn=0.5; p_p=10; q_p=9; Alpha=4.6; Beta_=1.55
        for i in range(n):
            for j in range(d):
                cosh2=(np.exp(t/max_iter)+np.exp(-t/max_iter))/2
                sinh2=(np.exp(t/max_iter)-np.exp(-t/max_iter))/2
                r1=np.random.random()
                A=(p_p-q_p*(t/max_iter)**(cosh2/sinh2))*r1
                if t<=T_thresh:
                    r2,r3,r4,r5=np.random.random(),np.random.random(),np.random.random(),np.random.random()
                    a1=3*(-1.3*t/max_iter+m)
                    sinh_=( np.exp(r3)-np.exp(-r3))/2; cosh_=(np.exp(r3)+np.exp(-r3))/2
                    W1=r2*a1*(cosh_+u*sinh_-1)
                    if A>1:
                        pop[i,j]=best_pos[j]+r4*W1*pop[i,j]*(1 if r5<=0.5 else -1)
                    else:
                        W3=r2*a1*(cosh_+u*sinh_)
                        pop[i,j]=best_pos[j]+r4*W3*pop[i,j]*(1 if r5<=0.5 else -1)
                else:
                    r2,r3,r4,r5=np.random.random(),np.random.random(),np.random.random(),np.random.random()
                    a2=2*(-t/max_iter+nn)
                    W2=r2*a2
                    sinh_=(np.exp(r3)-np.exp(-r3))/2; cosh_=(np.exp(r3)+np.exp(-r3))/2
                    if A<1:
                        pop[i,j]+=r5*sinh_/cosh_*abs(W2*best_pos[j]-pop[i,j])
                    else:
                        sign=1 if r4<=0.5 else -1
                        pop[i,j]+=sign*abs(0.003*W2*best_pos[j]-pop[i,j])
        pop[:,:-1]=np.clip(pop[:,:-1],lb2,ub2)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n; pop[:,-1]=new_fits
        if t==BS_next:
            order=self._order(pop[:,-1])
            dest2=pop[order[1] if len(order)>1 else order[0],:-1].copy()
            new_BS=BS_next+int(np.floor((max_iter-BS_next)/Alpha))
            best_pos2=pop[order[0],:-1].copy()
            lb2=np.clip(best_pos2-(1-t/max_iter)*np.abs(best_pos2-dest2),lo,hi)
            ub2=np.clip(best_pos2+(1-t/max_iter)*np.abs(best_pos2-dest2),lo,hi)
            state.payload.update({"dest2":dest2,"BS_next":new_BS,"lb2":lb2,"ub2":ub2})
        return pop, evals, {}
