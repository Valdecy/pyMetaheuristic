"""pyMetaheuristic src — Chimp Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ChOAEngine(PortedPopulationEngine):
    algorithm_id   = "choa"
    algorithm_name = "Chimp Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2020.113338"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        order = self._order(pop[:, -1])
        A_pos = pop[order[0], :-1].copy(); A_sc = pop[order[0], -1]
        B_pos = pop[order[1], :-1].copy() if n>1 else A_pos.copy()
        C_pos = pop[order[2], :-1].copy() if n>2 else A_pos.copy()
        D_pos = pop[order[3], :-1].copy() if n>3 else A_pos.copy()
        f = 2 - t*(2/max_iter)
        r = t/max_iter
        C1G1=1.95-(2*r**(1/3))/(1); C2G1=2*r**(1/3)+0.5
        C1G2=1.95-(2*r**(1/3)); C2G2=2*(r**3)+0.5
        C1G3=(-2*(r**3))+2.5; C2G3=2*r**(1/3)+0.5
        C1G4=(-2*(r**3))+2.5; C2G4=2*(r**3)+0.5
        m_chaos = 0.7  # simplified chaos map
        new_pos = pop[:, :-1].copy()
        for i in range(n):
            for j in range(d):
                A1=2*f*C1G1*np.random.random()-f; C1_=2*C2G1*np.random.random()
                A2=2*f*C1G2*np.random.random()-f; C2_=2*C2G2*np.random.random()
                A3=2*f*C1G3*np.random.random()-f; C3_=2*C2G3*np.random.random()
                A4=2*f*C1G4*np.random.random()-f; C4_=2*C2G4*np.random.random()
                X1=A_pos[j]-A1*abs(C1_*A_pos[j]-m_chaos*pop[i,j])
                X2=B_pos[j]-A2*abs(C2_*B_pos[j]-m_chaos*pop[i,j])
                X3=C_pos[j]-A3*abs(C3_*C_pos[j]-m_chaos*pop[i,j])
                X4=D_pos[j]-A4*abs(C4_*D_pos[j]-m_chaos*pop[i,j])
                new_pos[i,j]=(X1+X2+X3+X4)/4
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
