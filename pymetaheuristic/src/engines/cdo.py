"""pyMetaheuristic src — Cheetah Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CDOEngine(PortedPopulationEngine):
    """Cheetah Optimizer — three-speed spiral attack using top-3 best positions."""
    algorithm_id = "cdo"; algorithm_name = "Cheetah Optimizer"; family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        order = self._order(pop[:,-1])
        b1=pop[order[0],:-1]; b2=pop[order[1],:-1] if n>1 else b1; b3=pop[order[2],:-1] if n>2 else b2
        a = 3.0-3.0*t/T
        a1=np.log10(15999*np.random.random()+16000); a2=np.log10(269999*np.random.random()+270000); a3=np.log10(299999*np.random.random()+300000)
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            r1,r2=np.random.random(dim),np.random.random(dim)
            pa=np.pi*r1**2/(0.25*a1)-a*np.random.random(dim); c1=r2**2*np.pi
            pos_a=0.25*(b1-pa*np.abs(c1*b1-pop[i,:-1]))
            r3,r4=np.random.random(dim),np.random.random(dim)
            pb=np.pi*r3**2/(0.5*a2)-a*np.random.random(dim); c2=r4**2*np.pi
            pos_b=0.5*(b2-pb*np.abs(c2*b2-pop[i,:-1]))
            r5,r6=np.random.random(dim),np.random.random(dim)
            pc=np.pi*r5**2/a3-a*np.random.random(dim); c3=r6**2*np.pi
            pos_c=b3-pc*np.abs(c3*b3-pop[i,:-1])
            new_pos[i]=np.clip((pos_a+pos_b+pos_c)/3, self._lo, self._hi)
        new_fit=self._evaluate_population(new_pos)
        pop=np.hstack([new_pos, new_fit[:,None]]); return pop, n, {}
