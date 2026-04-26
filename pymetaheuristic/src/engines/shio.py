"""pyMetaheuristic src — Spotted Hyena Inspired Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SHIOEngine(PortedPopulationEngine):
    """Spotted Hyena Inspired Optimizer — three-best encircling attack (CDO-like) with shrinking a."""
    algorithm_id = "shio"; algorithm_name = "Spotted Hyena Inspired Optimizer"; family = "math"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2017.05.014"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension; evals=0
        order=self._order(pop[:,-1])
        b1=pop[order[0],:-1]; b2=pop[order[1],:-1] if n>1 else b1; b3=pop[order[2],:-1] if n>2 else b2
        a=1.5
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            a=max(0.0,a-0.04)
            r1=np.random.random(dim); r2=np.random.random(dim); r3=np.random.random(dim)
            x1=b1+(a*2*r1-a)*np.abs(r1*b1-pop[i,:-1])
            x2=b2+(a*2*r2-a)*np.abs(r2*b2-pop[i,:-1])
            x3=b3+(a*2*r3-a)*np.abs(r3*b3-pop[i,:-1])
            new_pos[i]=np.clip((x1+x2+x3)/3, self._lo, self._hi)
        new_fit=self._evaluate_population(new_pos)
        pop=np.hstack([new_pos,new_fit[:,None]]); return pop, n, {}
