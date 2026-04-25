"""pyMetaheuristic src — Artificial Ecosystem Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class AEOEngine(PortedPopulationEngine):
    """Artificial Ecosystem Optimization — production, herbivore/carnivore/omnivore consumption."""
    algorithm_id = "aeo"; algorithm_name = "Artificial Ecosystem Optimization"; family = "human"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1; evals=0
        order=self._order(pop[:,-1]); pop=pop[order]  # sort best first (idx 0 = best = producer)
        # Production: update worst agent
        a=(1.0-t/T)*np.random.random()
        pos=np.clip((1-a)*pop[-1,:-1]+a*np.random.uniform(self._lo,self._hi), self._lo, self._hi)
        pop[-1,-1]=float(self.problem.evaluate(pos)); pop[-1,:-1]=pos; evals+=1
        # Consumption
        new_pos=np.empty_like(pop[:-1,:-1])
        for i in range(n-1):
            r=np.random.random(); v1=np.random.normal(); v2=np.random.normal()+1e-30
            c=0.5*v1/abs(v2); j=1 if i==0 else np.random.randint(0,i)
            if r<1/3:
                pos=pop[i,:-1]+c*(pop[i,:-1]-pop[0,:-1])
            elif r<=2/3:
                pos=pop[i,:-1]+c*(pop[i,:-1]-pop[j,:-1])
            else:
                r2=np.random.random()
                pos=pop[i,:-1]+c*(r2*(pop[i,:-1]-pop[0,:-1])+(1-r2)*(pop[i,:-1]-pop[j,:-1]))
            new_pos[i]=np.clip(pos,self._lo,self._hi)
        new_fit=self._evaluate_population(new_pos); evals+=n-1
        mask=self._better_mask(new_fit,pop[:-1,-1])
        pop[:-1][mask]=np.hstack([new_pos,new_fit[:,None]])[mask]
        return pop, evals, {}
