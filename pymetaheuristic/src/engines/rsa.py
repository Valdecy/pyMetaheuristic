"""pyMetaheuristic src — Reptile Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RSAEngine(PortedPopulationEngine):
    algorithm_id   = "rsa"
    algorithm_name = "Reptile Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.116158"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        Alpha=0.1; Beta=0.005
        best_pos = pop[self._best_index(pop[:,-1]), :-1].copy()
        ES=2*np.random.randn()*(1-t/max_iter)
        new_pos = pop[:,:-1].copy()
        for i in range(1,n):
            for j in range(d):
                rand_idx=np.random.randint(n)
                R=(best_pos[j]-pop[rand_idx,j])/(best_pos[j]+1e-300)
                P=Alpha+(pop[i,j]-np.mean(pop[i,:-1]))/(best_pos[j]*(hi[j]-lo[j])+1e-300)
                Eta=best_pos[j]*P
                if t<max_iter/4:
                    new_pos[i,j]=best_pos[j]-Eta*Beta-R*np.random.random()
                elif t<max_iter/2:
                    new_pos[i,j]=best_pos[j]*pop[np.random.randint(n),j]*ES*np.random.random()
                elif t<3*max_iter/4:
                    new_pos[i,j]=best_pos[j]*P*np.random.random()
                else:
                    new_pos[i,j]=best_pos[j]-Eta*1e-300-R*np.random.random()
            new_pos[i]=np.clip(new_pos[i],lo,hi)
        new_fits=self._evaluate_population(new_pos[1:]); evals+=n-1
        mask=self._better_mask(new_fits,pop[1:,-1])
        combined=np.hstack([new_pos[1:],new_fits[:,None]])
        pop[1:][mask]=combined[mask]
        return pop, evals, {}
