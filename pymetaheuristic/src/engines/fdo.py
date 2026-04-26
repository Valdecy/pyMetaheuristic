"""pyMetaheuristic src — Flying Dobsonflies Optimizer Engine"""
from __future__ import annotations
import math, numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_fdo(dim):
    beta=1.5; s=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    return 0.01*np.random.normal(0,s,dim)/np.abs(np.random.normal(0,1,dim))**(1/beta)

class FDOEngine(PortedPopulationEngine):
    """Flying Dobsonflies Optimizer — fitness-weighted Lévy steps toward global best."""
    algorithm_id = "fdo"; algorithm_name = "Flying Dobsonflies Optimizer"; family = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2020.105574"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, weight_factor=0.1)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        wf  = float(self._params.get("weight_factor", 0.1))
        order = self._order(pop[:,-1]); best_pos = pop[order[0],:-1].copy(); best_fit = float(pop[order[0],-1])
        evals = 0
        for i in range(n):
            fi = float(pop[i,-1]); bf = best_fit
            if self._is_better(fi, bf): fw = 1
            elif abs(fi - bf) < 1e-30: fw = 0
            else: fw = wf*(abs(fi)/abs(bf) if bf != 0 else 0)
            lv = _levy_fdo(dim); dist = best_pos - pop[i,:-1]
            pace = (pop[i,:-1]*lv if fw==1 else dist*lv if fw==0 else dist*fw*np.sign(lv))
            pos = np.clip(pop[i,:-1]+pace, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, fi):
                pop[i,:-1]=pos; pop[i,-1]=fit
            else:
                pos2 = np.clip(pos + dist*fw + pace, self._lo, self._hi)
                fit2 = float(self.problem.evaluate(pos2)); evals+=1
                if self._is_better(fit2, float(pop[i,-1])): pop[i,:-1]=pos2; pop[i,-1]=fit2
                else:
                    pos3 = np.clip(pop[i,:-1]+pop[i,:-1]*_levy_fdo(dim), self._lo, self._hi)
                    fit3 = float(self.problem.evaluate(pos3)); evals+=1
                    if self._is_better(fit3, float(pop[i,-1])): pop[i,:-1]=pos3; pop[i,-1]=fit3
        return pop, evals, {}
