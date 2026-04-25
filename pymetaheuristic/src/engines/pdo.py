"""pyMetaheuristic src — Prairie Dog Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2); den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class PDOEngine(PortedPopulationEngine):
    algorithm_id   = "pdo"
    algorithm_name = "Prairie Dog Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-022-07530-5"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, rho=0.005, epsPD=0.1)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        rho=float(self._params.get("rho",0.005)); epsPD=float(self._params.get("epsPD",0.1))
        mu=(-1 if t%2==0 else 1)
        DS=1.5*np.random.randn()*(1-t/max_iter)**(2*t/max_iter)*mu
        PE=1.5*(1-t/max_iter)**(2*t/max_iter)*mu
        RL=_levy(n,d,1.5)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        TPD=np.tile(best_pos,(n,1))
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            for j in range(d):
                cpd=np.random.random()*((TPD[i,j]-pop[np.random.randint(n),j])/(TPD[i,j]+1e-300))
                P=rho+(pop[i,j]-np.mean(pop[i,:-1]))/(TPD[i,j]*(hi[j]-lo[j])+1e-300)
                eCB=best_pos[j]*P
                if t<max_iter/4:
                    new_pos[i,j]=best_pos[j]-eCB*epsPD-cpd*RL[i,j]
                elif t<max_iter/2:
                    new_pos[i,j]=best_pos[j]*pop[np.random.randint(n),j]*DS*RL[i,j]
                elif t<3*max_iter/4:
                    new_pos[i,j]=best_pos[j]*PE*np.random.random()
                else:
                    new_pos[i,j]=best_pos[j]-eCB*1e-300-cpd*np.random.random()
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
