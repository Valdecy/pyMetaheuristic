"""pyMetaheuristic src — Artificial Lemming Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class ALAEngine(PortedPopulationEngine):
    algorithm_id   = "ala"
    algorithm_name = "Artificial Lemming Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-024-11023-7"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        best_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        RB=np.random.randn(n,d); F=np.random.choice([-1,1])
        theta=2*np.arctan(1-t/max_iter)
        new_pos=np.empty_like(pop[:,:-1])
        for i in range(n):
            E=2*np.log(1/max(np.random.random(),1e-300))*theta
            if E>1:
                if np.random.random()<0.3:
                    r1=2*np.random.random(d)-1
                    ra=pop[np.random.randint(n),:-1]
                    new_pos[i]=best_pos+F*RB[i]*(r1*(best_pos-pop[i,:-1])+(1-r1)*(pop[i,:-1]-ra))
                else:
                    r2=np.random.random()*(1+np.sin(0.5*t))
                    ra=pop[np.random.randint(n),:-1]
                    new_pos[i]=pop[i,:-1]+F*r2*(best_pos-ra)
            else:
                if np.random.random()<0.5:
                    radius=np.linalg.norm(best_pos-pop[i,:-1])
                    r3=np.random.random()
                    spiral=radius*(np.sin(2*np.pi*r3)+np.cos(2*np.pi*r3))
                    new_pos[i]=best_pos+F*pop[i,:-1]*spiral*np.random.random()
                else:
                    G=2*np.sign(np.random.random()-0.5)*(1-t/max_iter)
                    new_pos[i]=best_pos+F*G*_levy1d(d)*(best_pos-pop[i,:-1])
        new_pos=np.clip(new_pos,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
