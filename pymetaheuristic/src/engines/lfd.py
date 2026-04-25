"""pyMetaheuristic src — Lévy Flight Distribution Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy1d(d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2); den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta); u=np.random.randn(d)*sigma; v=np.random.randn(d)
    return u/np.abs(v)**(1/beta)

class LFDEngine(PortedPopulationEngine):
    algorithm_id   = "lfd"
    algorithm_name = "Lévy Flight Distribution"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2020.103731"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, threshold=2.0)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        threshold=float(self._params.get("threshold",2.0))
        best_idx=self._best_index(pop[:,-1]); TargetPos=pop[best_idx,:-1].copy(); TargetFit=pop[best_idx,-1]
        new_pos=pop[:,:-1].copy()
        vec_flag=[-1,1]
        for i in range(n):
            S_i=np.zeros(d)
            for j in range(n):
                if i!=j:
                    dis=np.linalg.norm(pop[j,:-1]-pop[i,:-1])
                    if dis<threshold:
                        t_=pop[j,-1]/(pop[i,-1]+1e-300)
                        t_c=np.clip(t_,0.1,1.0)
                        flag=np.random.choice(vec_flag)
                        s_ij=flag*t_c*_levy1d(d)/n
                        S_i+=s_ij
            ra_idx=np.random.randint(n)
            X_rand=pop[ra_idx,:-1]
            levy_step=_levy1d(d)
            X_new=TargetPos+10*S_i+np.random.random()*0.00005*((TargetPos+0.005*X_rand)/2-pop[i,:-1])
            X_new=TargetPos+levy_step*(X_new-TargetPos)
            new_pos[i]=np.clip(X_new,lo,hi)
        new_fits=self._evaluate_population(new_pos); evals+=n
        mask=self._better_mask(new_fits,pop[:,-1])
        pop[mask]=np.hstack([new_pos,new_fits[:,None]])[mask]
        return pop, evals, {}
