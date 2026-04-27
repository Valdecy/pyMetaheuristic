"""pyMetaheuristic src — Gazelle Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class GazelleOAEngine(PortedPopulationEngine):
    algorithm_id   = "gazelle_oa"
    algorithm_name = "Gazelle Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-022-07854-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"fit_old": pop[:,-1].copy(), "prey_old": pop[:,:-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        fit_old=state.payload["fit_old"]; prey_old=state.payload["prey_old"]
        top_pos=pop[self._best_index(pop[:,-1]),:-1].copy()
        top_fit=pop[self._best_index(pop[:,-1]),-1]
        Inx=fit_old<pop[:,-1]
        for k in range(n):
            if Inx[k]: pop[k]=np.append(prey_old[k],fit_old[k])
        fit_old=pop[:,-1].copy(); prey_old=pop[:,:-1].copy()
        Elite=np.tile(top_pos,(n,1))
        CF=(1-t/max_iter)**(2*t/max_iter)
        RL=0.05*_levy(n,d,1.5); RB=np.random.randn(n,d)
        PSRs=0.34; S=0.88
        step=np.zeros((n,d))
        for i in range(n):
            for j in range(d):
                R=np.random.random(); r=np.random.random()
                mu=1 if t%2==0 else -1
                if r>0.5:
                    step[i,j]=RB[i,j]*(Elite[i,j]-RB[i,j]*pop[i,j])
                    pop[i,j]+=np.random.random()*R*step[i,j]
                else:
                    if i>n//2:
                        step[i,j]=RB[i,j]*(RL[i,j]*Elite[i,j]-pop[i,j])
                        pop[i,j]=Elite[i,j]+S*mu*CF*step[i,j]
                    else:
                        step[i,j]=RL[i,j]*(Elite[i,j]-RL[i,j]*pop[i,j])
                        pop[i,j]+=S*mu*R*step[i,j]
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n
        pop[:,-1]=new_fits
        for k in range(n):
            if pop[k,-1]<top_fit: top_pos=pop[k,:-1].copy(); top_fit=pop[k,-1]
        Inx=fit_old<pop[:,-1]
        for k in range(n):
            if Inx[k]: pop[k]=np.append(prey_old[k],fit_old[k])
        fit_old=pop[:,-1].copy(); prey_old=pop[:,:-1].copy()
        if np.random.random()<PSRs:
            U=np.random.random((n,d))<PSRs
            pop[:,:-1]+=CF*((lo+np.random.random((n,d))*(hi-lo))*U)
        else:
            r=np.random.random()
            p1=np.random.permutation(n); p2=np.random.permutation(n)
            step=(PSRs*(1-r)+r)*(pop[p1,:-1]-pop[p2,:-1])
            pop[:,:-1]+=step
        pop[:,:-1]=np.clip(pop[:,:-1],lo,hi)
        new_fits=self._evaluate_population(pop[:,:-1]); evals+=n
        pop[:,-1]=new_fits
        state.payload.update({"fit_old":pop[:,-1].copy(),"prey_old":pop[:,:-1].copy()})
        return pop, evals, {}
