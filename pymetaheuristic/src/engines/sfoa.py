"""pyMetaheuristic src — Starfish Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SFOAEngine(PortedPopulationEngine):
    """Starfish Optimization Algorithm — cosine-arm exploration and five-arm exploitation."""
    algorithm_id   = "sfoa"
    algorithm_name = "Starfish Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-024-10694-1"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, gp=0.5)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        gp = float(self._params.get("gp",0.5))
        order=self._order(pop[:,-1]); best_pos=pop[order[0],:-1].copy()
        theta=np.pi/2*t/T; tEO=(T-t)/T*np.cos(theta)
        evals=0
        if np.random.random()<gp:  # exploration
            new_pos=np.empty_like(pop[:,:-1])
            for i in range(n):
                pos=pop[i,:-1].copy()
                if dim>5:
                    jp=np.random.choice(dim,5,replace=False)
                    pm=(2*np.random.random(dim)-1)*np.pi
                    p1=pos+pm*(best_pos-pos)*np.cos(theta); p2=pos-pm*(best_pos-pos)*np.sin(theta)
                    p=np.where(np.random.random(dim)<gp,p1,p2); pos[jp]=p[jp]
                    oob=(pos[jp]<self._lo[jp])|(pos[jp]>self._hi[jp]); pos[jp]=np.where(oob,pop[i,jp],pos[jp])
                else:
                    jp=np.random.randint(dim)
                    im=np.random.choice(n,2,replace=False)
                    d1=pop[im[0],jp]-pos[jp]; d2=pop[im[1],jp]-pos[jp]
                    pos[jp]=tEO*pos[jp]+(2*np.random.random()-1)*d1+(2*np.random.random()-1)*d2
                    if pos[jp]>self._hi[jp] or pos[jp]<self._lo[jp]: pos[jp]=pop[i,jp]
                new_pos[i]=np.clip(pos,self._lo,self._hi)
            new_fit=self._evaluate_population(new_pos); evals+=n
            mask=self._better_mask(new_fit,pop[:,-1]); pop[mask]=np.hstack([new_pos,new_fit[:,None]])[mask]
        else:  # exploitation — five arms
            df=np.random.choice(n,5,replace=False)
            arms=[best_pos-pop[df[i],:-1] for i in range(5)]
            new_pos=np.empty_like(pop[:,:-1])
            for i in range(n):
                arm_idx=np.random.randint(5)
                pos=np.clip(pop[i,:-1]+np.random.random(dim)*arms[arm_idx],self._lo,self._hi)
                new_pos[i]=pos
            new_fit=self._evaluate_population(new_pos); evals+=n
            mask=self._better_mask(new_fit,pop[:,-1]); pop[mask]=np.hstack([new_pos,new_fit[:,None]])[mask]
        return pop, evals, {}
