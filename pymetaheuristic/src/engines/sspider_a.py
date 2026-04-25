"""pyMetaheuristic src — Social Spider Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SSPIDERAEngine(PortedPopulationEngine):
    """Social Spider Algorithm — vibration-intensity attenuation on web with mask-based walk."""
    algorithm_id = "sspider_a"; algorithm_name = "Social Spider Algorithm"; family = "swarm"
    _REFERENCE   = {"doi": "10.1016/j.asoc.2015.02.014"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, r_a=1.0, p_c=0.7, p_m=0.1)

    def _initialize_payload(self, pop):
        n, dim = pop.shape[0], self.problem.dimension
        intensity = np.log(1.0/(np.abs(pop[:,-1])+1e-30)+1)
        mask      = np.random.randint(0,2,(n,dim))
        local_vec = np.zeros((n,dim))
        target_sol = pop[:,:-1].copy()
        return {"intensity":intensity, "mask":mask, "local_vec":local_vec, "target_sol":target_sol}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        r_a = float(self._params.get("r_a",1.0)); p_c=float(self._params.get("p_c",0.7)); p_m=float(self._params.get("p_m",0.1))
        intensity = np.asarray(state.payload.get("intensity", np.ones(n)), dtype=float)
        mask      = np.asarray(state.payload.get("mask", np.ones((n,dim),int)), dtype=int)
        local_vec = np.asarray(state.payload.get("local_vec", np.zeros((n,dim))), dtype=float)
        target_sol= np.asarray(state.payload.get("target_sol", pop[:,:-1].copy()), dtype=float)
        EPS = 1e-30
        all_pos = pop[:,:-1]; base_dist = float(np.mean(np.std(all_pos,axis=0)))+EPS
        dists = np.sqrt(((all_pos[:,None,:]-all_pos[None,:,:])**2).sum(axis=2))
        atten = np.exp(-dists/(base_dist*r_a))
        int_recv = (intensity[None,:]*atten).sum(axis=1)
        best_int  = int(np.argmax(int_recv))
        new_pos = np.empty_like(pop[:,:-1]); evals=0
        for i in range(n):
            if intensity[best_int]>intensity[i]: target_sol[i]=pop[best_int,:-1].copy()
            if np.random.random()>p_c: mask[i]=np.where(np.random.random(dim)<p_m,0,1)
            ref = np.where(mask[i]==0, target_sol[i], pop[np.random.randint(n),:-1])
            pos = pop[i,:-1]+np.random.normal()*(pop[i,:-1]-local_vec[i])+(ref-pop[i,:-1])*np.random.normal()
            new_pos[i] = np.clip(pos, self._lo, self._hi)
        new_fit = self._evaluate_population(new_pos); evals=n
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i,-1])):
                local_vec[i] = new_pos[i]-pop[i,:-1]
                intensity[i] = np.log(1/(abs(new_fit[i])+EPS)+1)
                pop[i,:-1]=new_pos[i]; pop[i,-1]=new_fit[i]
        return pop, evals, {"intensity":intensity,"mask":mask,"local_vec":local_vec,"target_sol":target_sol}
