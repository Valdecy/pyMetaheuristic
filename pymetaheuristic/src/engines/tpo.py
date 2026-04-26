"""pyMetaheuristic src — Tree Physiology Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TPOEngine(PortedPopulationEngine):
    """Tree Physiology Optimization — carbon/nutrient feedback leaf update around best."""
    algorithm_id   = "tpo"
    algorithm_name = "Tree Physiology Optimization"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1080/0305215X.2017.1305421"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, alpha=0.3, beta=50.0, theta=0.9, n_leafs=5)

    def _initialize_payload(self, pop):
        n=pop.shape[0]; nl=max(2,int(self._params.get("n_leafs",5)))
        roots=np.random.uniform(self._lo,self._hi,(nl,self.problem.dimension))
        return {"roots":roots,"_theta":float(self._params.get("theta",0.9)),"nl":nl}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        alpha=float(self._params.get("alpha",0.3)); beta=float(self._params.get("beta",50.0))
        roots=np.asarray(state.payload.get("roots",np.random.uniform(self._lo,self._hi,(5,dim))),dtype=float)
        theta_=float(state.payload.get("_theta",0.9)); nl=int(state.payload.get("nl",5))
        order=self._order(pop[:,-1]); best_pos=pop[order[0],:-1].copy()
        evals=0; new_pop_arr=pop.copy()
        for i in range(n):
            cg=theta_*best_pos-roots
            roots_old=roots.copy(); roots+=alpha*cg*np.random.uniform(-0.5,0.5,(nl,dim))
            nutr=theta_*(roots-roots_old)
            leaf_pos=np.clip(best_pos+beta*nutr,self._lo,self._hi)
            leaf_fit=self._evaluate_population(leaf_pos); evals+=nl
            best_leaf=int(np.argmin(leaf_fit)) if self.problem.objective=="min" else int(np.argmax(leaf_fit))
            if self._is_better(float(leaf_fit[best_leaf]),float(pop[i,-1])):
                new_pop_arr[i,:-1]=leaf_pos[best_leaf]; new_pop_arr[i,-1]=leaf_fit[best_leaf]
        theta_*=float(self._params.get("theta",0.9))
        return new_pop_arr, evals, {"roots":roots,"_theta":theta_,"nl":nl}
