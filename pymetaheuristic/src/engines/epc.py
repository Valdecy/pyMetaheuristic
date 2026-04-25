"""pyMetaheuristic src — Emperor Penguin Colony Engine"""
from __future__ import annotations
import math, numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EPCEngine(PortedPopulationEngine):
    """Emperor Penguin Colony — heat-radiation attenuated spiral movement between pair."""
    algorithm_id = "epc"; algorithm_name = "Emperor Penguin Colony"; family = "swarm"
    _REFERENCE   = {"doi": "10.1016/j.knosys.2018.06.001"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, heat_damping=0.95, mutation_factor=0.5, spiral_a=1.0, spiral_b=0.5)

    def _initialize_payload(self, pop):
        return {"heat_radiation": 1.0}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        hd   = float(self._params.get("heat_damping",0.95)); mf=float(self._params.get("mutation_factor",0.5))
        sa   = float(self._params.get("spiral_a",1.0)); sb=float(self._params.get("spiral_b",0.5))
        hr   = float(state.payload.get("heat_radiation",1.0))*hd
        cmf  = mf*(1-t/T)
        evals=0
        for i in range(n):
            for j in range(n):
                if self._is_better(float(pop[j,-1]),float(pop[i,-1])):
                    d = float(np.linalg.norm(pop[j,:-1]-pop[i,:-1]))+1e-30
                    att = math.exp(-hr*d); att = 1.0/(1.0+att) if att>1 else att
                    l=np.random.uniform(-1,1); b_val=sa*math.exp(sb*l)*math.cos(2*math.pi*l)
                    pos = np.clip(pop[i,:-1]+att*(pop[j,:-1]-pop[i,:-1])*b_val+cmf*np.random.normal(0,1,dim)*pop[i,:-1], self._lo, self._hi)
                    fit = float(self.problem.evaluate(pos)); evals+=1
                    if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
        return pop, evals, {"heat_radiation":hr}
