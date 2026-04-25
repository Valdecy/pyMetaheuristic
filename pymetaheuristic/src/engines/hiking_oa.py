"""pyMetaheuristic src — Hiking Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HikingOAEngine(PortedPopulationEngine):
    algorithm_id   = "hiking_oa"
    algorithm_name = "Hiking Optimization Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2024.111880"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        best_pos = pop[self._best_index(pop[:,-1]), :-1].copy()
        new_pos = np.empty_like(pop[:,:-1])
        for j in range(n):
            theta = np.random.randint(0,51)
            s = np.tan(np.deg2rad(theta))
            SF = np.random.choice([1,2])
            Vel = 6*np.exp(-3.5*np.abs(s+0.05))
            newVel = Vel + np.random.random(d)*(best_pos-SF*pop[j,:-1])
            new_pos[j] = np.clip(pop[j,:-1]+newVel, lo, hi)
        new_fits = self._evaluate_population(new_pos); evals += n
        mask = self._better_mask(new_fits, pop[:,-1])
        pop[mask] = np.hstack([new_pos, new_fits[:,None]])[mask]
        return pop, evals, {}
