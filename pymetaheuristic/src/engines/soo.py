"""pyMetaheuristic src — Star Oscillator Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SOOEngine(PortedPopulationEngine):
    """Star Oscillator Optimization — sine/cosine oscillation around top-3 stars."""
    algorithm_id   = "soo"
    algorithm_name = "Star Oscillator Optimization"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1007/s10586-024-04976-5"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, initial_period=3)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        ip      = float(self._params.get("initial_period", 3))
        caf     = 2.0 * np.pi / (ip + 0.001 * t)
        scaler  = 2.0 * (1.0 - t / T)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # Phase 1: oscillatory position update
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r1 = np.random.random(dim); r2 = np.random.random(dim); r3 = np.random.random(dim)
            osc1  = scaler*(caf*r1-1)*(pop[i,:-1] - np.abs(r1*np.sin(r2)*np.abs(r3*best_pos)))
            osc2  = scaler*(caf*r1-1)*(pop[i,:-1] - np.abs(r1*np.cos(r2)*np.abs(r3*best_pos)))
            pos   = r3*(best_pos - r1*r3*osc1 + best_pos - r2*r3*osc2) / 2.0
            new_pos[i] = np.clip(pos, self._lo, self._hi)
        new_fit = self._evaluate_population(new_pos); evals += n
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = np.hstack([new_pos, new_fit[:, None]])[mask]

        # Phase 2: top-3 average oscillatory update
        order2   = self._order(pop[:, -1])
        top3_pos = pop[order2[:3], :-1].mean(axis=0)
        upd_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            others = [k for k in range(n) if k != i]
            r1, r2, r3 = np.random.choice(others, 3, replace=False)
            rf   = np.random.random()
            pos  = top3_pos + 0.5*(np.sin(rf*np.pi)*(pop[r1,:-1]-pop[r2,:-1]) +
                                   np.cos((1-rf)*np.pi)*(pop[r1,:-1]-pop[r3,:-1]))
            pos  = np.where(np.random.random(dim) <= 0.5, pos, pop[i,:-1])
            upd_pos[i] = np.clip(pos, self._lo, self._hi)
        upd_fit = self._evaluate_population(upd_pos); evals += n
        mask2   = self._better_mask(upd_fit, pop[:, -1])
        pop[mask2] = np.hstack([upd_pos, upd_fit[:, None]])[mask2]
        return pop, evals, {}
