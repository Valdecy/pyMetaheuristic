"""pyMetaheuristic src — Brown-Bear Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BBOAEngine(PortedPopulationEngine):
    """Brown-Bear Optimization Algorithm — pedal-marking and sniffing foraging phases."""
    algorithm_id   = "bboa"
    algorithm_name = "Brown-Bear Optimization Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        pp      = t / T
        evals   = 0

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0],  :-1].copy()
        wrst_pos = pop[order[-1], :-1].copy()

        # Phase 1: Pedal marking
        pm_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if pp <= 1.0 / 3.0:
                pos = pop[i, :-1] + (-pp * np.random.random(dim) * pop[i, :-1])
            elif pp <= 2.0 / 3.0:
                qq  = pp * np.random.random(dim)
                r   = np.random.randint(1, 3)
                pos = pop[i, :-1] + qq * (best_pos - r * wrst_pos)
            else:
                ww  = 2.0 * pp * np.pi * np.random.random(dim)
                pos = pop[i, :-1] + (ww * best_pos  - np.abs(pop[i, :-1])) \
                                  - (ww * wrst_pos - np.abs(pop[i, :-1]))
            pm_pos[i] = np.clip(pos, self._lo, self._hi)
        pm_fit = self._evaluate_population(pm_pos); evals += n
        mask   = self._better_mask(pm_fit, pop[:, -1])
        pop[mask] = np.hstack([pm_pos, pm_fit[:, None]])[mask]

        # Phase 2: Sniffing
        sn_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            k   = np.random.choice([x for x in range(n) if x != i])
            if self._is_better(float(pop[i, -1]), float(pop[k, -1])):
                pos = pop[i, :-1] + np.random.random() * (pop[i, :-1] - pop[k, :-1])
            else:
                pos = pop[i, :-1] + np.random.random() * (pop[k, :-1] - pop[i, :-1])
            sn_pos[i] = np.clip(pos, self._lo, self._hi)
        sn_fit = self._evaluate_population(sn_pos); evals += n
        mask   = self._better_mask(sn_fit, pop[:, -1])
        pop[mask] = np.hstack([sn_pos, sn_fit[:, None]])[mask]

        return pop, evals, {}
