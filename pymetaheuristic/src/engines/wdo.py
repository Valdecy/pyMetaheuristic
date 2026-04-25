"""pyMetaheuristic src — Wind Driven Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WDOEngine(PortedPopulationEngine):
    """Wind Driven Optimization — air parcel velocity update via atmospheric dynamics."""
    algorithm_id   = "wdo"
    algorithm_name = "Wind Driven Optimization"
    family         = "physics"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        vel = np.random.uniform(-1, 1, (pop.shape[0], self.problem.dimension))
        return {"velocity": vel}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        RT     = float(self._params.get("RT", 3))
        g_c    = float(self._params.get("g_c", 0.2))
        alp    = float(self._params.get("alp", 0.4))
        c_e    = float(self._params.get("c_e", 0.4))
        max_v  = float(self._params.get("max_v", 0.3))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        vel      = np.asarray(state.payload.get("velocity", np.zeros((n, dim))), dtype=float)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            rand_dim  = np.random.randint(dim)
            temp      = vel[i, rand_dim] * np.ones(dim)
            vel[i]    = (1.0 - alp) * vel[i] - g_c * pop[i, :-1] + \
                        (1.0 - 1.0 / (i + 1)) * RT * (best_pos - pop[i, :-1]) + \
                        c_e * temp / (i + 1)
            vel[i]    = np.clip(vel[i], -max_v, max_v)
            new_pos[i] = np.clip(pop[i, :-1] + vel[i], self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"velocity": vel}
