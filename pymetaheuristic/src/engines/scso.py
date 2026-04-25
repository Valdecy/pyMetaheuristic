"""pyMetaheuristic src — Sand Cat Swarm Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SCSoEngine(PortedPopulationEngine):
    """Sand Cat Swarm Optimization — sensory range guiding exploration vs. exploitation."""
    algorithm_id   = "scso"
    algorithm_name = "Sand Cat Swarm Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00366-022-01604-x"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, ss=2.0)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        # roulette probability vector p ∝ 1/(1..n)
        pp = 1.0 / (np.arange(1, n + 1, dtype=float))
        pp /= pp.sum()
        return {"pp": pp}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        ss      = float(self._params.get("ss", 2.0))
        pp      = np.asarray(state.payload.get("pp", np.ones(n) / n))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        guides_r = ss - ss * t / T

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r  = np.random.random() * guides_r
            R  = 2.0 * guides_r * np.random.random() - guides_r
            pos = pop[i, :-1].copy()
            for j in range(dim):
                # roulette angle selection
                teta = np.searchsorted(np.cumsum(pp), np.random.random())
                teta = float(teta) * 2.0 * np.pi / n
                if abs(R) <= 1.0:                           # exploitation
                    rand_p = abs(np.random.random() * best_pos[j] - pop[i, j])
                    pos[j] = best_pos[j] - r * rand_p * np.cos(teta)
                else:                                       # exploration
                    cp = np.random.randint(n)
                    pos[j] = r * (pop[cp, j] - np.random.random() * pop[i, j])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # replace all (original behaviour)
        return pop, n, {"pp": pp}
