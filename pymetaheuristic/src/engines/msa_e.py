"""pyMetaheuristic src — Moth Search Algorithm Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_msa(dim: int, scale: float) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return scale * u / v**(1/beta)

class MSAEngine(PortedPopulationEngine):
    """Moth Search Algorithm — Lévy-flight partition with golden-ratio exploitation."""
    algorithm_id   = "msa_e"
    algorithm_name = "Moth Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s12293-016-0212-3"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_best=5, partition=0.5, max_step_size=1.0)
    _GOLDEN   = (1.0 + 5.0**0.5) / 2.0

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        n_best      = min(int(self._params.get("n_best", 5)), n // 2)
        partition   = float(self._params.get("partition", 0.5))
        max_step    = float(self._params.get("max_step_size", 1.0))
        n_moth1     = max(1, int(np.ceil(partition * n)))

        order    = self._order(pop[:, -1])
        pop      = pop[order]                          # keep sorted best-first
        elites   = pop[:n_best].copy()
        best_pos = pop[0, :-1].copy()
        scale    = max_step / (t + 1)

        new_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            if i < n_moth1:                            # Lévy migration
                pos = pop[i, :-1] + np.random.random(dim) * _levy_msa(dim, scale)
            else:                                      # golden-ratio exploitation
                r   = np.random.random(dim)
                tc1 = pop[i, :-1] + r * self._GOLDEN * (best_pos - pop[i, :-1])
                tc2 = pop[i, :-1] + r * (1.0 / self._GOLDEN) * (best_pos - pop[i, :-1])
                pos = np.where(np.random.random(dim) < 0.5, tc2, tc1)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        # Re-sort and reinsert elites at the back (replace worst)
        order2  = self._order(pop[:, -1])
        pop     = pop[order2]
        for k in range(n_best):
            pop[n - 1 - k] = elites[k]
        return pop, n, {}
