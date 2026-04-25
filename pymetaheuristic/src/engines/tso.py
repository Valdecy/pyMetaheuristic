"""pyMetaheuristic src — Tuna Swarm Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TSOEngine(PortedPopulationEngine):
    """Tuna Swarm Optimization — spiral and parabolic foraging with random migration."""
    algorithm_id   = "tso"
    algorithm_name = "Tuna Swarm Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, aa=0.7, zz=0.05)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        aa      = float(self._params.get("aa", 0.7))
        zz      = float(self._params.get("zz", 0.05))

        C       = t / T
        a1      = aa + (1.0 - aa) * C
        a2      = (1.0 - aa) - (1.0 - aa) * C
        tt      = (1.0 - t / T) ** (t / T)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if i == 0:                              # leader: spiral update
                r1   = np.random.random()
                beta = np.exp(r1 * np.exp(3.0 * np.cos(np.pi * (T - t) / T))) * np.cos(2.0 * np.pi * r1)
                pos  = a1 * (best_pos + beta * np.abs(best_pos - pop[i, :-1])) + a2 * pop[i-1, :-1]
            else:
                if np.random.random() < zz:        # random migration
                    pos = np.random.uniform(self._lo, self._hi)
                else:
                    r1 = np.random.random()
                    beta = np.exp(r1 * np.exp(3.0 * np.cos(np.pi * (T - t) / T))) * np.cos(2.0 * np.pi * r1)
                    if np.random.random() > 0.5:   # spiral following
                        ref = best_pos if np.random.random() < C else np.random.uniform(self._lo, self._hi)
                        pos = a1 * (ref + beta * np.abs(ref - pop[i, :-1])) + a2 * pop[i-1, :-1]
                    else:                          # parabolic updates
                        tf  = np.random.choice([-1, 1])
                        if np.random.random() < 0.5:
                            pos = best_pos + np.random.random(dim) * (best_pos - pop[i, :-1]) \
                                  + tf * tt**2 * (best_pos - pop[i, :-1])
                        else:
                            pos = tf * tt**2 * pop[i, :-1]
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # full replacement (original)
        return pop, n, {}
