"""pyMetaheuristic src — Electromagnetic Field Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EFOEngine(PortedPopulationEngine):
    """Electromagnetic Field Optimization — positive, neutral, negative field interactions."""
    algorithm_id   = "efo"
    algorithm_name = "Electromagnetic Field Optimization"
    family         = "physics"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, r_rate=0.3, ps_rate=0.85,
                     p_field=0.1, n_field=0.45)

    _PHI = (1.0 + np.sqrt(5.0)) / 2.0   # golden ratio

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        r_rate   = float(self._params.get("r_rate",  0.3))
        ps_rate  = float(self._params.get("ps_rate", 0.85))
        p_field  = float(self._params.get("p_field", 0.1))
        n_field  = float(self._params.get("n_field", 0.45))

        order     = self._order(pop[:, -1])
        sorted_pop = pop[order]                                 # best → worst

        p_end = max(1, int(n * p_field))                       # positive field top
        n_start = min(n - 1, int(n * (1.0 - n_field)))         # negative field bottom
        mid_end = max(p_end + 1, n_start)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r_idx1 = np.random.randint(0, p_end)
            r_idx2 = np.random.randint(n_start, n)
            r_idx3 = np.random.randint(p_end, mid_end) if mid_end > p_end else r_idx1

            if np.random.random() < ps_rate:
                pos = (sorted_pop[r_idx1, :-1] +
                       self._PHI * np.random.random() * (sorted_pop[order[0], :-1] - sorted_pop[r_idx3, :-1]) -
                       np.random.random() * (sorted_pop[r_idx2, :-1] - sorted_pop[r_idx3, :-1]))
            else:
                pos = np.random.uniform(self._lo, self._hi)

            # Mutation (r_rate) — reset one random dimension
            if np.random.random() < r_rate:
                ri  = np.random.randint(dim)
                pos[ri] = np.random.uniform(self._lo[ri], self._hi[ri])

            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
