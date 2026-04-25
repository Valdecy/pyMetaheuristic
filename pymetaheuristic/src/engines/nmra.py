"""pyMetaheuristic src — Naked Mole-Rat Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class NMRAEngine(PortedPopulationEngine):
    """Naked Mole-Rat Algorithm — breeder/worker role division for exploitation and exploration."""
    algorithm_id   = "nmra"
    algorithm_name = "Naked Mole-Rat Algorithm"
    family         = "swarm"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pb=0.75)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        pb      = float(self._params.get("pb", 0.75))
        order   = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        size_b  = max(1, int(pb * n))          # breeders
        # Sort so breeders are the best
        pop     = pop[order]

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if i < size_b:                     # breeding operator
                alpha = np.random.random()
                pos   = (1.0 - alpha) * pop[i, :-1] + alpha * (best_pos - pop[i, :-1])
            else:                              # working operator
                worker_pool = [k for k in range(size_b, n) if k != i]
                if len(worker_pool) < 2:
                    worker_pool = [k for k in range(n) if k != i]
                t1, t2 = np.random.choice(worker_pool, 2, replace=False)
                pos = pop[i, :-1] + np.random.random() * (pop[t1, :-1] - pop[t2, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
