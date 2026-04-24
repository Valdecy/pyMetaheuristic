"""pyMetaheuristic src — Coral Reefs Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class CROEngine(PortedPopulationEngine):
    """Coral Reefs Optimization — broadcast spawning, brooding, settlement, depredation."""
    algorithm_id = "cro"
    algorithm_name = "Coral Reefs Optimization"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=25, broadcast_prob=0.5, mutation_rate=0.1, depredation_prob=0.05, larvae_factor=1.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        larvae_n = max(1, int(n * float(self._params.get("larvae_factor", 1.0))))
        larvae = []
        elite = pop[self._order(pop[:, -1])[:max(2, n // 3)], :-1]
        for _ in range(larvae_n):
            if np.random.rand() < float(self._params.get("broadcast_prob", 0.5)) and elite.shape[0] > 1:
                a, b = elite[np.random.choice(elite.shape[0], 2, replace=False)]
                w = np.random.rand(dim)
                child = w * a + (1.0 - w) * b
            else:
                parent = pop[np.random.randint(n), :-1]
                child = parent.copy()
            mask = np.random.rand(dim) < float(self._params.get("mutation_rate", 0.1))
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            child[mask] += np.random.normal(0.0, 0.1 * self._span[mask])
            larvae.append(np.clip(child, self._lo, self._hi))
        larvae_pop = self._pop_from_positions(np.asarray(larvae))
        evals = larvae_pop.shape[0]
        for larva in larvae_pop:
            w = self._worst_index(pop[:, -1])
            if self._is_better(larva[-1], pop[w, -1]) or np.random.rand() < 0.05:
                pop[w] = larva
        dep = np.random.rand(n) < float(self._params.get("depredation_prob", 0.05))
        if np.any(dep):
            worst = self._order(pop[:, -1])[-int(dep.sum()):]
            repl = self._pop_from_positions(self._new_positions(len(worst)))
            evals += len(worst)
            pop[worst] = repl
        return pop, evals, {}
