"""pyMetaheuristic src — Bees Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class BEAEngine(PortedPopulationEngine):
    """Bees Algorithm — recruited neighbourhood search around elite sites."""
    algorithm_id = "bea"
    algorithm_name = "Bees Algorithm"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, m=5, e=4, ngh=0.15, nep=4, nsp=2, shrink=0.95)

    def _initialize_payload(self, pop):
        return {"ngh": float(self._params.get("ngh", 0.15))}

    def _step_impl(self, state, pop):
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        m = min(int(self._params.get("m", 5)), pop.shape[0])
        e = min(int(self._params.get("e", 4)), m)
        nep = max(1, int(self._params.get("nep", 4)))
        nsp = max(1, int(self._params.get("nsp", 2)))
        ngh = float(state.payload.get("ngh", self._params.get("ngh", 0.15)))
        new_rows = []
        evals = 0
        for rank in range(m):
            recruits = nep if rank < e else nsp
            center = pop[rank, :-1]
            steps = np.random.uniform(-ngh, ngh, (recruits, self.problem.dimension)) * self._span
            cand = np.clip(center + steps, self._lo, self._hi)
            cand_pop = self._pop_from_positions(cand)
            evals += recruits
            pool = np.vstack((pop[rank:rank+1], cand_pop))
            new_rows.append(pool[self._best_index(pool[:, -1])])
        scouts = max(0, pop.shape[0] - len(new_rows))
        if scouts:
            scout_pop = self._pop_from_positions(self._new_positions(scouts))
            evals += scouts
            new_rows.extend(list(scout_pop))
        new_pop = np.vstack(new_rows)
        return new_pop, evals, {"ngh": max(1e-12, ngh * float(self._params.get("shrink", 0.95)))}
