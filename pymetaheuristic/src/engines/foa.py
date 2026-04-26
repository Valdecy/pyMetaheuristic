"""pyMetaheuristic src — Forest Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class FOAEngine(PortedPopulationEngine):
    """Forest Optimization Algorithm — local/global seeding with tree ages."""
    algorithm_id = "foa"
    algorithm_name = "Forest Optimization Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2014.05.009"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=10, lifetime=3, area_limit=10, local_seeding_changes=1, global_seeding_changes=1, transfer_rate=0.1)

    def _initialize_payload(self, pop):
        return {"age": np.zeros(pop.shape[0], dtype=int)}

    def _seed(self, parent, changes):
        child = parent.copy()
        dims = np.random.choice(self.problem.dimension, size=max(1, min(changes, self.problem.dimension)), replace=False)
        child[dims] += np.random.normal(0.0, 0.1 * self._span[dims])
        return np.clip(child, self._lo, self._hi)

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        age = np.asarray(state.payload.get("age", np.zeros(n)), dtype=int)
        if age.shape[0] != n:
            age = np.zeros(n, dtype=int)
        age += 1
        local_changes = int(self._params.get("local_seeding_changes", 1))
        seedlings = [self._seed(pop[i, :-1], local_changes) for i in range(n) if age[i] <= int(self._params.get("lifetime", 3))]
        evals = 0
        if seedlings:
            seed_pop = self._pop_from_positions(np.asarray(seedlings)); evals += len(seedlings)
            pop = np.vstack((pop, seed_pop))
            age = np.concatenate((age, np.zeros(len(seedlings), dtype=int)))
        full_pop = pop.copy()
        order = self._order(pop[:, -1])
        area_limit = max(2, int(self._params.get("area_limit", n)))
        keep = order[:area_limit]
        candidate_pool = order[area_limit:]
        pop, age = pop[keep].copy(), age[keep].copy()
        transferred = max(1, int(float(self._params.get("transfer_rate", 0.1)) * max(1, len(candidate_pool))))
        global_changes = int(self._params.get("global_seeding_changes", 1))
        if len(candidate_pool):
            chosen = np.random.choice(candidate_pool, size=min(transferred, len(candidate_pool)), replace=False)
            glob = [self._seed(full_pop[i, :-1], global_changes) for i in chosen]
        else:
            glob = [self._new_positions(1)[0] for _ in range(max(1, n - pop.shape[0]))]
        if glob:
            glob_pop = self._pop_from_positions(np.asarray(glob)); evals += len(glob)
            pop = np.vstack((pop, glob_pop)); age = np.concatenate((age, np.zeros(len(glob), dtype=int)))
        if pop.shape[0] < n:
            repl = self._pop_from_positions(self._new_positions(n - pop.shape[0])); evals += n - pop.shape[0]
            pop = np.vstack((pop, repl)); age = np.concatenate((age, np.zeros(repl.shape[0], dtype=int)))
        order = self._order(pop[:, -1])[:n]
        return pop[order], evals, {"age": age[order]}
