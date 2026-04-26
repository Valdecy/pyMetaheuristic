"""pyMetaheuristic src — Lion Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class LOAEngine(PortedPopulationEngine):
    """Lion Optimization Algorithm — pride/nomad roaming and mating operators."""
    algorithm_id = "loa"
    algorithm_name = "Lion Optimization Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.jcde.2015.06.003"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, nomad_ratio=0.2, mating_prob=0.4, roaming_prob=0.2, mutation_rate=0.1)

    def _initialize_payload(self, pop):
        n_nomad = int(float(self._params.get("nomad_ratio", 0.2)) * pop.shape[0])
        nomad = np.zeros(pop.shape[0], dtype=bool)
        if n_nomad > 0:
            nomad[np.random.choice(pop.shape[0], n_nomad, replace=False)] = True
        return {"nomad": nomad}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        nomad = np.asarray(state.payload.get("nomad", np.zeros(n, dtype=bool)), dtype=bool)
        if nomad.shape[0] != n: nomad = np.zeros(n, dtype=bool)
        elite = pop[self._order(pop[:, -1])[:max(2, n // 4)], :-1]
        trials = []
        for i in range(n):
            x = pop[i, :-1]
            if nomad[i] or np.random.rand() < float(self._params.get("roaming_prob", 0.2)):
                y = x + np.random.normal(0.0, 0.2 * self._span, dim)
            elif np.random.rand() < float(self._params.get("mating_prob", 0.4)):
                mate = elite[np.random.randint(elite.shape[0])]
                beta = np.random.rand(dim)
                y = beta * x + (1.0 - beta) * mate
            else:
                leader = elite[0]
                y = x + np.random.rand(dim) * (leader - x) + np.random.normal(0.0, 0.03 * self._span, dim)
            mask = np.random.rand(dim) < float(self._params.get("mutation_rate", 0.1))
            y[mask] += np.random.normal(0.0, 0.1 * self._span[mask])
            trials.append(np.clip(y, self._lo, self._hi))
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        # territorial takeover: best nomads can replace worst pride members
        if np.any(nomad) and np.any(~nomad):
            best_nom = np.where(nomad)[0][self._order(pop[nomad, -1])[:max(1, int(nomad.sum()/4))]]
            pride_idx = np.where(~nomad)[0]
            worst_pride = pride_idx[self._order(pop[pride_idx, -1])[-len(best_nom):]]
            for a, b in zip(best_nom, worst_pride):
                if self._is_better(pop[a, -1], pop[b, -1]):
                    pop[[a, b]] = pop[[b, a]]
                    nomad[a], nomad[b] = nomad[b], nomad[a]
        return pop, n, {"nomad": nomad}
