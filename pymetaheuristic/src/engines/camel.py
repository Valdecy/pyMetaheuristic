"""pyMetaheuristic src — Camel Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class CamelEngine(PortedPopulationEngine):
    """Camel Algorithm — endurance, supply and visibility driven desert walks."""
    algorithm_id = "camel"
    algorithm_name = "Camel Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "10.13140/RG.2.2.21814.56649"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, endurance_init=1.0, supply_init=1.0, visibility=0.5, burden_factor=0.1, death_rate=0.05)

    def _initialize_payload(self, pop):
        return {"endurance": np.full(pop.shape[0], float(self._params.get("endurance_init", 1.0))), "supply": np.full(pop.shape[0], float(self._params.get("supply_init", 1.0)))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        endurance = np.asarray(state.payload.get("endurance", np.ones(n)), dtype=float)
        supply = np.asarray(state.payload.get("supply", np.ones(n)), dtype=float)
        if endurance.shape[0] != n: endurance = np.ones(n)
        if supply.shape[0] != n: supply = np.ones(n)
        best = pop[self._best_index(pop[:, -1]), :-1]
        trials = []
        for i in range(n):
            temp = np.random.rand()
            visibility = float(self._params.get("visibility", 0.5)) * supply[i]
            burden = float(self._params.get("burden_factor", 0.1)) * temp
            step = endurance[i] * (visibility * np.random.rand(dim) * (best - pop[i, :-1]) + np.random.normal(0, 0.05, dim) * self._span) - burden
            trials.append(np.clip(pop[i, :-1] + step, self._lo, self._hi))
            endurance[i] *= 0.99 - 0.1 * temp
            supply[i] *= 0.995
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        endurance[mask] = float(self._params.get("endurance_init", 1.0))
        supply[mask] = float(self._params.get("supply_init", 1.0))
        weak = (endurance <= 0.05) | (supply <= 0.05) | (np.random.rand(n) < float(self._params.get("death_rate", 0.05))/max(n,1))
        if np.any(weak):
            repl = self._pop_from_positions(self._new_positions(int(weak.sum())))
            pop[weak] = repl
            endurance[weak] = float(self._params.get("endurance_init", 1.0)); supply[weak] = float(self._params.get("supply_init", 1.0))
            return pop, n + int(weak.sum()), {"endurance": endurance, "supply": supply}
        return pop, n, {"endurance": endurance, "supply": supply}
