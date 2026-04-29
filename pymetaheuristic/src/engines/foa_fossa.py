"""pyMetaheuristic src — Fossa Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class FOAFossaEngine(PortedPopulationEngine):
    """
    Fossa Optimization Algorithm (fossa-hunt themed FOA).

 
    """

    algorithm_id = "foa_fossa"
    algorithm_name = "Fossa Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10462-024-10953-0",
        "title": "Fossa optimization algorithm: a new bio-inspired optimizer for engineering applications",
        "authors": "Mohammad Dehghani et al.",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("foa_fossa requires population_size >= 2.")

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        order = self._order(pop[:, -1])
        current = pop.copy()
        trials = np.zeros((n, dim), dtype=float)
        for i in range(n):
            rp = np.random.rand()
            prey_pool = [j for j in order if j != i and self._is_better(current[j, -1], current[i, -1])]
            if rp <= 0.5:
                prey_idx = int(np.random.choice(prey_pool)) if prey_pool else int(order[0])
                prey = current[prey_idx, :-1]
                r = np.random.rand(dim)
                I = np.random.randint(1, 3, size=dim)
                trials[i] = current[i, :-1] + r * (prey - I * current[i, :-1])
            else:
                r = np.random.rand(dim)
                trials[i] = current[i, :-1] + (1.0 - 2.0 * r) * (self._span / float(t))
        trial_pop = self._pop_from_positions(np.clip(trials, self._lo, self._hi))
        mask = self._better_mask(trial_pop[:, -1], current[:, -1])
        current[mask] = trial_pop[mask]
        return current, n, {}
