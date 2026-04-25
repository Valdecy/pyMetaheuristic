"""pyMetaheuristic src — Artificial Hummingbird Algorithm / Affix Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class AFTEngine(PortedPopulationEngine):
    """Affix Optimization (AFT) — perception-guided thief tracking with local best memory."""
    algorithm_id   = "aft"
    algorithm_name = "Affix Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s00521-021-06392-x"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"local_best": pop.copy()}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1

        lb  = np.asarray(state.payload.get("local_best", pop.copy()), dtype=float)
        Pp  = 0.1 * np.log(2.75 * (t / T) ** 0.1 + 1e-30)
        Td  = 2.0 * np.exp(-2.0 * (t / T) ** 2)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        followers = np.random.randint(0, n, size=n)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            direction = np.sign(np.random.random() - 0.5)
            lb_i = lb[i, :-1]
            lb_f = lb[followers[i], :-1]
            movement = (Td * (lb_i - pop[i, :-1]) * np.random.random() +
                        Td * (pop[i, :-1] - lb_f) * np.random.random())
            if np.random.random() >= 0.5:
                if np.random.random() > Pp:
                    pos = best_pos + movement * direction
                else:
                    pos = self._lo + Td * self._span * np.random.random(dim)
            else:
                pos = best_pos - movement * direction
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        # Update local bests (greedy)
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(lb[i, -1])):
                lb[i] = new_pop[i]
        pop = new_pop          # full replacement (original paper replaces all)
        return pop, n, {"local_best": lb}
