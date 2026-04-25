"""pyMetaheuristic src — Spotted Hyena Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SHOEngine(PortedPopulationEngine):
    """Spotted Hyena Optimizer — encircling and hunting prey with pack coordination."""
    algorithm_id   = "sho"
    algorithm_name = "Spotted Hyena Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2017.05.014"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, h_factor=5.0, n_trials=10)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim    = pop.shape[0], self.problem.dimension
        T         = max(1, self.config.max_steps or 500)
        t         = state.step + 1
        h_factor  = float(self._params.get("h_factor", 5.0))
        n_trials  = int(self._params.get("n_trials", 10))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        hh       = h_factor - t * (h_factor / T)
        evals    = 0

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            rd1 = np.random.random(dim)
            rd2 = np.random.random(dim)
            B   = 2.0 * rd1
            E   = 2.0 * hh * rd2 - hh

            if np.random.random() < 0.5:
                D_h = np.abs(np.dot(B, best_pos) - pop[i, :-1])
                pos = best_pos - E * D_h
            else:
                # Collect N hyenas around prey
                N = 1
                circle = []
                for _ in range(n_trials):
                    pos_temp = np.clip(
                        best_pos + np.random.normal(0, 1, dim) * np.random.uniform(self._lo, self._hi),
                        self._lo, self._hi)
                    fit_temp = float(self.problem.evaluate(pos_temp)); evals += 1
                    if self._is_better(fit_temp, float(pop[order[0], -1])):
                        N += 1
                        circle.append(pos_temp)
                        break
                    N += 1
                N = min(N, n)
                idx_list = np.random.choice(n, N, replace=False)
                for j in idx_list:
                    D_h = np.abs(np.dot(B, best_pos) - pop[j, :-1])
                    circle.append(best_pos - E * D_h)
                pos = np.mean(circle, axis=0)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos); evals += n
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, evals, {}
