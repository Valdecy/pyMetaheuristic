"""pyMetaheuristic src — Teamwork Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class TOAEngine(PortedPopulationEngine):
    """Teamwork Optimization Algorithm — three-stage supervisor/sharing/individual update."""
    algorithm_id   = "toa"
    algorithm_name = "Teamwork Optimization Algorithm"
    family         = "human"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        order   = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals   = 0

        for i in range(n):
            # Stage 1: Supervisor guidance
            r1  = np.random.randint(1, 3)
            pos = pop[i, :-1] + np.random.random() * (best_pos - r1 * pop[i, :-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(pop[i, -1])):
                pop[i, :-1] = pos; pop[i, -1] = fit

            # Stage 2: Information sharing with better agents
            better_idxs = [j for j in range(n) if j != i
                           and self._is_better(float(pop[j, -1]), float(pop[i, -1]))]
            if better_idxs:
                sf_pos = pop[better_idxs, :-1].mean(axis=0)
                sf_fit = float(np.mean(pop[better_idxs, -1]))
            else:
                sf_pos = best_pos.copy()
                sf_fit = float(pop[order[0], -1])

            sign = np.sign(float(pop[i, -1]) - sf_fit) if self.problem.objective == "min" \
                   else np.sign(sf_fit - float(pop[i, -1]))
            r2   = np.random.randint(1, 3)
            pos2 = pop[i, :-1] + np.random.random() * (sf_pos - r2 * pop[i, :-1]) * sign
            pos2 = np.clip(pos2, self._lo, self._hi)
            fit2 = float(self.problem.evaluate(pos2)); evals += 1
            if self._is_better(fit2, float(pop[i, -1])):
                pop[i, :-1] = pos2; pop[i, -1] = fit2

            # Stage 3: Individual activity
            pos3 = pop[i, :-1] + (-0.01 + np.random.random() * 0.02) * pop[i, :-1]
            pos3 = np.clip(pos3, self._lo, self._hi)
            fit3 = float(self.problem.evaluate(pos3)); evals += 1
            if self._is_better(fit3, float(pop[i, -1])):
                pop[i, :-1] = pos3; pop[i, -1] = fit3

        return pop, evals, {}
