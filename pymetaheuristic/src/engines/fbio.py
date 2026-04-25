"""pyMetaheuristic src — Forensic-Based Investigation Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FBIOEngine(PortedPopulationEngine):
    """Forensic-Based Investigation Optimization — investigation and pursuit team phases."""
    algorithm_id   = "fbio"
    algorithm_name = "Forensic-Based Investigation Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2020.106339"}
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

        # Team A – Step A1: Gaussian perturbation around two neighbours' mean
        for i in range(n):
            j_c  = np.random.randint(dim)
            nb1, nb2 = np.random.choice([k for k in range(n) if k != i], 2, replace=False)
            pos  = pop[i, :-1].copy()
            pos[j_c] += np.random.normal() * (pop[i, j_c] - (pop[nb1, j_c] + pop[nb2, j_c]) / 2.0)
            pos  = np.clip(pos, self._lo, self._hi)
            fit  = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(pop[i, -1])):
                pop[i, :-1] = pos; pop[i, -1] = fit

        # Fitness-proportional probability
        fit_arr = pop[:, -1]
        if self.problem.objective == "min":
            inv  = 1.0 / (np.abs(fit_arr) + 1e-30)
            prob = inv / inv.sum()
        else:
            prob = np.abs(fit_arr) / (np.abs(fit_arr).sum() + 1e-30)

        # Team A – Step A2: best + two random or random exploration
        a2_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() > prob[i]:
                r1, r2, r3 = np.random.choice([k for k in range(n) if k != i], 3, replace=False)
                temp = best_pos + pop[r1, :-1] + np.random.random() * (pop[r2, :-1] - pop[r3, :-1])
                cond = np.random.random(dim) < 0.5
                pos  = np.where(cond, temp, pop[i, :-1])
            else:
                pos = np.random.uniform(self._lo, self._hi)
            a2_pos[i] = np.clip(pos, self._lo, self._hi)
        a2_fit = self._evaluate_population(a2_pos); evals += n
        mask   = self._better_mask(a2_fit, pop[:, -1])
        pop[mask] = np.hstack([a2_pos, a2_fit[:, None]])[mask]

        # Team B – Step B1: convex combination toward best
        b1_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r  = np.random.random(dim)
            pos = r * pop[i, :-1] + np.random.random(dim) * (best_pos - pop[i, :-1])
            b1_pos[i] = np.clip(pos, self._lo, self._hi)
        b1_fit = self._evaluate_population(b1_pos); evals += n
        mask   = self._better_mask(b1_fit, pop[:, -1])
        pop[mask] = np.hstack([b1_pos, b1_fit[:, None]])[mask]

        return pop, evals, {}
