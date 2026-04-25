"""pyMetaheuristic src — Queuing Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class QSAEngine(PortedPopulationEngine):
    """Queuing Search Algorithm — three business-function analogies for queue management."""
    algorithm_id   = "qsa"
    algorithm_name = "Queuing Search Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s12652-020-02849-4"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _business1(self, pop):
        """Rearrange positions relative to best."""
        n, dim   = pop.shape[0], self.problem.dimension
        order    = self._order(pop[:, -1])
        new_pop  = pop.copy()
        best_pos = pop[order[0], :-1]
        A        = np.zeros((dim, dim)); np.fill_diagonal(A, np.random.uniform(0.1, 1.0, dim))
        for i in range(n):
            pos = best_pos + np.dot(A, (pop[i, :-1] - best_pos))
            new_pop[i, :-1] = np.clip(pos, self._lo, self._hi)
        fits = self._evaluate_population(new_pop[:, :-1])
        new_pop[:, -1] = fits
        mask = self._better_mask(fits, pop[:, -1])
        result = pop.copy(); result[mask] = new_pop[mask]
        return result, int(n)

    def _business2(self, pop):
        """Ranked displacement update."""
        n, dim  = pop.shape[0], self.problem.dimension
        order   = self._order(pop[:, -1])
        ranked  = pop[order]
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if i == 0:
                pos = ranked[0, :-1] + np.random.random() * (ranked[0, :-1] - ranked[1, :-1])
            else:
                pos = ranked[i, :-1] + np.random.random() * (ranked[i-1, :-1] - ranked[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)
        fits = self._evaluate_population(new_pos); evals = n
        new_pop = np.hstack([new_pos, fits[:, None]])
        mask = self._better_mask(fits, ranked[:, -1])
        ranked[mask] = new_pop[mask]
        return ranked[self._order(ranked[:, -1])], evals

    def _business3(self, pop, best_pos):
        """Local search around best."""
        n, dim   = pop.shape[0], self.problem.dimension
        new_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            pr  = np.random.random(dim)
            pos = np.where(pr > 0.5,
                           pop[i, :-1] + np.random.random() * (best_pos - pop[i, :-1]),
                           np.random.uniform(self._lo, self._hi))
            new_pos[i] = np.clip(pos, self._lo, self._hi)
        fits    = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, fits[:, None]])
        mask    = self._better_mask(fits, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n

    def _step_impl(self, state, pop: np.ndarray):
        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0
        pop, e1 = self._business1(pop); evals += e1
        pop, e2 = self._business2(pop); evals += e2
        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        pop, e3 = self._business3(pop, best_pos); evals += e3
        return pop, evals, {}
