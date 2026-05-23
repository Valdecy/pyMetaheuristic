"""pyMetaheuristic src — Singer Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class SingerOAEngine(PortedPopulationEngine):
    """Singer Optimization Algorithm.

    Human-inspired optimizer based on the beginner/imitation/creation learning
    stages described for singers. The implementation keeps the paper's greedy
    update structure while using normalized exponential arguments for numerical
    stability on arbitrary user-defined bounds.
    """

    algorithm_id = "singer_oa"
    algorithm_name = "Singer Optimization Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.22266/ijies2025.0630.09",
        "title": "Singer Optimization Algorithm: An Effective Human-Based Approach for Solving Optimization Tasks",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _stable_exp_factor(self, delta: np.ndarray) -> np.ndarray:
        # The paper writes exp(X_best - X_i). Normalizing by the box span keeps
        # the qualitative distance-dependent adaptation but avoids overflow and
        # scale sensitivity when variables have large physical units.
        z = np.clip(delta / (self._span + 1.0e-12), -60.0, 60.0)
        return np.exp(z)

    def _greedy_apply(self, pop: np.ndarray, trial: np.ndarray) -> tuple[np.ndarray, int]:
        trial = np.clip(trial, self._lo, self._hi)
        fit = self._evaluate_population(trial)
        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1] = trial[mask]
        pop[mask, -1] = fit[mask]
        return pop, int(trial.shape[0])

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, state.step + 1)
        best = pop[self._best_index(pop[:, -1]), :-1].copy()
        evals = 0

        # Phase 1 — imitation/adaptive mimicry of the best singer.
        r = np.random.rand(n, dim)
        I = np.random.randint(1, 3, size=(n, dim))
        delta = best[None, :] - pop[:, :-1]
        factor = 1.0 - self._stable_exp_factor(delta)
        trial = pop[:, :-1] + r * factor * (best[None, :] - I * pop[:, :-1])
        pop, used = self._greedy_apply(pop, trial)
        evals += used

        # Phase 2 — creation/novel perturbation. The 1/t term tightens the
        # creative shift as the run proceeds, as in the source equation.
        best = pop[self._best_index(pop[:, -1]), :-1].copy()
        r = np.random.rand(n, dim)
        I = np.random.randint(1, 3, size=(n, dim))
        delta = best[None, :] - pop[:, :-1]
        factor = 1.0 - 2.0 * (1.0 - self._stable_exp_factor(delta))
        trial = pop[:, :-1] + (factor * (best[None, :] - I * pop[:, :-1])) / float(t)
        pop, used = self._greedy_apply(pop, trial)
        evals += used

        return pop, evals, {}
