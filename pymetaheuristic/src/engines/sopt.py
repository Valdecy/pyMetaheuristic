"""pyMetaheuristic src — Simple Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class SOPTEngine(PortedPopulationEngine):
    """Simple Optimization algorithm (SOPT).

    Population-based optimizer with two sequential operators: exploitation near
    the current best solution and exploration with a larger standard-deviation
    radius computed from the current population columns.
    """

    algorithm_id = "sopt"
    algorithm_name = "Simple Optimization Algorithm"
    family = "distribution"
    _REFERENCE = {
        "doi": "https://scispace.com/pdf/an-efficient-metaheuristic-algorithm-for-engineering-2vvsafbir9.pdf",
        "title": "An Efficient Metaheuristic Algorithm for Engineering Optimization: SOPT",
        "year": 2012,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, lambda1=1.0, offspring_size=None)

    def _generate_around_best(self, pop: np.ndarray, scale: float, count: int) -> np.ndarray:
        best = pop[self._best_index(pop[:, -1]), :-1]
        sigma = np.std(pop[:, :-1], axis=0, ddof=0)
        sigma = np.where(sigma <= 1.0e-12, 0.05 * self._span, sigma)
        trial = best[None, :] + float(scale) * np.random.normal(0.0, sigma, size=(count, self.problem.dimension))
        return np.clip(trial, self._lo, self._hi)

    def _select_best(self, pop: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, int]:
        candidates = np.clip(candidates, self._lo, self._hi)
        fit = self._evaluate_population(candidates)
        trial_pop = np.hstack((candidates, fit[:, None]))
        combined = np.vstack((pop, trial_pop))
        order = self._order(combined[:, -1])
        return combined[order[: pop.shape[0]]].copy(), int(candidates.shape[0])

    def _step_impl(self, state, pop: np.ndarray):
        n = pop.shape[0]
        count = self._params.get("offspring_size", None)
        count = n if count is None else max(1, int(count))
        lambda1 = float(self._params.get("lambda1", 1.0))
        lambda2 = 0.5 * lambda1
        evals = 0

        # The SOPT procedure applies exploitation first, then exploration.
        pop, used = self._select_best(pop, self._generate_around_best(pop, lambda2, count))
        evals += used
        pop, used = self._select_best(pop, self._generate_around_best(pop, lambda1, count))
        evals += used
        return pop, evals, {}
