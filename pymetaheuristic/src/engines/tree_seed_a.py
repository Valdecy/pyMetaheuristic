"""pyMetaheuristic src — Tree-Seed Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class TreeSeedAEngine(PortedPopulationEngine):

    algorithm_id = "tree_seed_a"
    algorithm_name = "Tree-Seed Algorithm"
    family = "nature"
    _REFERENCE = {
        "doi": "10.1016/j.eswa.2015.04.055",
        "title": "TSA: Tree-seed algorithm for continuous optimization",
        "authors": "Mustafa Servet Kiran",
        "year": 2015,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=10,
        st=0.1,
        min_seed_fraction=0.10,
        max_seed_fraction=0.25,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("tree_seed_a requires population_size >= 2.")
        st = float(self._params.get("st", 0.1))
        if not 0.0 <= st <= 1.0:
            raise ValueError("tree_seed_a st must be in [0, 1].")
        min_frac = float(self._params.get("min_seed_fraction", 0.10))
        max_frac = float(self._params.get("max_seed_fraction", 0.25))
        if min_frac <= 0.0 or max_frac <= 0.0:
            raise ValueError("tree_seed_a seed fractions must be positive.")
        if min_frac > max_frac:
            raise ValueError("tree_seed_a min_seed_fraction cannot exceed max_seed_fraction.")

    def _seed_bounds(self, n: int) -> tuple[int, int]:
        min_frac = float(self._params.get("min_seed_fraction", 0.10))
        max_frac = float(self._params.get("max_seed_fraction", 0.25))
        lo = max(1, int(np.ceil(min_frac * n)))
        hi = max(lo, int(np.ceil(max_frac * n)))
        return lo, hi

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        positions = pop[:, :-1].copy()
        fitness = pop[:, -1].copy()
        source_positions = positions.copy()
        best = source_positions[self._best_index(fitness)].copy()
        st = float(self._params.get("st", 0.1))
        min_seeds, max_seeds = self._seed_bounds(n)

        evals = 0
        for i in range(n):
            seed_count = int(np.random.randint(min_seeds, max_seeds + 1))
            r_idx = np.array([self._rand_indices(n, i, 1)[0] for _ in range(seed_count)], dtype=int)
            random_trees = source_positions[r_idx]
            tree = source_positions[i]
            alpha = np.random.uniform(-1.0, 1.0, size=(seed_count, dim))

            toward_best = tree + alpha * (best - random_trees)       # Eq. (3)
            away_random = tree + alpha * (tree - random_trees)       # Eq. (4)
            use_best_rule = np.random.rand(seed_count, dim) < st
            seeds = np.where(use_best_rule, toward_best, away_random)
            seeds = np.clip(seeds, self._lo, self._hi)
            seed_fit = self._evaluate_population(seeds)
            evals += seed_count

            best_seed_idx = self._best_index(seed_fit)
            best_seed_fit = float(seed_fit[best_seed_idx])
            if self._is_better(best_seed_fit, fitness[i]):
                positions[i] = seeds[best_seed_idx]
                fitness[i] = best_seed_fit

        return np.hstack((positions, fitness[:, None])), evals, {}
