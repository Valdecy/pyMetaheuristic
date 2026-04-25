"""pyMetaheuristic src — Chaos Game Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CGOEngine(PortedPopulationEngine):
    """Chaos Game Optimization — four seed generation strategies from fractal IFS."""
    algorithm_id   = "cgo"
    algorithm_name = "Chaos Game Optimization"
    family         = "math"
    _REFERENCE     = {"doi": "10.1007/s10462-020-09867-w"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        new_pos  = np.empty_like(pop[:, :-1])
        seeds_all= []
        for i in range(n):
            s1, s2, s3 = np.random.choice([k for k in range(n) if k != i], 3, replace=False)
            MG = (pop[s1,:-1] + pop[s2,:-1] + pop[s3,:-1]) / 3.0

            a1 = np.random.random()
            a2 = 2.0 * np.random.random()
            a3 = 1.0 + np.random.random() * np.random.random()
            beta = np.random.randint(0, 2, 3)
            gama = np.random.randint(0, 2, 3)
            k    = np.random.randint(0, dim)
            k_idx= np.random.choice(dim, k, replace=False)

            seed1 = pop[i,:-1] + a1*(beta[0]*best_pos - gama[0]*MG)
            seed2 = best_pos   + a2*(beta[1]*pop[i,:-1] - gama[1]*MG)
            seed3 = MG         + a3*(beta[2]*pop[i,:-1] - gama[2]*best_pos)
            seed4 = pop[i,:-1].copy()
            if len(k_idx): seed4[k_idx] += np.random.random(len(k_idx))

            candidates = np.vstack([np.clip(s, self._lo, self._hi)
                                    for s in [seed1,seed2,seed3,seed4]])
            seeds_all.append(candidates)

        # Evaluate all 4n seeds, keep best per individual
        all_seeds = np.vstack(seeds_all)
        all_fits  = self._evaluate_population(all_seeds)
        for i in range(n):
            block_fits = all_fits[i*4:(i+1)*4]
            bi = self._best_index(block_fits)
            new_pos[i] = all_seeds[i*4+bi]
        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n + 4*n, {}
