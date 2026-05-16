"""pyMetaheuristic src — Satin Bowerbird Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SBOEngine(PortedPopulationEngine):
    """Satin Bowerbird Optimizer — roulette-guided step with Gaussian mutation."""
    algorithm_id   = "sbo"
    algorithm_name = "Satin Bowerbird Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2017.01.006"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, alpha=0.94, p_m=0.05, psw=0.02)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        alpha  = float(self._params.get("alpha", 0.94))
        p_m    = float(self._params.get("p_m",  0.05))
        psw    = float(self._params.get("psw",  0.02))

        sigma  = psw * self._span

        # Fitness-proportionate roulette. Raw objective values can be
        # negative (e.g. Easom) or all near-identical, so reciprocal/raw-fitness
        # probabilities are numerically invalid. Convert fitness to a bounded
        # non-negative quality score where larger means better.
        fit = pop[:, -1].copy()
        quality = np.asarray(self._quality(fit), dtype=float)
        quality = np.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
        quality = np.maximum(quality, 0.0) + 1.0e-12
        prob = quality / np.sum(quality)

        order    = self._order(fit)
        best_pos = pop[order[0], :-1].copy()

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            rdx  = np.random.choice(n, p=prob)
            lam  = alpha * np.random.random()
            pos  = pop[i, :-1] + lam * ((pop[rdx, :-1] + best_pos) / 2.0 - pop[i, :-1])
            # Gaussian mutation
            mut  = pop[i, :-1] + np.random.normal(0, 1, dim) * sigma
            mask = np.random.random(dim) < p_m
            pos  = np.where(mask, mut, pos)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask2   = self._better_mask(new_fit, pop[:, -1])
        pop[mask2] = new_pop[mask2]
        return pop, n, {}
