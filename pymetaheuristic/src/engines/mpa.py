"""pyMetaheuristic src — Marine Predators Algorithm Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class MPAEngine(PortedPopulationEngine):
    """Marine Predators Algorithm — three-phase Lévy/Brownian motion model of oceanic predation."""
    algorithm_id   = "mpa"
    algorithm_name = "Marine Predators Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2020.113377"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, fads=0.2, p=0.5)

    @staticmethod
    def _levy(dim: int, beta: float = 1.5) -> np.ndarray:
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.abs(np.random.normal(0, 1, dim))
        return u / (v ** (1 / beta))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        FADS    = float(self._params.get("fads", 0.2))
        P       = float(self._params.get("p", 0.5))

        order   = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        CF  = (1.0 - t / T) ** (2.0 * t / T)
        RL  = 0.05 * self._levy(n * dim, beta=1.5).reshape(n, dim)
        RB  = np.random.standard_normal((n, dim))
        per1 = np.random.permutation(n)
        per2 = np.random.permutation(n)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            R = np.random.random(dim)
            if t < T / 3:                                 # Phase 1: high exploration
                step  = RB[i] * (best_pos - RB[i] * pop[i, :-1])
                pos   = pop[i, :-1] + P * R * step
            elif t < 2 * T / 3:                           # Phase 2: transition
                if i > n // 2:
                    step = RB[i] * (RB[i] * best_pos - pop[i, :-1])
                    pos  = best_pos + P * CF * step
                else:
                    step = RL[i] * (best_pos - RL[i] * pop[i, :-1])
                    pos  = pop[i, :-1] + P * R * step
            else:                                         # Phase 3: exploitation
                step  = RL[i] * (RL[i] * best_pos - pop[i, :-1])
                pos   = best_pos + P * CF * step
            pos = np.clip(pos, self._lo, self._hi)

            # Fish Aggregating Devices  (FADs effect)
            if np.random.random() < FADS:
                u   = (np.random.random(dim) < FADS).astype(float)
                pos = pos + CF * (self._lo + np.random.random(dim) * self._span) * u
            else:
                r_   = np.random.random()
                pos  = pos + (FADS * (1 - r_) + r_) * (pop[per1[i], :-1] - pop[per2[i], :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
