"""pyMetaheuristic src — Human Evolutionary Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


def _levy1d(d, beta=1.5):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    return u / (np.abs(v) ** (1 / beta) + 1.0e-12)


class HEOAEngine(PortedPopulationEngine):
    algorithm_id   = "heoa"
    algorithm_name = "Human Evolutionary Optimization Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2023.122638"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, levy_scale=0.08, local_scale=0.05, elite_fraction=0.08)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        span = self._span
        t = state.step + 1
        T = max(1, self.config.max_steps or int(self._params.get("max_iterations", 500)))
        progress = min(1.0, t / T)
        levy_scale = max(0.0, float(self._params.get("levy_scale", 0.08)))
        local_scale = max(0.0, float(self._params.get("local_scale", 0.05)))
        elite_fraction = min(max(float(self._params.get("elite_fraction", 0.08)), 0.0), 0.5)

        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        best = pop[0, :-1].copy()
        worst = pop[-1, :-1].copy()
        mean = np.mean(pop[:, :-1], axis=0)
        elite_n = max(1, int(round(elite_fraction * n)))

        # Four role groups used by the native method, with a stable bounded
        # interpretation: leaders exploit, explorers diversify, followers move
        # toward the best, and risk takers perform escape moves.
        LNn = max(1, round(n * 0.35))
        ENn = max(1, round(n * 0.35))
        FNn = max(1, round(n * 0.15))
        decay = 1.0 - progress
        oscillation = abs(np.cos(np.pi * progress / 2.0))
        new_pos = pop[:, :-1].copy()

        for j in range(n):
            x = pop[j, :-1]
            if j < elite_n:
                # Preserve elites; they may still be improved by local refinement.
                candidate = x + np.random.normal(0.0, local_scale * span * (0.1 + decay), d)
            elif j < LNn:
                # Learners: strong attraction to best with Levy perturbation.
                candidate = x + np.random.random(d) * (best - x)
                candidate += levy_scale * decay * span * _levy1d(d)
            elif j < LNn + ENn:
                # Explorers: move around the population centroid and away from worst.
                direction = mean - x + np.random.random(d) * (x - worst)
                candidate = x + oscillation * np.random.random(d) * direction
                candidate += np.random.normal(0.0, 0.15 * decay * span, d)
            elif j < LNn + ENn + FNn:
                # Followers: contract toward the current best.
                candidate = x + oscillation * np.random.random(d) * (best - x)
            else:
                # Risk takers: generate a bounded sample around the best.
                radius = (0.20 * decay + 0.02) * span
                candidate = best + np.random.normal(0.0, radius, d)

            new_pos[j] = np.clip(candidate, lo, hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = pop.copy()
        mask = self._better_mask(new_fit, pop[:, -1])
        new_pop[mask, :-1] = new_pos[mask]
        new_pop[mask, -1] = new_fit[mask]
        # Keep sorted order for stable role assignment in the next iteration.
        new_pop = new_pop[self._order(new_pop[:, -1])]
        return new_pop, n, {}
