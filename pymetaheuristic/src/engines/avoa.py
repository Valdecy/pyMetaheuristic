"""pyMetaheuristic src — African Vultures Optimization Algorithm Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_avoa(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return u / v ** (1 / beta)

class AVOAEngine(PortedPopulationEngine):
    """African Vultures Optimization Algorithm — three-phase satiation-driven foraging."""
    algorithm_id   = "avoa"
    algorithm_name = "African Vultures Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.cie.2021.107408"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        p1     = float(self._params.get("p1", 0.6))
        p2     = float(self._params.get("p2", 0.4))
        p3     = float(self._params.get("p3", 0.6))
        alpha  = float(self._params.get("alpha", 0.8))
        gama   = float(self._params.get("gama", 2.5))
        EPS    = 1e-10

        ratio  = t / T
        a_val  = np.random.uniform(-2, 2) * (
                    np.sin((np.pi / 2) * ratio) ** gama
                  + np.cos((np.pi / 2) * ratio) - 1.0)
        ppp    = (2.0 * np.random.random() + 1.0) * (1.0 - ratio) + a_val

        order  = self._order(pop[:, -1])
        best1  = pop[order[0], :-1].copy()
        best2  = pop[order[1], :-1].copy() if n > 1 else best1.copy()

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            F        = ppp * (2.0 * np.random.random() - 1.0)
            rand_pos = best1 if np.random.random() < alpha else best2

            if abs(F) >= 1.0:                          # Exploration
                if np.random.random() < p1:
                    pos = rand_pos - (abs(2.0 * np.random.random() * rand_pos - pop[i, :-1])) * F
                else:
                    pos = rand_pos - F + np.random.random() * (self._span * np.random.random() + self._lo)
            else:                                       # Exploitation
                if abs(F) < 0.5:                        # Phase 1
                    b1, b2 = best1, best2
                    if np.random.random() < p2:
                        A = b1 - (b1 * pop[i, :-1]) / (b1 - pop[i, :-1] ** 2 + EPS) * F
                        B = b2 - (b2 * pop[i, :-1]) / (b2 - pop[i, :-1] ** 2 + EPS) * F
                        pos = (A + B) / 2.0
                    else:
                        pos = rand_pos - abs(rand_pos - pop[i, :-1]) * F * _levy_avoa(dim)
                else:                                   # Phase 2
                    if np.random.random() < p3:
                        pos = (abs(2.0 * np.random.random() * rand_pos - pop[i, :-1])) * \
                              (F + np.random.random()) - (rand_pos - pop[i, :-1])
                    else:
                        s1  = rand_pos * (np.random.random() * pop[i, :-1] / (2.0 * np.pi)) * np.cos(pop[i, :-1])
                        s2  = rand_pos * (np.random.random() * pop[i, :-1] / (2.0 * np.pi)) * np.sin(pop[i, :-1])
                        pos = rand_pos - (s1 + s2)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        # AVOA replaces all (no greedy in original)
        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])
        return pop, n, {}
