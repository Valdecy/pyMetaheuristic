"""pyMetaheuristic src — Aquila Optimizer Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_ao(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return u / v ** (1 / beta)

class AOEngine(PortedPopulationEngine):
    """Aquila Optimizer — four hunting strategies inspired by the Aquila hunting behaviour."""
    algorithm_id   = "ao"
    algorithm_name = "Aquila Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.cie.2021.107250"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)

        alpha = delta = 0.1
        g1  = 2.0 * np.random.random() - 1.0
        g2  = 2.0 * (1.0 - t / T)
        # Guard single-step runs: the original denominator (1 - T)^2 is zero when T == 1.
        # In that degenerate schedule, the quality factor should stay finite.
        qf_denom = (1.0 - T) ** 2
        QF  = 1.0 if qf_denom == 0.0 else t ** ((2.0 * np.random.random() - 1.0) / qf_denom)

        # Spiral shape helpers  (Eqs. 9–10)
        dims  = np.arange(1, dim + 1, dtype=float)
        miu, r0, w = 0.00565, 10.0, 0.005
        phi0  = 3.0 * np.pi / 2.0
        phi   = -w * dims + phi0
        r_    = r0 + miu * dims
        x_sp  = r_ * np.sin(phi)
        y_sp  = r_ * np.cos(phi)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            levy  = _levy_ao(dim)
            if t <= (2.0 / 3.0) * T:               # Exploration (Eqs. 3–5)
                if np.random.random() < 0.5:        # Eq. 3
                    pos = best_pos * (1.0 - t / T) + np.random.random() * (x_mean - best_pos)
                else:                               # Eq. 5
                    j   = np.random.choice([k for k in range(n) if k != i])
                    pos = best_pos * levy + pop[j, :-1] + np.random.random() * (y_sp - x_sp)
            else:                                    # Exploitation (Eqs. 13–14)
                if np.random.random() < 0.5:        # Eq. 13
                    pos = alpha * (best_pos - x_mean) - np.random.random() * (
                        np.random.random() * self._span + self._lo) * delta
                else:                               # Eq. 14
                    pos = QF * best_pos - (g2 * pop[i, :-1] * np.random.random()) - g2 * levy + np.random.random() * g1
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
