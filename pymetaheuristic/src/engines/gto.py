"""pyMetaheuristic src — Giant Trevally Optimizer Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_gto(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return 0.01 * u / v ** (1 / beta)

class GTOEngine(PortedPopulationEngine):
    """Giant Trevally Optimizer — three-phase search: extensive, area selection, attacking."""
    algorithm_id   = "gto"
    algorithm_name = "Giant Trevally Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2022.3223388"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, A=0.4, H=2.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        A_      = float(self._params.get("A", 0.4))
        H_param = float(self._params.get("H", 2.0))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # Phase 1: Extensive Search  (Eq. 4)
        p1_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            levy = _levy_gto(dim)
            pos  = best_pos * np.random.random() + (self._span * np.random.random() + self._lo) * levy
            p1_pos[i] = np.clip(pos, self._lo, self._hi)
        p1_fit = self._evaluate_population(p1_pos); evals += n
        mask   = self._better_mask(p1_fit, pop[:, -1])
        pop[mask] = np.hstack([p1_pos, p1_fit[:, None]])[mask]

        # Phase 2: Choosing Area  (Eq. 7)
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)
        p2_pos   = np.empty_like(pop[:, :-1])
        for i in range(n):
            r3   = np.random.random()
            pos  = best_pos * A_ * r3 + x_mean - pop[i, :-1] * r3
            p2_pos[i] = np.clip(pos, self._lo, self._hi)
        p2_fit = self._evaluate_population(p2_pos); evals += n
        mask   = self._better_mask(p2_fit, pop[:, -1])
        pop[mask] = np.hstack([p2_pos, p2_fit[:, None]])[mask]

        # Phase 3: Attacking  (Eqs. 10–13, 15)
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        H_val    = np.random.random() * H_param * (1.0 - t / T)
        p3_pos   = np.empty_like(pop[:, :-1])
        for i in range(n):
            dist  = np.sum(np.abs(best_pos - pop[i, :-1]))
            theta2 = np.random.uniform(0, 360)
            theta1 = (1.33 / 1.00029) * np.sin(np.radians(theta2))
            VD     = np.sin(np.radians(theta1)) * dist
            fit_i  = float(pop[i, -1])
            pos    = pop[i, :-1] * np.sin(np.radians(theta2)) * fit_i + VD + H_val
            p3_pos[i] = np.clip(pos, self._lo, self._hi)
        p3_fit = self._evaluate_population(p3_pos); evals += n
        mask   = self._better_mask(p3_fit, pop[:, -1])
        pop[mask] = np.hstack([p3_pos, p3_fit[:, None]])[mask]

        return pop, evals, {}
