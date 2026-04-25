"""pyMetaheuristic src — Seahorse Optimizer Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_seaho(dim: int) -> np.ndarray:
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return 0.01 * u / v**(1/beta)

class SeaHOEngine(PortedPopulationEngine):
    """Seahorse Optimizer — motor behavior, predation and reproduction phases."""
    algorithm_id   = "seaho"
    algorithm_name = "Seahorse Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10489-022-03994-3"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, uu=0.05, vv=0.05, ll=0.05)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        uu      = float(self._params.get("uu", 0.05))
        vv      = float(self._params.get("vv", 0.05))
        ll      = float(self._params.get("ll", 0.05))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        levy     = _levy_seaho(dim)

        # Phase 1: Motor behavior
        motor_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            beta  = np.random.normal(0, 1, dim)
            theta = 2.0 * np.pi * np.random.random(dim)
            row   = uu * np.exp(theta * vv)
            xx    = row * np.cos(theta); yy = row * np.sin(theta); zz = row * theta
            if np.random.normal() > 0:              # Eq. 4
                pos = pop[i, :-1] + levy * ((best_pos - pop[i, :-1]) * xx * yy * zz + best_pos)
            else:                                   # Eq. 7
                pos = pop[i, :-1] + np.random.random(dim) * ll * beta * (best_pos - beta * best_pos)
            motor_pos[i] = np.clip(pos, self._lo, self._hi)

        # Phase 2: Predation
        alpha   = (1.0 - t / T) ** (2.0 * t / T)
        pred_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r1 = np.random.random(dim)
            if np.random.random() >= 0.1:           # Eq. 10
                pos = alpha * (best_pos - r1 * motor_pos[i]) + (1.0 - alpha) * best_pos
            else:                                   # Eq. 11
                pos = (1.0 - alpha) * (motor_pos[i] - r1 * best_pos) + alpha * motor_pos[i]
            pred_pos[i] = np.clip(pos, self._lo, self._hi)

        pred_fit = self._evaluate_population(pred_pos)
        pred_pop = np.hstack([pred_pos, pred_fit[:, None]])
        pred_order = self._order(pred_fit)

        # Phase 3: Reproduction — top half = dads, bottom = moms
        half    = n // 2
        dads    = pred_pop[pred_order[:half], :-1]
        moms    = pred_pop[pred_order[half:half*2], :-1]
        off_pos = np.empty((min(half, len(moms)), dim))
        for k in range(off_pos.shape[0]):
            r3 = np.random.random()
            off_pos[k] = np.clip(r3 * dads[k] + (1.0 - r3) * moms[k], self._lo, self._hi)
        off_fit = self._evaluate_population(off_pos)
        off_pop = np.hstack([off_pos, off_fit[:, None]])

        combined  = np.vstack([pred_pop, off_pop])
        ord_comb  = self._order(combined[:, -1])
        pop       = combined[ord_comb[:n]]
        return pop, n + half, {}
