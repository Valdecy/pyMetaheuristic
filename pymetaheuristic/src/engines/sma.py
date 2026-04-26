"""pyMetaheuristic src — Slime Mould Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SMAEngine(PortedPopulationEngine):
    """Slime Mould Algorithm — oscillation weights model foraging behaviour of slime mould."""
    algorithm_id   = "sma"
    algorithm_name = "Slime Mould Algorithm"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1016/j.future.2020.03.055"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, p_t=0.03)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"weights": np.ones((pop.shape[0], self.problem.dimension))}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        p_t    = float(self._params.get("p_t", 0.03))

        order  = self._order(pop[:, -1])
        best_f = float(pop[order[0], -1])
        worst_f = float(pop[order[-1], -1])
        best_pos = pop[order[0], :-1].copy()

        ss     = abs(best_f - worst_f) + 1e-30

        # Update oscillation weights  (Eq. 2.5)
        weights = np.empty((n, dim))
        for idx in range(n):
            fit_i = float(pop[idx, -1])
            log_val = np.log10((best_f - fit_i) / ss + 1)
            if self.problem.objective == "min":
                if idx <= n // 2:
                    weights[idx] = 1.0 + np.random.uniform(0, 1, dim) * log_val
                else:
                    weights[idx] = 1.0 - np.random.uniform(0, 1, dim) * log_val
            else:
                # maximisation: reverse ordering roles
                if idx <= n // 2:
                    weights[idx] = 1.0 - np.random.uniform(0, 1, dim) * log_val
                else:
                    weights[idx] = 1.0 + np.random.uniform(0, 1, dim) * log_val

        a  = np.arctanh(max(1e-12, 1.0 - t / T))   # Eq. 2.4
        b  = 1.0 - t / T

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() < p_t:               # random dispersion  (Eq. 2.7)
                new_pos[i] = np.random.uniform(self._lo, self._hi)
            else:
                p  = np.tanh(abs(float(pop[i, -1]) - best_f))  # Eq. 2.2
                vb = np.random.uniform(-a, a, dim)              # Eq. 2.3
                vc = np.random.uniform(-b, b, dim)
                ids = list(range(n)); ids.remove(i)
                ia, ib = np.random.choice(ids, 2, replace=False)
                pos1   = best_pos + vb * (weights[i] * pop[ia, :-1] - pop[ib, :-1])
                pos2   = vc * pop[i, :-1]
                cond   = np.random.random(dim) < p
                new_pos[i] = np.where(cond, pos1, pos2)
            new_pos[i] = np.clip(new_pos[i], self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"weights": weights}
