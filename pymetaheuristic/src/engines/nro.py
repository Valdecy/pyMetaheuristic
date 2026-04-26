"""pyMetaheuristic src — Nuclear Reaction Optimization Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class NROEngine(PortedPopulationEngine):
    """Nuclear Reaction Optimization — fission (NFi) and fusion (NFu) nuclear analogy."""
    algorithm_id   = "nro"
    algorithm_name = "Nuclear Reaction Optimization"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1109/ACCESS.2019.2918406"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    @staticmethod
    def _levy_nro(dim: int) -> np.ndarray:
        beta = 1.5
        sv   = 1.0
        su   = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
                (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        return np.random.normal(0, su, dim) / np.sqrt(np.abs(np.random.normal(0, sv, dim)))**(1/beta)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        levy    = self._levy_nro(dim)
        Pb      = np.random.random()
        Pfi     = np.random.random()
        evals   = 0

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        # NFi phase — fission products
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            others = [k for k in range(n) if k != i]
            i1 = np.random.choice(others)
            Nei = (pop[i, :-1] + pop[i1, :-1]) / 2.0
            sigma_base = (np.log(t + 1) / (t + 1)) * np.abs(pop[i, :-1] - best_pos) + 1e-12

            if np.random.random() <= Pfi:
                if np.random.random() <= Pb:
                    gauss = np.random.normal(best_pos, sigma_base)
                    Xi = gauss + np.random.random() * best_pos - float(round(np.random.random() + 1)) * Nei
                else:
                    i2 = np.random.choice(others)
                    sigma2 = (np.log(t + 1) / (t + 1)) * np.abs(pop[i2, :-1] - best_pos) + 1e-12
                    gauss  = np.random.normal(pop[i, :-1], sigma2)
                    Xi = gauss + np.random.random() * best_pos - float(round(np.random.random() + 2)) * Nei
            else:
                i3 = np.random.choice(others)
                sigma2 = (np.log(t + 1) / (t + 1)) * np.abs(pop[i3, :-1] - best_pos) + 1e-12
                Xi = np.random.normal(pop[i, :-1], sigma2)
            new_pos[i] = np.clip(Xi, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos); evals += n
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = np.hstack([new_pos, new_fit[:, None]])[mask]

        # NFu phase — ionisation + fusion
        order2   = self._order(pop[:, -1])
        ranked   = np.empty(n, dtype=int)
        for rank, idx in enumerate(order2): ranked[idx] = rank

        fus_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            X_ion = pop[i, :-1].copy()
            if (ranked[i] / n) < np.random.random():
                i1, i2 = np.random.choice([k for k in range(n) if k != i], 2, replace=False)
                X_ion  = pop[i, :-1] + levy * np.abs(pop[i, :-1] - best_pos) + \
                          np.random.random() * (pop[i1, :-1] - pop[i2, :-1])
            fus_pos[i] = np.clip(X_ion, self._lo, self._hi)

        fus_fit = self._evaluate_population(fus_pos); evals += n
        mask2   = self._better_mask(fus_fit, pop[:, -1])
        pop[mask2] = np.hstack([fus_pos, fus_fit[:, None]])[mask2]
        return pop, evals, {}
