"""pyMetaheuristic src — Weighting and Inertia Random Walk Optimizer (INFO) Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class INFOEngine(PortedPopulationEngine):
    """INFO — weighted mean rule combining local and global influence vectors."""
    algorithm_id   = "info"
    algorithm_name = "Weighting and Inertia Random Walk Optimizer"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2022.116516"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        EPS     = 1e-25

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        best_fit = float(pop[order[0], -1])
        worst_pos= pop[order[-1], :-1].copy()
        worst_fit= float(pop[order[-1], -1])

        alpha = 2.0 * np.exp(-4.0 * t / T)
        # pick "better" as a random member from 3rd–6th best
        bet_idx = min(n-1, np.random.randint(2, 6))
        better_pos = pop[order[bet_idx], :-1].copy()
        better_fit = float(pop[order[bet_idx], -1])

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            delta = 2.0*np.random.random()*alpha - alpha
            sigma = 2.0*np.random.random()*alpha - alpha
            a, b, c = np.random.choice([k for k in range(n) if k != i], 3, replace=False)
            eps   = EPS * np.random.random()

            fa, fb, fc = float(pop[a,-1]), float(pop[b,-1]), float(pop[c,-1])
            omg1  = max(abs(fa), abs(fb), abs(fc)) + 1e-30
            MM1   = np.array([fa-fb, fa-fc, fb-fc])
            w1 = np.cos(MM1[0]+np.pi)*np.exp(-abs(MM1[0]/omg1))
            w2 = np.cos(MM1[1]+np.pi)*np.exp(-abs(MM1[1]/omg1))
            w3 = np.cos(MM1[2]+np.pi)*np.exp(-abs(MM1[2]/omg1))
            Wt1 = w1+w2+w3
            WM1 = delta*(w1*(pop[a,:-1]-pop[b,:-1]) +
                         w2*(pop[a,:-1]-pop[c,:-1]) +
                         w3*(pop[b,:-1]-pop[c,:-1]))/(Wt1+1) + eps

            omg2  = max(abs(best_fit), abs(better_fit), abs(worst_fit)) + 1e-30
            MM2   = np.array([best_fit-better_fit, best_fit-worst_fit, better_fit-worst_fit])
            w4 = np.cos(MM2[0]+np.pi)*np.exp(-abs(MM2[0]/omg2))
            w5 = np.cos(MM2[1]+np.pi)*np.exp(-abs(MM2[1]/omg2))
            w6 = np.cos(MM2[2]+np.pi)*np.exp(-abs(MM2[2]/omg2))
            Wt2 = w4+w5+w6
            WM2 = delta*(w4*(best_pos-better_pos) +
                         w5*(best_pos-worst_pos) +
                         w6*(better_pos-worst_pos))/(Wt2+1) + eps

            r    = np.random.uniform(0.1, 0.5)
            MR   = r*WM1 + (1-r)*WM2
            fit_i  = float(pop[i, -1])

            if np.random.random() < 0.5:
                z1 = pop[i,:-1] + sigma*np.random.random()*MR + np.random.random()*(best_pos - pop[a,:-1])/(best_fit - fa + 1)
                z2 = best_pos   + sigma*np.random.random()*MR + np.random.random()*(pop[a,:-1] - pop[b,:-1])/(fa - fb + 1)
            else:
                z1 = pop[a,:-1] + sigma*np.random.random()*MR + np.random.random()*(pop[b,:-1] - pop[c,:-1])/(fb - fc + 1)
                z2 = better_pos + sigma*np.random.random()*MR + np.random.random()*(pop[a,:-1] - pop[b,:-1])/(fa - fb + 1)

            u = np.random.random(dim)
            pos = np.where(u < 0.5, z1, z2)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
