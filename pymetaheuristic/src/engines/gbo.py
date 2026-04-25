"""pyMetaheuristic src — Gradient-Based Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GBOEngine(PortedPopulationEngine):
    """Gradient-Based Optimizer — gradient search rule with local escaping operator."""
    algorithm_id   = "gbo"
    algorithm_name = "Gradient-Based Optimizer"
    family         = "math"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pr=0.5, beta_min=0.2, beta_max=1.2)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        T        = max(1, self.config.max_steps or 500)
        t        = state.step + 1
        pr       = float(self._params.get("pr", 0.5))
        bmin     = float(self._params.get("beta_min", 0.2))
        bmax     = float(self._params.get("beta_max", 1.2))
        EPS      = 5e-3

        beta  = bmin + (bmax - bmin) * (1.0 - (t / T) ** 3) ** 2
        alpha = abs(beta * np.sin(3.0 * np.pi / 2.0 + np.sin(beta * 3.0 * np.pi / 2.0)))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0],  :-1].copy()
        worst_pos= pop[order[-1], :-1].copy()
        denom    = np.where(np.abs(worst_pos - best_pos) < EPS, EPS, worst_pos - best_pos)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            p1 = 2.0 * np.random.random() * alpha - alpha
            p2 = 2.0 * np.random.random() * alpha - alpha
            ids  = [k for k in range(n) if k != i]
            r1,r2,r3,r4 = np.random.choice(ids, 4, replace=False)
            r0   = (pop[r1,:-1] + pop[r2,:-1] + pop[r3,:-1] + pop[r4,:-1]) / 4.0
            eps2 = EPS * np.random.random()
            delta= 2.0 * np.random.random() * np.abs(r0 - pop[i,:-1])
            step = (best_pos - pop[r1,:-1] + delta) / 2.0
            dx   = np.random.choice(n) * np.abs(step)

            x1  = pop[i,:-1] - np.random.normal()*p1*2*dx*pop[i,:-1]/denom \
                  + np.random.random()*p2*(best_pos - pop[i,:-1])
            z   = pop[i,:-1] - np.random.normal()*2*dx*pop[i,:-1]/denom
            yp  = np.random.random()*((z + pop[i,:-1])/2 + np.random.random()*dx)
            yq  = np.random.random()*((z + pop[i,:-1])/2 - np.random.random()*dx)
            x2  = best_pos - np.random.normal()*p1*2*dx*pop[i,:-1]/(np.where(np.abs(yp-yq)<EPS,EPS,yp-yq)) \
                  + np.random.random()*p2*(pop[r1,:-1] - pop[r2,:-1])
            x3  = pop[i,:-1] - p1*(x2 - x1)
            ra, rb = np.random.random(), np.random.random()
            pos = ra*(rb*x1 + (1-rb)*x2) + (1-ra)*x3

            # Local escaping operator
            if np.random.random() < pr:
                f1  = np.random.uniform(-1, 1)
                f2  = np.random.normal(0, 1)
                L1  = round(1 - np.random.random())
                u1  = L1*2*np.random.random() + (1-L1)
                u2  = L1*np.random.random()   + (1-L1)
                u3  = L1*np.random.random()   + (1-L1)
                L2  = round(1 - np.random.random())
                x_rand = np.random.uniform(self._lo, self._hi)
                x_p    = pop[np.random.randint(n), :-1]
                x_m    = L2*x_p + (1-L2)*x_rand
                if np.random.random() < 0.5:
                    pos = pos + f1*(u1*best_pos - u2*x_m) + f2*p1*(u3*(x2-x1) + u2*(pop[r1,:-1]-pop[r2,:-1]))/2
                else:
                    pos = best_pos + f1*(u1*best_pos - u2*x_m) + f2*p1*(u3*(x2-x1) + u2*(pop[r1,:-1]-pop[r2,:-1]))/2
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
