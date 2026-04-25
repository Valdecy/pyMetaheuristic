"""pyMetaheuristic src — RUNge Kutta Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RUNEngine(PortedPopulationEngine):
    """RUNge Kutta Optimizer — search mechanism based on the Runge-Kutta numerical method."""
    algorithm_id   = "run"
    algorithm_name = "RUNge Kutta Optimizer"
    family         = "math"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.115079"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    @staticmethod
    def _rk(xb: np.ndarray, xw: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """4th-order Runge-Kutta search mechanism (simplified)."""
        f  = lambda x: (xb - xw) / (np.where(np.abs(xb-xw) < 1e-30, 1e-30, xb-xw)) * (x - xb) + xb
        k1 = f(xb)
        k2 = f(xb + 0.5*dx)
        k3 = f(xb + 0.5*dx)
        k4 = f(xb + dx)
        return (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        evals   = 0

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)

        f  = 20.0 * np.exp(-12.0 * t / T)
        SF = 2.0 * (0.5 - np.random.random(n)) * f

        new_pop = pop.copy()
        for i in range(n):
            gam  = np.random.random() * (pop[i,:-1] - np.random.random(dim) * self._span) * np.exp(-4.0*t/T)
            stp  = np.random.random(dim) * ((best_pos - np.random.random()*x_mean) + gam)
            dx   = 2.0 * np.random.random(dim) * np.abs(stp)

            ids  = [k for k in range(n) if k != i]
            a, b, c = np.random.choice(ids, 3, replace=False)
            fit_abc  = [float(pop[a,-1]), float(pop[b,-1]), float(pop[c,-1])]
            best_local = [a,b,c][int(np.argmin(fit_abc)) if self.problem.objective=="min"
                                  else int(np.argmax(fit_abc))]

            if self._is_better(float(pop[i,-1]), float(pop[best_local,-1])):
                xb, xw = pop[i,:-1], pop[best_local,:-1]
            else:
                xb, xw = pop[best_local,:-1], pop[i,:-1]

            SM   = self._rk(xb, xw, dx)
            L    = np.random.randint(0, 2, dim)
            xc   = L*pop[i,:-1] + (1-L)*pop[a,:-1]
            xm   = L*best_pos   + (1-L)*pop[order[0],:-1]
            r    = np.random.choice([1,-1], dim)
            g    = 2.0 * np.random.random()
            mu   = 0.5 + np.random.random(dim)

            if np.random.random() < 0.5:
                pos = xc + r*SF[i]*g*xc + SF[i]*SM + mu*(xm - xc)
            else:
                pos = xm + r*SF[i]*g*xm + SF[i]*SM + mu*(pop[a,:-1] - pop[b,:-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit, float(pop[i,-1])):
                new_pop[i,:-1] = pos; new_pop[i,-1] = fit

            # Enhanced Solution Quality (ESQ)
            if np.random.random() < 0.5:
                w    = np.random.uniform(0,2,dim)*np.exp(-5*np.random.random()*t/T)
                r_   = np.floor(np.random.uniform(-1,2))
                u    = 2*np.random.random(dim)
                a2,b2,c2 = np.random.choice(ids, 3, replace=False)
                x_av= (pop[a2,:-1]+pop[b2,:-1]+pop[c2,:-1])/3
                beta_= np.random.random(dim)
                x1   = beta_*best_pos + (1-beta_)*x_av
                x2   = np.where(w<1,
                                x1 + r_*w*np.abs(np.random.normal(0,1,dim) + x1 - x_av),
                                x1 - x_av + r_*w*np.abs(np.random.normal(0,1,dim) + u*x1 - x_av))
                pos2 = np.clip(x2, self._lo, self._hi)
                fit2 = float(self.problem.evaluate(pos2)); evals += 1
                if self._is_better(fit2, float(new_pop[i,-1])):
                    new_pop[i,:-1] = pos2; new_pop[i,-1] = fit2

        return new_pop, evals, {}
