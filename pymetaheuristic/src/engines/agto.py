"""pyMetaheuristic src — Artificial Gorilla Troops Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class AGTOEngine(PortedPopulationEngine):
    """Artificial Gorilla Troops Optimizer — silverback-led exploration & exploitation."""
    algorithm_id   = "agto"
    algorithm_name = "Artificial Gorilla Troops Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1002/int.22535"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, p1=0.03, p2=0.8, beta=3.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        p1     = float(self._params.get("p1", 0.03))
        p2     = float(self._params.get("p2", 0.8))
        beta   = float(self._params.get("beta", 3.0))

        a = (np.cos(2.0 * np.random.random()) + 1.0) * (1.0 - t / T)
        c = a * (2.0 * np.random.random() - 1.0)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # Exploration
        exp_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() < p1:
                pos = np.random.uniform(self._lo, self._hi)
            else:
                if np.random.random() >= 0.5:
                    z   = np.random.uniform(-a, a, dim)
                    j   = np.random.randint(n)
                    pos = (np.random.random() - a) * pop[j, :-1] + c * z * pop[i, :-1]
                else:
                    ids = [k for k in range(n) if k != i]
                    i1, i2 = np.random.choice(ids, 2, replace=False)
                    pos = pop[i, :-1] - c * (c * pop[i, :-1] - pop[i1, :-1]) + \
                          np.random.random() * (pop[i, :-1] - pop[i2, :-1])
            exp_pos[i] = np.clip(pos, self._lo, self._hi)

        exp_fit = self._evaluate_population(exp_pos); evals += n
        mask    = self._better_mask(exp_fit, pop[:, -1])
        pop[mask] = np.hstack([exp_pos, exp_fit[:, None]])[mask]

        # Exploitation
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)
        ex2_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            if a >= p2:
                g     = 2.0 ** c
                delta = abs(x_mean) ** g + 1e-30
                delta = delta ** (1.0 / g)
                pos   = c * delta * (pop[i, :-1] - best_pos) + pop[i, :-1]
            else:
                h   = np.random.normal(0, 1, dim) if np.random.random() >= 0.5 else np.random.normal(0, 1)
                r1  = np.random.random()
                pos = best_pos - (2.0 * r1 - 1.0) * (best_pos - pop[i, :-1]) * (beta * h)
            ex2_pos[i] = np.clip(pos, self._lo, self._hi)

        ex2_fit = self._evaluate_population(ex2_pos); evals += n
        mask    = self._better_mask(ex2_fit, pop[:, -1])
        pop[mask] = np.hstack([ex2_pos, ex2_fit[:, None]])[mask]
        return pop, evals, {}
