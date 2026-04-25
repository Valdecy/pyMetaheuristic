"""pyMetaheuristic src — Bald Eagle Search Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BESEngine(PortedPopulationEngine):
    """Bald Eagle Search — three-stage soar/spiral/swoop foraging strategy."""
    algorithm_id   = "bes"
    algorithm_name = "Bald Eagle Search"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-019-09732-5"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, a_factor=10, R_factor=1.5, alpha=2.0, c1=2.0, c2=2.0)

    def _xy_spiral(self, n: int):
        a = float(self._params.get("a_factor", 10))
        R = float(self._params.get("R_factor", 1.5))
        theta = a * np.pi * np.random.uniform(0, 1, n)
        r     = theta + R * np.random.uniform(0, 1, n)
        xp    = r * np.sin(theta);  yp = r * np.cos(theta)
        x1p   = r * np.sin(theta);  y1p = r * np.cos(theta)
        # normalise to [-1,1]
        def _norm(v):
            s = np.abs(v).max() + 1e-30; return v / s
        return _norm(xp), _norm(yp), _norm(x1p), _norm(y1p)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        alpha  = float(self._params.get("alpha", 2.0))
        c1     = float(self._params.get("c1", 2.0))
        c2     = float(self._params.get("c2", 2.0))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0
        x_sp, y_sp, x1_sp, y1_sp = self._xy_spiral(n)

        # Stage 1: Select space
        x_mean = pop[:, :-1].mean(axis=0)
        s1_pos = np.clip(
            best_pos + alpha * np.random.random((n, dim)) * (x_mean - pop[:, :-1]),
            self._lo, self._hi)
        s1_fit = self._evaluate_population(s1_pos); evals += n
        mask   = self._better_mask(s1_fit, pop[:, -1])
        pop[mask] = np.hstack([s1_pos, s1_fit[:, None]])[mask]

        # Stage 2: Search in space
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)
        s2_pos   = np.empty_like(pop[:, :-1])
        for i in range(n):
            j   = np.random.choice([k for k in range(n) if k != i])
            pos = pop[i, :-1] + y_sp[i] * (pop[i, :-1] - pop[j, :-1]) + x_sp[i] * (pop[i, :-1] - x_mean)
            s2_pos[i] = np.clip(pos, self._lo, self._hi)
        s2_fit = self._evaluate_population(s2_pos); evals += n
        mask   = self._better_mask(s2_fit, pop[:, -1])
        pop[mask] = np.hstack([s2_pos, s2_fit[:, None]])[mask]

        # Stage 3: Swoop
        best_pos = pop[self._best_index(pop[:, -1]), :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)
        s3_pos   = np.clip(
            np.random.random((n, dim)) * best_pos
            + x1_sp[:, None] * (pop[:, :-1] - c1 * x_mean)
            + y1_sp[:, None] * (pop[:, :-1] - c2 * best_pos),
            self._lo, self._hi)
        s3_fit = self._evaluate_population(s3_pos); evals += n
        mask   = self._better_mask(s3_fit, pop[:, -1])
        pop[mask] = np.hstack([s3_pos, s3_fit[:, None]])[mask]

        return pop, evals, {}
