"""pyMetaheuristic src — Mantis Shrimp Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class MSHOAEngine(PortedPopulationEngine):
    """Mantis Shrimp Optimization Algorithm — PTI foraging, attack and defense moves."""
    algorithm_id = "mshoa"
    algorithm_name = "Mantis Shrimp Optimization Algorithm"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, k_value=0.3, use_reflection=True)

    def _initialize_payload(self, pop):
        return {"pti": np.round(1 + 2 * np.random.rand(pop.shape[0])).astype(int)}

    def _reflect(self, x):
        y = np.asarray(x, dtype=float).copy()
        for _ in range(10):
            below, above = y < self._lo, y > self._hi
            if not (np.any(below) or np.any(above)): break
            y = np.where(below, self._lo + (self._lo - y), y)
            y = np.where(above, self._hi - (y - self._hi), y)
        return np.clip(y, self._lo, self._hi)

    def _ptype(self, angles):
        pi8 = np.pi / 8.0
        out = np.full(angles.shape, 2, dtype=int)
        out[(3*pi8 <= angles) & (angles <= 5*pi8)] = 1
        out[((pi8 < angles) & (angles < 3*pi8)) | ((5*pi8 < angles) & (angles < 7*pi8))] = 3
        return out

    def _adiff(self, angles, pt):
        refs = {1: [np.pi/2], 2: [0, np.pi], 3: [np.pi/4, 3*np.pi/4]}
        out = np.zeros_like(angles)
        for k, vals in refs.items():
            m = pt == k
            if np.any(m):
                out[m] = np.min([np.abs(angles[m] - v) for v in vals], axis=0)
        return out

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        pti = np.asarray(state.payload.get("pti", np.ones(n, dtype=int)), dtype=int)
        if pti.shape[0] != n: pti = np.ones(n, dtype=int)
        old = pop[:, :-1].copy()
        best = pop[self._best_index(pop[:, -1]), :-1]
        new = old.copy()
        for i in range(n):
            if pti[i] == 1:
                r = np.random.randint(n - 1); r = r + (r >= i)
                D = np.random.uniform(-1.0, 1.0)
                new[i] = best - (old[i] - best) + D * (old[r] - old[i])
            elif pti[i] == 2:
                theta = np.random.uniform(np.pi, 2*np.pi)
                new[i] = best * np.cos(theta)
            else:
                sign = 1.0 if np.random.rand() < 0.5 else -1.0
                new[i] = best + sign * float(self._params.get("k_value", 0.3)) * best
            new[i] = self._reflect(new[i]) if bool(self._params.get("use_reflection", True)) else np.clip(new[i], self._lo, self._hi)
        fit = self._evaluate_population(new)
        pop = np.hstack((new, fit[:, None]))
        dot = np.sum(old * new, axis=1)
        denom = np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
        cosang = np.clip(dot / np.where(denom != 0, denom, 1.0), -1.0, 1.0)
        lpa = np.arccos(cosang); lpa[denom == 0] = 0.0
        rpa = np.random.rand(n) * np.pi
        lpt, rpt = self._ptype(lpa), self._ptype(rpa)
        pti = np.where(self._adiff(lpa, lpt) < self._adiff(rpa, rpt), lpt, rpt)
        return pop, n, {"pti": pti}
