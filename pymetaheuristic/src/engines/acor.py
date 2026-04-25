"""pyMetaheuristic src — Ant Colony Optimization for Continuous Domains Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ACOREngine(PortedPopulationEngine):
    """ACOR — Gaussian kernel mixture sampled from ranked archive."""
    algorithm_id = "acor"; algorithm_name = "Ant Colony Optimization (Continuous)"; family = "swarm"
    _REFERENCE   = {"doi": "10.1007/s10732-008-9062-4"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, sample_count=25, intent_factor=0.5, zeta=1.0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        sc  = max(1, int(self._params.get("sample_count", 25)))
        q   = float(self._params.get("intent_factor", 0.5))
        zeta= float(self._params.get("zeta", 1.0))
        order = self._order(pop[:, -1]); pop = pop[order]
        ranks = np.arange(1, n + 1, dtype=float)
        qn = q * n
        w  = (1.0 / (np.sqrt(2*np.pi)*qn)) * np.exp(-0.5*((ranks-1)/qn)**2)
        p  = w / w.sum()
        matrix_pos = pop[:, :-1]
        sigma = np.array([zeta * np.sum(np.abs(matrix_pos - pop[i, :-1]), axis=0) / (n-1) for i in range(n)])
        new_pos = np.empty((sc, dim))
        for s in range(sc):
            for j in range(dim):
                r = int(np.searchsorted(np.cumsum(p), np.random.random()))
                new_pos[s, j] = pop[r, j] + np.random.normal() * sigma[r, j]
        new_pos = np.clip(new_pos, self._lo, self._hi)
        new_fit = self._evaluate_population(new_pos)
        combined = np.vstack([pop, np.hstack([new_pos, new_fit[:, None]])])
        ord2 = self._order(combined[:, -1])
        return combined[ord2[:n]], sc, {}
