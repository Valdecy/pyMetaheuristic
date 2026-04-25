"""pyMetaheuristic src — Equilibrium Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EOEngine(PortedPopulationEngine):
    """Equilibrium Optimizer — concentration-equilibrium analogy for exploration/exploitation."""
    algorithm_id   = "eo"
    algorithm_name = "Equilibrium Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2019.105190"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, a1=2.0, a2=1.0, gp=0.5)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        a1     = float(self._params.get("a1", 2.0))
        a2     = float(self._params.get("a2", 1.0))
        GP     = float(self._params.get("gp", 0.5))

        # Build equilibrium pool: 4 best + their centroid (Eq. 7–8)
        order   = self._order(pop[:, -1])
        n_eq    = min(4, n)
        eq_pos  = pop[order[:n_eq], :-1].copy()           # (≤4, dim)
        centroid = np.clip(eq_pos.mean(axis=0), self._lo, self._hi)
        pool    = np.vstack([eq_pos, centroid[np.newaxis, :]])  # (5, dim)

        # t-factor  (Eq. 9)
        tfac = (1.0 - t / T) ** (a2 * t / T)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            c_eq  = pool[np.random.randint(len(pool))]    # random pool member
            lam   = np.random.uniform(0.0, 1.0, dim)
            r     = np.random.uniform(0.0, 1.0, dim)
            # Exponential factor  (Eq. 11)
            F     = a1 * np.sign(r - 0.5) * (np.exp(-lam * tfac) - 1.0)
            # Generation rate  (Eqs. 13–15)
            r1, r2 = np.random.random(), np.random.random()
            gcp   = 0.5 * r1 * np.ones(dim) * float(r2 >= GP)
            g0    = gcp * (c_eq - lam * pop[i, :-1])
            g     = g0 * F
            # Position update  (Eq. 16)
            denom = np.where(np.abs(lam) < 1e-12, 1e-12, lam)
            pos   = c_eq + (pop[i, :-1] - c_eq) * F + (g / denom) * (1.0 - F)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        # Greedy selection
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
