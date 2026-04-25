"""pyMetaheuristic src — Brain Storm Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BSOEngine(PortedPopulationEngine):
    """Brain Storm Optimization — k-means cluster-guided idea generation."""
    algorithm_id   = "bso"
    algorithm_name = "Brain Storm Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/978-3-642-21515-5_36"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5)

    @staticmethod
    def _k_centers(pop: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Lightweight k-means (max 10 iters) — returns centers and labels."""
        idx  = np.random.choice(pop.shape[0], k, replace=False)
        ctrs = pop[idx].copy()
        lbls = np.zeros(pop.shape[0], dtype=int)
        for _ in range(10):
            dists = np.array([[np.sum((x - c)**2) for c in ctrs] for x in pop])
            lbls  = np.argmin(dists, axis=1)
            new   = np.array([pop[lbls == j].mean(axis=0) if np.any(lbls == j)
                              else pop[np.random.randint(pop.shape[0])] for j in range(k)])
            if np.allclose(ctrs, new): break
            ctrs = new
        return ctrs, lbls

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        m           = max(2, int(self._params.get("m_clusters", 5)))
        p1, p2, p3, p4 = (float(self._params.get(k, v))
                           for k, v in [("p1",.25),("p2",.5),("p3",.75),("p4",.5)])

        m = min(m, n)
        x   = (0.5 * T - t) / max(1, T // 20)
        eps = np.random.random() / (1.0 + np.exp(-x))

        ctrs, lbls = self._k_centers(pop[:, :-1], m)

        # Possibly replace a cluster center
        if np.random.random() < p1:
            k_idx = np.random.randint(m)
            ctrs[k_idx] = np.random.uniform(self._lo, self._hi)

        m_sol   = max(1, n // m)
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            c_id = lbls[i]
            if np.random.random() < p2:               # single-cluster idea
                if np.random.random() < p3:
                    c_id = np.random.randint(m)
                c_members = np.where(lbls == c_id)[0]
                if len(c_members):
                    if np.random.random() < p3:
                        pos = ctrs[c_id] + eps * np.random.normal(0, 1, dim)
                    else:
                        j = np.random.choice(c_members)
                        pos = pop[j, :-1] + np.random.normal(0, 1, dim)
                else:
                    pos = ctrs[c_id] + eps * np.random.normal(0, 1, dim)
            else:                                     # two-cluster idea
                c1, c2 = np.random.choice(m, 2, replace=False)
                if np.random.random() < p4:
                    pos = 0.5*(ctrs[c1] + ctrs[c2]) + eps * np.random.normal(0, 1, dim)
                else:
                    j1 = np.random.choice(np.where(lbls == c1)[0]) if np.any(lbls==c1) else 0
                    j2 = np.random.choice(np.where(lbls == c2)[0]) if np.any(lbls==c2) else 0
                    pos = 0.5*(pop[j1, :-1] + pop[j2, :-1]) + eps * np.random.normal(0, 1, dim)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
