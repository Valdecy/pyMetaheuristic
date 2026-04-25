"""pyMetaheuristic src — Henry Gas Solubility Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HGSOEngine(PortedPopulationEngine):
    """Henry Gas Solubility Optimization — cluster-based gas pressure/solubility analogy."""
    algorithm_id   = "hgso"
    algorithm_name = "Henry Gas Solubility Optimization"
    family         = "physics"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_clusters=2)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n_cl  = max(2, min(int(self._params.get("n_clusters", 2)), pop.shape[0] // 2))
        n_el  = pop.shape[0] // n_cl
        H_j   = np.random.random() * 5e-4        # Henry's coeff (simplified)
        C_j   = np.random.random() * 0.1
        P_ij  = np.random.random() * 1.0
        return {"n_cl": n_cl, "n_el": n_el, "H_j": H_j, "C_j": C_j, "P_ij": P_ij}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        T        = max(1, self.config.max_steps or 500)
        t        = state.step + 1
        n_cl     = int(state.payload.get("n_cl", 2))
        n_el     = int(state.payload.get("n_el", max(1, n // n_cl)))
        H_j      = float(state.payload.get("H_j", 1e-4))
        C_j      = float(state.payload.get("C_j", 0.05))
        P_ij     = float(state.payload.get("P_ij", 0.5))
        T0, K, alpha, beta, eps = 298.15, 1.0, 1, 1.0, 0.05
        evals    = 0

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        best_fit = float(pop[order[0], -1])

        # Update Henry's coefficient (Eq. 8)
        H_j = H_j * np.exp(-C_j * (1.0/np.exp(-t/T) - 1.0/T0))
        S_ij = K * H_j * P_ij

        # Cluster update
        new_pop = pop.copy()
        for cl in range(n_cl):
            start = cl * n_el
            end   = min(start + n_el, n)
            if start >= n: break
            grp   = order[start:end]
            if not len(grp): continue
            # best in cluster
            p_best_pos = pop[grp[0], :-1].copy()
            p_best_fit = float(pop[grp[0], -1])

            for j in grp:
                F     = -1.0 if np.random.random() < 0.5 else 1.0
                gama  = beta * np.exp(-((p_best_fit + eps) / (float(pop[j, -1]) + eps)))
                pos   = pop[j, :-1] + F * np.random.random() * gama * (p_best_pos - pop[j, :-1]) \
                      + F * np.random.random() * alpha * (S_ij * best_pos - pop[j, :-1])
                pos   = np.clip(pos, self._lo, self._hi)
                fit   = float(self.problem.evaluate(pos)); evals += 1
                if self._is_better(fit, float(pop[j, -1])):
                    new_pop[j, :-1] = pos; new_pop[j, -1] = fit

        # Replace worst N_w agents (Eq. 12)
        N_w = max(1, int(n * (np.random.uniform(0, 0.1) + 0.1)))
        worst_idx = self._order(new_pop[:, -1])[-N_w:]
        for idx in worst_idx:
            pos = np.random.uniform(self._lo, self._hi)
            fit = float(self.problem.evaluate(pos)); evals += 1
            new_pop[idx, :-1] = pos; new_pop[idx, -1] = fit

        return new_pop, evals, {"n_cl": n_cl, "n_el": n_el, "H_j": H_j, "C_j": C_j, "P_ij": P_ij}
