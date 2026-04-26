"""pyMetaheuristic src — Earthworm Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EOAEngine(PortedPopulationEngine):
    """Earthworm Optimization Algorithm — reproduction and Cauchy mutation on sorted population."""
    algorithm_id   = "eoa"
    algorithm_name = "Earthworm Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1504/IJBIC.2015.10004283"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, p_c=0.9, p_m=0.01, n_best=2,
                     alpha=0.98, beta=0.9, gama=0.9)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"dyn_beta": float(self._params.get("beta", 0.9))}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim     = pop.shape[0], self.problem.dimension
        p_m        = float(self._params.get("p_m",    0.01))
        n_best     = min(int(self._params.get("n_best", 2)), n // 2)
        alpha      = float(self._params.get("alpha",  0.98))
        gama       = float(self._params.get("gama",   0.9))
        dyn_beta   = float(state.payload.get("dyn_beta", 0.9))

        order    = self._order(pop[:, -1])
        pop      = pop[order]            # sort best-first
        best_pos = pop[0, :-1].copy()

        # Reproduction phase
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            x_t1 = self._lo + self._hi - alpha * pop[i, :-1]  # Eq 1
            if i >= n_best:                    # crossover with two parents
                idx_20 = max(1, int(n * 0.2))
                pool_top  = list(range(min(idx_20, n)))
                pool_rest = list(range(min(idx_20, n), n))
                if np.random.random() < 0.5:
                    p1, p2 = np.random.choice(pool_top, 2, replace=False) if len(pool_top) >= 2 else (0, min(1, n-1))
                else:
                    p1, p2 = np.random.choice(pool_rest, 2, replace=False) if len(pool_rest) >= 2 else (0, min(1, n-1))
                r = np.random.random()
                x_child = r * pop[p2, :-1] + (1.0 - r) * pop[p1, :-1]
            else:
                x_child = pop[np.random.randint(n), :-1]
            x_t1 = dyn_beta * x_t1 + (1.0 - dyn_beta) * x_child
            new_pos[i] = np.clip(x_t1, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = np.hstack([new_pos, new_fit[:, None]])[mask]

        dyn_beta = gama * dyn_beta

        # Re-sort
        pop = pop[self._order(pop[:, -1])]
        best_pos = pop[0, :-1].copy()
        x_mean   = pop[:, :-1].mean(axis=0)

        # Cauchy mutation on non-elite
        cm_pos  = np.empty_like(pop[n_best:, :-1])
        for k, i in enumerate(range(n_best, n)):
            cw    = best_pos.copy()
            cond  = np.random.random(dim) < p_m
            cw    = np.where(cond, x_mean, cw)
            pos   = np.clip((cw + best_pos) / 2.0, self._lo, self._hi)
            cm_pos[k] = pos
        cm_fit = self._evaluate_population(cm_pos)
        mask2  = self._better_mask(cm_fit, pop[n_best:, -1])
        cm_new = np.hstack([cm_pos, cm_fit[:, None]])
        pop[n_best:][mask2] = cm_new[mask2]

        return pop, n, {"dyn_beta": dyn_beta}
