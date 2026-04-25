"""pyMetaheuristic src — Archimedes Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ARCHOAEngine(PortedPopulationEngine):
    """Archimedes Optimization Algorithm — buoyancy and density update in fluid mechanics."""
    algorithm_id   = "arch_oa"
    algorithm_name = "Archimedes Optimization Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1007/s10489-020-01893-z"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, c1=2.0, c2=6.0, c3=2.0, c4=0.5,
                     acc_max=0.9, acc_min=0.1)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        return {
            "den": np.random.random(n),
            "vol": np.random.random(n),
            "acc": np.random.uniform(
                float(self._params.get("acc_min", 0.1)),
                float(self._params.get("acc_max", 0.9)), n),
        }

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        T        = max(1, self.config.max_steps or 500)
        t        = state.step + 1
        c1       = float(self._params.get("c1", 2.0))
        c2       = float(self._params.get("c2", 6.0))
        c3       = float(self._params.get("c3", 2.0))
        c4       = float(self._params.get("c4", 0.5))
        acc_max  = float(self._params.get("acc_max", 0.9))
        acc_min  = float(self._params.get("acc_min", 0.1))
        EPS      = 1e-10

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        best_idx = int(order[0])

        den  = np.asarray(state.payload.get("den", np.random.random(n)), dtype=float)
        vol  = np.asarray(state.payload.get("vol", np.random.random(n)), dtype=float)
        acc  = np.asarray(state.payload.get("acc", np.ones(n) * 0.5), dtype=float)

        # Best object's den/vol/acc stored at its index
        best_den = den[best_idx]; best_vol = vol[best_idx]; best_acc = acc[best_idx]

        tf  = np.exp(t / T)        # transfer operator  (Eq. 8)
        ddf = np.exp(1.0 - t / T) - t / T   # density decreasing factor  (Eq. 9)

        new_acc = np.empty(n)
        for i in range(n):
            den[i] = den[i] + np.random.random() * (best_den - den[i])
            vol[i] = vol[i] + np.random.random() * (best_vol - vol[i])
            if tf <= 0.5:
                j   = np.random.choice([k for k in range(n) if k != i])
                raw = (den[j] + vol[j] * acc[j]) / (den[i] * vol[i] + EPS)
            else:
                raw = (best_den + best_vol * best_acc) / (den[i] * vol[i] + EPS)
            new_acc[i] = raw

        mn, mx = new_acc.min(), new_acc.max()
        acc    = acc_max * (new_acc - mn) / (mx - mn + EPS) + acc_min

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if tf <= 0.5:
                j    = np.random.choice([k for k in range(n) if k != i])
                pos  = pop[i, :-1] + c1 * np.random.random() * acc[i] * ddf * (
                    pop[j, :-1] - pop[i, :-1])
            else:
                p    = 2.0 * np.random.random() - c4
                f    = 1 if p <= 0.5 else -1
                t_c  = c3 * tf
                pos  = best_pos + f * c2 * np.random.random() * acc[i] * ddf * (
                    t_c * best_pos - pop[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"den": den, "vol": vol, "acc": acc}
