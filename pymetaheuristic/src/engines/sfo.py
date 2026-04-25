"""pyMetaheuristic src — Sailfish Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SFOEngine(PortedPopulationEngine):
    """Sailfish Optimizer — sailfish hunt sardine schools with attack-power mechanism."""
    algorithm_id   = "sfo"
    algorithm_name = "Sailfish Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2019.01.001"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pp=0.1, AP=4.0, epsilon=0.0001)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        pp      = float(self._params.get("pp", 0.1))
        s_size  = max(5, int(pop.shape[0] / pp)) if pp > 0 else pop.shape[0] * 10
        s_size  = min(s_size, 500)
        sardines = np.random.uniform(self._lo, self._hi, (s_size, self.problem.dimension))
        s_fit    = self._evaluate_population(sardines)
        s_pop    = np.hstack([sardines, s_fit[:, None]])
        return {"s_pop": s_pop}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        AP_init = float(self._params.get("AP", 4.0))
        eps     = float(self._params.get("epsilon", 0.0001))

        s_pop   = np.asarray(state.payload.get("s_pop"), dtype=float)
        s_size  = s_pop.shape[0]
        evals   = 0

        # Best sardine
        s_best_idx = self._best_index(s_pop[:, -1])
        s_gbest    = s_pop[s_best_idx, :-1].copy()
        order_sf   = self._order(pop[:, -1])

        # Update sailfish positions  (Eq. 6)
        PD = 1.0 - n / (n + s_size)
        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            lam = 2.0 * np.random.random() * PD - PD
            pos = s_gbest - lam * (np.random.random() * (pop[i, :-1] + s_gbest) / 2.0 - pop[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)
        new_fit = self._evaluate_population(new_pos); evals += n
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = np.hstack([new_pos, new_fit[:, None]])[mask]

        # Attack power  (Eq. 10)
        AP = AP_init * (1.0 - 2.0 * t * eps)
        if AP < 0.5:
            alpha = max(1, int(s_size * abs(AP)))
            beta  = max(1, int(dim * abs(AP)))
            chosen = np.random.choice(s_size, alpha, replace=False)
            for idx in chosen:
                dims_chosen = np.random.choice(dim, beta, replace=False)
                best_sf = pop[order_sf[0], :-1]
                s_pop[idx, dims_chosen] = np.clip(
                    (np.random.random(dim) * (best_sf - s_pop[idx, :-1] + AP))[dims_chosen],
                    self._lo[dims_chosen], self._hi[dims_chosen])
                s_pop[idx, -1] = float(self.problem.evaluate(s_pop[idx, :-1])); evals += 1
        else:
            best_sf = pop[order_sf[0], :-1]
            for idx in range(s_size):
                pos = np.clip(np.random.random() * (best_sf - s_pop[idx, :-1] + AP),
                              self._lo, self._hi)
                s_pop[idx, :-1] = pos
                s_pop[idx, -1]  = float(self.problem.evaluate(pos)); evals += 1

        # Sailfish absorb best sardines
        order_sf2 = self._order(pop[:, -1])
        order_s   = self._order(s_pop[:, -1])
        for i, sf_i in enumerate(order_sf2):
            if not order_s.size: break
            s_i = order_s[0]
            if self._is_better(float(s_pop[s_i, -1]), float(pop[sf_i, -1])):
                pop[sf_i] = s_pop[s_i]
                s_pop = np.delete(s_pop, s_i, axis=0)
                order_s = self._order(s_pop[:, -1]) if s_pop.size else np.array([], dtype=int)

        # Replenish sardines if depleted
        needed = max(0, 5 - s_pop.shape[0])
        if needed:
            new_s = np.random.uniform(self._lo, self._hi, (needed, dim))
            nf    = self._evaluate_population(new_s); evals += needed
            s_pop = np.vstack([s_pop, np.hstack([new_s, nf[:, None]])])

        return pop, evals, {"s_pop": s_pop}
