"""pyMetaheuristic src — Hunger Games Search Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HGSEngine(PortedPopulationEngine):
    """Hunger Games Search — hunger-weighted position update inspired by animal foraging drives."""
    algorithm_id   = "hgs"
    algorithm_name = "Hunger Games Search"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.114864"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pup=0.08, lh=10000.0)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"hunger": np.zeros(pop.shape[0])}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        PUP     = float(self._params.get("pup", 0.08))
        LH      = float(self._params.get("lh", 10000.0))
        EPS     = 1e-10
        hunger  = np.asarray(state.payload.get("hunger", np.zeros(n)), dtype=float)

        order     = self._order(pop[:, -1])
        best_fit  = float(pop[order[0],  -1])
        worst_fit = float(pop[order[-1], -1])
        best_pos  = pop[order[0], :-1].copy()

        # Update hunger values  (Eq. 2.2)
        for i in range(n):
            fit_i = float(pop[i, -1])
            if best_fit == worst_fit:
                r = 1.0
            else:
                r = abs(fit_i - best_fit) / (abs(worst_fit - best_fit) + EPS)
            hunger[i] = min(hunger[i] + r, LH)

        shrink       = 2.0 * (1.0 - t / T)
        total_hunger = float(np.sum(hunger)) + EPS

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            fit_i = float(pop[i, -1])
            E     = 1.0 / (np.exp(abs(fit_i - best_fit)) + EPS)   # Eq. 2.2 sech approx
            R     = 2.0 * shrink * np.random.random() - shrink     # Eq. 2.3
            if np.random.random() < PUP:
                W1 = hunger[i] * n / total_hunger * np.random.random()
            else:
                W1 = 1.0
            W2 = (1.0 - np.exp(-abs(hunger[i] - total_hunger))) * np.random.random() * 2.0
            r1, r2 = np.random.random(), np.random.random()
            if r1 < PUP:
                pos = pop[i, :-1] * (1.0 + np.random.normal(0, 1, dim))
            else:
                if r2 > E:
                    pos = W1 * best_pos + R * W2 * abs(best_pos - pop[i, :-1])
                else:
                    pos = W1 * best_pos - R * W2 * abs(best_pos - pop[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"hunger": hunger}
