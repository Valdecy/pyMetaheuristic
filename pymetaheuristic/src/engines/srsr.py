"""pyMetaheuristic src — Shuffle-based Runner-Root Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SRSREngine(PortedPopulationEngine):
    """Shuffle-based Runner-Root Algorithm — master/slave robot Gaussian exploration."""
    algorithm_id   = "srsr"
    algorithm_name = "Shuffle-based Runner-Root Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.asoc.2017.02.028"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, mu_factor=0.667)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"SIF": 6.0, "sigma_arr": np.zeros(pop.shape[0])}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        mu_f        = float(self._params.get("mu_factor", 2.0/3.0))
        SIF         = float(state.payload.get("SIF", 6.0))
        sigma_arr   = np.asarray(state.payload.get("sigma_arr", np.zeros(n)), dtype=float)

        order    = self._order(pop[:, -1])
        master   = pop[order[0], :-1].copy()       # master robot = current best

        # Phase 1 – Accumulation: new positions via Gaussian(mu, sigma)
        new_pos1 = np.empty_like(pop[:, :-1])
        for i in range(n):
            mu_i = mu_f * master + (1.0 - mu_f) * pop[i, :-1]
            sigma_arr[i] = SIF * np.random.random() * np.max(np.abs(master - pop[i, :-1]) + 1e-12)
            pos = np.random.normal(mu_i, np.abs(sigma_arr[i]) + 1e-12, dim)
            new_pos1[i] = np.clip(pos, self._lo, self._hi)

        new_fit1 = self._evaluate_population(new_pos1)
        delta_fit = new_fit1 - pop[:, -1]          # improvement (negative=better for min)
        mask1 = self._better_mask(new_fit1, pop[:, -1])
        pop[mask1] = np.hstack([new_pos1, new_fit1[:, None]])[mask1]

        # Update SIF from best improver
        if self.problem.objective == "min":
            best_mover = int(np.argmin(delta_fit))
        else:
            best_mover = int(np.argmax(delta_fit))
        sigma_factor = 1.0 + np.random.random() * float(np.max(self._span))
        SIF = sigma_factor * float(sigma_arr[best_mover])
        if SIF > float(np.max(self._hi)):
            SIF = float(np.max(self._hi)) * np.random.random()

        # Phase 2 – Exploration: move toward master
        new_pos2 = np.empty_like(pop[:, :-1])
        order2   = self._order(pop[:, -1])
        master2  = pop[order2[0], :-1].copy()
        for i in range(n):
            gb   = np.random.uniform(-1, 1, dim)
            gb   = np.sign(gb) if gb.any() else np.ones(dim)
            pos  = pop[i, :-1] * np.random.random() \
                   + gb * (master2 - pop[i, :-1]) \
                   + (self._span) * np.random.random(dim)
            new_pos2[i] = np.clip(pos, self._lo, self._hi)

        new_fit2 = self._evaluate_population(new_pos2)
        mask2    = self._better_mask(new_fit2, pop[:, -1])
        pop[mask2] = np.hstack([new_pos2, new_fit2[:, None]])[mask2]

        return pop, n * 2, {"SIF": SIF, "sigma_arr": sigma_arr}
