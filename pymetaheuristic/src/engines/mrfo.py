"""pyMetaheuristic src — Manta Ray Foraging Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class MRFOEngine(PortedPopulationEngine):
    """Manta Ray Foraging Optimization — chain, cyclone and somersault foraging strategies."""
    algorithm_id   = "mrfo"
    algorithm_name = "Manta Ray Foraging Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2019.103300"}
    capabilities   = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, somersault_range=2.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        S           = float(self._params.get("somersault_range", 2.0))

        order       = self._order(pop[:, -1])
        best_pos    = pop[order[0], :-1].copy()

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() < 0.5:               # Cyclone foraging  (Eqs. 5–7)
                r1   = np.random.random()
                beta = 2.0 * np.exp(r1 * (T - t) / T) * np.sin(2.0 * np.pi * r1)
                if t / T < np.random.random():         # towards random point
                    x_rand = np.random.uniform(self._lo, self._hi)
                    ref    = x_rand
                    prev   = x_rand if i == 0 else pop[i - 1, :-1]
                else:                                  # towards best
                    ref    = best_pos
                    prev   = best_pos if i == 0 else pop[i - 1, :-1]
                pos = ref + np.random.random() * (prev - pop[i, :-1]) + beta * (ref - pop[i, :-1])
            else:                                       # Chain foraging  (Eqs. 1–2)
                r     = np.random.random()
                alpha = 2.0 * r * np.sqrt(abs(np.log(r + 1e-30)))
                prev  = best_pos if i == 0 else pop[i - 1, :-1]
                pos   = pop[i, :-1] + r * (prev - pop[i, :-1]) + alpha * (best_pos - pop[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit  = self._evaluate_population(new_pos)
        new_pop  = np.hstack([new_pos, new_fit[:, None]])
        mask     = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        evals    = n

        # Somersault foraging  (Eq. 8) — all individuals
        som_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            r1, r2   = np.random.random(), np.random.random()
            best_now = pop[self._best_index(pop[:, -1]), :-1]
            s_pos    = pop[i, :-1] + S * (r1 * best_now - r2 * pop[i, :-1])
            som_pos[i] = np.clip(s_pos, self._lo, self._hi)
        som_fit  = self._evaluate_population(som_pos)
        som_pop  = np.hstack([som_pos, som_fit[:, None]])
        mask2    = self._better_mask(som_fit, pop[:, -1])
        pop[mask2] = som_pop[mask2]
        evals   += n

        return pop, evals, {}
