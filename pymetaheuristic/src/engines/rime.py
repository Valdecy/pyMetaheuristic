"""pyMetaheuristic src — RIME-ice Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class RIMEEngine(PortedPopulationEngine):
    """RIME-ice Algorithm — soft-rime search and hard-rime puncture strategies."""
    algorithm_id   = "rime"
    algorithm_name = "RIME-ice Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.neucom.2023.02.010"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, sr=5.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T      = max(1, self.config.max_steps or 500)
        t      = state.step + 1
        sr     = float(self._params.get("sr", 5.0))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        rime_factor = (np.random.random() - 0.5) * 2.0 * np.cos(np.pi * t / (T / 10.0)) * \
                      (1.0 - np.round(t * sr / T) / sr)
        ee  = np.sqrt(t / T)

        fit_arr   = pop[:, -1].copy()
        norm_fit  = np.linalg.norm(fit_arr) + 1e-30
        fits_norm = fit_arr / norm_fit              # per-individual normalised fitness

        new_pos = pop[:, :-1].copy()
        for i in range(n):
            for j in range(dim):
                # Soft-rime search strategy
                if np.random.random() < ee:
                    new_pos[i, j] = best_pos[j] + rime_factor * (
                        self._lo[j] + np.random.random() * (self._hi[j] - self._lo[j]))
                # Hard-rime puncture mechanism
                if np.random.random() < fits_norm[i]:
                    new_pos[i, j] = best_pos[j]
            new_pos[i] = np.clip(new_pos[i], self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
