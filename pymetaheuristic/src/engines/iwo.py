"""pyMetaheuristic src — Invasive Weed Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class IWOEngine(PortedPopulationEngine):
    """Invasive Weed Optimization — seed dispersal with adaptive sigma reduction."""
    algorithm_id   = "iwo"
    algorithm_name = "Invasive Weed Optimization"
    family         = "bio"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, seed_min=2, seed_max=10,
                     exponent=2, sigma_start=1.0, sigma_end=0.01)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        seed_min    = int(self._params.get("seed_min", 2))
        seed_max    = int(self._params.get("seed_max", 10))
        exponent    = int(self._params.get("exponent", 2))
        sigma_start = float(self._params.get("sigma_start", 1.0))
        sigma_end   = float(self._params.get("sigma_end", 0.01))

        sigma  = (1.0 - t / T) ** exponent * (sigma_start - sigma_end) + sigma_end
        order  = self._order(pop[:, -1])
        best_f = float(pop[order[0],  -1])
        worst_f= float(pop[order[-1], -1])

        seeds = []
        for i in range(n):
            fit_i = float(pop[i, -1])
            denom = abs(worst_f - best_f) + 1e-30
            ratio = abs(fit_i - worst_f) / denom if self.problem.objective == "min" \
                    else abs(fit_i - best_f) / denom
            s = int(np.ceil(seed_min + (seed_max - seed_min) * ratio))
            for _ in range(s):
                pos = pop[i, :-1] + sigma * np.random.normal(0, 1, dim)
                seeds.append(np.clip(pos, self._lo, self._hi))

        if not seeds:
            return pop, 0, {}

        seed_arr = np.vstack(seeds)
        seed_fit = self._evaluate_population(seed_arr)
        combined = np.vstack([pop, np.hstack([seed_arr, seed_fit[:, None]])])
        order2   = self._order(combined[:, -1])
        pop      = combined[order2[:n]]
        return pop, len(seeds), {}
