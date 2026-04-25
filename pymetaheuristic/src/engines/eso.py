"""pyMetaheuristic src — Electric Squirrel Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ESOEngine(PortedPopulationEngine):
    """Electric Squirrel Optimizer — field-resistance ionization and conductivity model."""
    algorithm_id   = "eso"
    algorithm_name = "Electric Squirrel Optimizer"
    family         = "physics"
    _REFERENCE     = {"doi": "10.3390/make7010024"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1

        pos_pop    = pop[:, :-1]
        mean_pos   = pos_pop.mean(axis=0)
        std_pos    = float(np.sqrt(np.mean(np.sum((pos_pop - mean_pos)**2, axis=1))) + 1e-30)
        ptp        = float(np.max(np.max(pos_pop, axis=0) - np.min(pos_pop, axis=0)) + 1e-30)
        resistance = std_pos / ptp

        # Field conductivity
        try:
            exp_r  = np.exp(min(resistance, 300))
            beta_  = 1.0 / (1.0 + np.exp(-exp_r/resistance) * (resistance - abs(np.log(max(1e-9, 1.0 - resistance)))))
            fc     = exp_r + np.exp(min(1.0 - resistance, 300)) * abs(np.log(max(1e-9, resistance))) * beta_
        except Exception:
            fc = 1.0
        fc = min(max(fc, 0.01), 10.0)

        # Field intensity
        try:
            iter_r = t / T
            exp_r2 = np.exp(min(resistance, 300))
            gama_  = 1.0 / (1.0 + np.exp(-exp_r2 / resistance * (resistance - abs(np.log(max(1e-9, 1.0 - iter_r))))))
            fi     = fc * gama_
        except Exception:
            fi = fc
        fi = min(max(fi, 0.01), 10.0)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()

        # Ionised region = above-threshold candidates
        pct = min(99.0, max(1.0, resistance / 2.0 * 100.0))
        fit = pop[:, -1]
        thr_fit = float(np.percentile(fit, pct))
        ionized = np.where(fit <= thr_fit)[0] if self.problem.objective == "min" \
                  else np.where(fit >= thr_fit)[0]

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if len(ionized):
                ion_pos = pop[ionized[np.random.randint(len(ionized))], :-1]
            else:
                ion_pos = best_pos
            pos = pop[i, :-1] + fc * np.random.random(dim) * (ion_pos - fi * pop[i, :-1])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
