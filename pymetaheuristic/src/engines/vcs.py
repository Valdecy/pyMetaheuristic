"""pyMetaheuristic src — Virus Colony Search Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class VCSEngine(PortedPopulationEngine):
    """Virus Colony Search — diffusion, host-cell infection and immune-response phases."""
    algorithm_id   = "vcs"
    algorithm_name = "Virus Colony Search"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.advengsoft.2015.11.004"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, lamda=0.5, sigma=1.5)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        sigma_p = float(self._params.get("sigma", 1.5))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # 1. Virus diffusion
        diff_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            sig   = (np.log1p(t) / T) * (pop[i, :-1] - best_pos)
            gauss = np.random.normal(np.random.normal(best_pos, np.abs(sig) + 1e-12))
            pos   = gauss + np.random.random() * best_pos - np.random.random() * pop[i, :-1]
            diff_pos[i] = np.clip(pos, self._lo, self._hi)
        diff_fit = self._evaluate_population(diff_pos); evals += n
        mask     = self._better_mask(diff_fit, pop[:, -1])
        pop[mask] = np.hstack([diff_pos, diff_fit[:, None]])[mask]

        # 2. Host-cell infection — Gaussian around weighted mean
        order2    = self._order(pop[:, -1])
        keep_n    = max(1, int(n * float(self._params.get("lamda", 0.5))))
        x_mean    = pop[order2[:keep_n], :-1].mean(axis=0)
        sig_decay = sigma_p * (1.0 - t / T)
        inf_pos   = np.clip(
            x_mean + sig_decay * np.random.normal(0, 1, (n, dim)),
            self._lo, self._hi)
        inf_fit   = self._evaluate_population(inf_pos); evals += n
        mask      = self._better_mask(inf_fit, pop[:, -1])
        pop[mask] = np.hstack([inf_pos, inf_fit[:, None]])[mask]

        # 3. Immune response — rank-based crossover
        order3   = self._order(pop[:, -1])
        imm_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            pr   = (dim - i + 1) / dim
            ids  = [k for k in range(n) if k != i]
            i1, i2 = np.random.choice(ids, 2, replace=False)
            temp = pop[i1, :-1] - (pop[i2, :-1] - pop[i, :-1]) * np.random.random()
            cond = np.random.random(dim) < pr
            pos  = np.where(cond, pop[i, :-1], temp)
            imm_pos[i] = np.clip(pos, self._lo, self._hi)
        imm_fit  = self._evaluate_population(imm_pos); evals += n
        mask     = self._better_mask(imm_fit, pop[:, -1])
        pop[mask] = np.hstack([imm_pos, imm_fit[:, None]])[mask]

        return pop, evals, {}
