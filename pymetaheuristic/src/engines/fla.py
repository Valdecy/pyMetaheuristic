"""pyMetaheuristic src — Fick's Law Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FLAEngine(PortedPopulationEngine):
    """Fick's Law Algorithm — two-fluid diffusion analogy with transfer operator phases."""
    algorithm_id   = "fla"
    algorithm_name = "Fick's Law Algorithm"
    family         = "physics"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2022.110146"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, C1=0.5, C2=2.0, C3=0.1, C4=0.2, C5=2.0, DD=0.01)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        C1 = float(self._params.get("C1", 0.5)); C2 = float(self._params.get("C2", 2.0))
        C3 = float(self._params.get("C3", 0.1)); C4 = float(self._params.get("C4", 0.2))
        C5 = float(self._params.get("C5", 2.0)); DD = float(self._params.get("DD", 0.01))
        EPS = 1e-30

        order = self._order(pop[:, -1])
        n1    = n // 2; n2 = n - n1
        pop1  = pop[order[:n1]]; pop2 = pop[order[n1:]]
        best1 = pop1[0, :-1]; best2 = pop2[0, :-1]
        xm1   = pop1[:, :-1].mean(axis=0); xm2 = pop2[:, :-1].mean(axis=0)

        tf  = np.sinh(t / T) ** C1
        dof = np.exp(-(C2 * tf - np.random.random())) ** C2

        new_pos = np.empty((n, dim))
        if tf < 0.9:
            tdo = C5 * tf - np.random.random()
            if tdo < np.random.random():
                nt12 = max(1, int(np.round((C4 * n1 - C3 * n1) * np.random.random() + C3 * n1)))
                for i in range(nt12):
                    dfg = np.random.randint(1, 3)
                    norm = np.linalg.norm(best2 - pop1[i, :-1]) + EPS
                    jj   = -DD * (xm2 - xm1) / norm
                    pos  = best2 + dfg * dof * np.random.random(dim) * (jj * best2 - pop1[i, :-1])
                    new_pos[order[i]] = np.clip(pos, self._lo, self._hi)
                for i in range(nt12, n1):
                    tt  = pop1[i, :-1] + dof * (np.random.random(dim) * self._span + self._lo)
                    pp  = np.random.random(dim)
                    pos = np.where(pp < 0.8, best1, np.where(pp >= 0.9, pop1[i, :-1], tt))
                    new_pos[order[i]] = np.clip(pos, self._lo, self._hi)
                for i in range(n2):
                    pos  = best2 + dof * (np.random.random(dim) * self._span + self._lo)
                    new_pos[order[n1 + i]] = np.clip(pos, self._lo, self._hi)
            else:
                nt12 = max(1, int(np.round((0.2 * n2 - 0.1 * n2) * np.random.random() + 0.1 * n2)))
                for i in range(nt12):
                    dfg = np.random.randint(1, 3)
                    norm = np.linalg.norm(best1 - pop2[i, :-1]) + EPS
                    jj   = -DD * (xm1 - xm2) / norm
                    pos  = best1 + dfg * dof * np.random.random(dim) * (jj * best1 - pop2[i, :-1])
                    new_pos[order[n1 + i]] = np.clip(pos, self._lo, self._hi)
                for i in range(nt12, n2):
                    tt  = pop2[i, :-1] + dof * (np.random.random(dim) * self._span + self._lo)
                    pp  = np.random.random(dim)
                    pos = np.where(pp < 0.8, best2, np.where(pp >= 0.9, pop2[i, :-1], tt))
                    new_pos[order[n1 + i]] = np.clip(pos, self._lo, self._hi)
                for i in range(n1):
                    pos  = best1 + dof * (np.random.random(dim) * self._span + self._lo)
                    new_pos[order[i]] = np.clip(pos, self._lo, self._hi)
        else:                      # tf >= 0.9 — pure exploitation around global best
            best_all = pop[order[0], :-1]
            for i in range(n):
                pos = best_all + dof * np.random.random(dim) * (best_all - pop[i, :-1])
                new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
