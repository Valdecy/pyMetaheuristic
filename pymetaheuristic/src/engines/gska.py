"""pyMetaheuristic src — Gaining-Sharing Knowledge Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class GSKAEngine(PortedPopulationEngine):
    """Gaining-Sharing Knowledge Algorithm — junior and senior knowledge sharing phases."""
    algorithm_id   = "gska"
    algorithm_name = "Gaining-Sharing Knowledge Algorithm"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s13042-019-01053-x"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pb=0.1, kr=0.7, kf=0.5, kg=1)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        pb      = float(self._params.get("pb", 0.1))
        kr      = float(self._params.get("kr", 0.7))
        kf      = float(self._params.get("kf", 0.5))
        kg      = int(self._params.get("kg", 1))

        order    = self._order(pop[:, -1])
        pop      = pop[order]             # sort best-first
        dd       = max(0, int(dim * (1.0 - t / T) ** kg))

        id1_end  = max(1, int(pb * n))
        id2_start = min(n - 1, int(id1_end + n * (1 - 2 * pb)))

        new_pos  = np.empty_like(pop[:, :-1])
        for i in range(n):
            previ = max(0, i - 1); nexti = min(n - 1, i + 1)
            if i == 0: previ, nexti = 2, 1
            elif i == n - 1: previ, nexti = i - 2, i - 1

            candidates = list(set(range(n)) - {previ, i, nexti})
            rand_idx   = np.random.choice(candidates) if candidates else max(0, i - 1)
            pos        = pop[i, :-1].copy()

            for j in range(dim):
                if j < dd:                # junior gaining and sharing
                    if np.random.random() <= kr:
                        if self._is_better(float(pop[rand_idx, -1]), float(pop[i, -1])):
                            pos[j] += kf * (pop[previ, j] - pop[nexti, j] + pop[rand_idx, j] - pop[i, j])
                        else:
                            pos[j] += kf * (pop[previ, j] - pop[nexti, j] + pop[i, j] - pop[rand_idx, j])
                else:                     # senior gaining and sharing
                    if np.random.random() <= kr:
                        best_pool = list(set(range(id1_end)) - {i}) or [0]
                        worst_pool = list(set(range(id2_start, n)) - {i}) or [n - 1]
                        mid_pool   = list(set(range(id1_end, id2_start)) - {i}) or list(range(n))
                        rb = np.random.choice(best_pool)
                        rw = np.random.choice(worst_pool)
                        rm = np.random.choice(mid_pool)
                        if self._is_better(float(pop[rm, -1]), float(pop[i, -1])):
                            pos[j] += kf * (pop[rb, j] - pop[rw, j] + pop[rm, j] - pop[i, j])
                        else:
                            pos[j] += kf * (pop[rb, j] - pop[rw, j] + pop[i, j] - pop[rm, j])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        mask    = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {}
