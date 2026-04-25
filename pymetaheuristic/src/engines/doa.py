"""pyMetaheuristic src — Deer Hunting Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class DOAEngine(PortedPopulationEngine):
    """Deer Hunting Optimization — 5-group cosine-based exploration with exploitation phase."""
    algorithm_id   = "doa"
    algorithm_name = "Deer Hunting Optimization Algorithm"
    family         = "human"
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

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        explore_end = int(0.9 * T)
        if t <= explore_end:
            new_pos = np.empty_like(pop[:, :-1])
            for m in range(5):
                g_start = int((m / 5) * n)
                g_end   = int(((m + 1) / 5) * n)
                grp     = pop[g_start:g_end]
                if not len(grp): continue
                pbest_idx = self._best_index(grp[:, -1])
                pbest     = grp[pbest_idx, :-1].copy()

                aa  = max(1, int(np.ceil(dim / 8 / (m + 1))))
                bb  = int(np.ceil(dim / 3 / (m + 1))) + 1
                bb  = max(aa + 1, bb)
                kk  = max(1, np.random.randint(aa, bb))

                for li, gi in enumerate(range(g_start, g_end)):
                    pos      = pbest.copy()
                    in_idx   = np.random.choice(dim, kk, replace=False)
                    cos_term = (np.cos((t + T / 10) * np.pi / T) + 1.0) / 2.0
                    if np.random.random() < 0.9:
                        for j in in_idx:
                            pos[j] = pbest[j] + np.random.random() * self._span[j] * cos_term + self._lo[j] * cos_term
                            if pos[j] > self._hi[j] or pos[j] < self._lo[j]:
                                if dim > 15 and len(range(n)) > 1:
                                    rdx = np.random.choice([k for k in range(n) if k != gi])
                                    pos[j] = pop[rdx, j]
                                else:
                                    pos[j] = np.random.uniform(self._lo[j], self._hi[j])
                    else:
                        for j in in_idx:
                            rdx    = np.random.choice([k for k in range(n) if k != gi])
                            pos[j] = pop[rdx, j]
                    new_pos[gi] = np.clip(pos, self._lo, self._hi)
            new_fit = self._evaluate_population(new_pos); evals += n
            pop     = np.hstack([new_pos, new_fit[:, None]])
        else:
            # Exploitation phase
            new_pos = np.empty_like(pop[:, :-1])
            for i in range(n):
                km   = max(2, int(np.ceil(dim / 3)))
                k    = np.random.randint(2, km + 1)
                idxs = np.random.choice(dim, k, replace=False)
                pos  = best_pos.copy()
                for j in idxs:
                    others = [x for x in range(n) if x != i]
                    r1, r2 = np.random.choice(others, 2, replace=False) if len(others) >= 2 else (0, 1)
                    pos[j] = best_pos[j] + np.random.random() * (pop[r1, j] - pop[r2, j])
                new_pos[i] = np.clip(pos, self._lo, self._hi)
            new_fit = self._evaluate_population(new_pos); evals += n
            mask    = self._better_mask(new_fit, pop[:, -1])
            pop[mask] = np.hstack([new_pos, new_fit[:, None]])[mask]

        return pop, evals, {}
