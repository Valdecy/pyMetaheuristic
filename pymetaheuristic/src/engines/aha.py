"""pyMetaheuristic src — Artificial Hummingbird Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class AHAEngine(PortedPopulationEngine):
    """Artificial Hummingbird Algorithm — foraging with visit-table-guided directed flight."""
    algorithm_id   = "aha"
    algorithm_name = "Artificial Hummingbird Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.cma.2022.114194"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        n = pop.shape[0]
        vt = np.full((n, n), np.nan)
        np.fill_diagonal(vt, np.nan)
        return {"visit_table": vt}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        vt = state.payload["visit_table"]
        evals = 0

        for i in range(n):
            # Direction vector
            r = np.random.random()
            if r < 1/3:
                dv = np.zeros(d)
                k = np.random.randint(1, max(2, d))
                dims = np.random.permutation(d)[:k]
                dv[dims] = 1.0
            elif r > 2/3:
                dv = np.ones(d)
            else:
                dv = np.zeros(d)
                dv[np.random.randint(d)] = 1.0

            if np.random.random() < 0.5:
                # Guided foraging
                row = vt[i].copy()
                row[i] = -np.inf
                # handle nan
                valid = ~np.isnan(row)
                if not valid.any():
                    target = np.random.randint(n)
                else:
                    row_valid = np.where(valid, row, -np.inf)
                    candidates = np.where(row_valid == row_valid.max())[0]
                    target = candidates[np.argmin(pop[candidates, -1])]

                new_pos = np.clip(
                    pop[target, :-1] + np.random.randn() * dv * (pop[i, :-1] - pop[target, :-1]),
                    lo, hi)
                new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1

                vt[i] += 1
                vt[i, i] = np.nan
                if self._is_better(new_fit, pop[i, -1]):
                    pop[i] = np.append(new_pos, new_fit)
                    vt[i, target] = 0
                    col_max = np.nanmax(vt, axis=1)
                    col_max[i] = np.nan
                    vt[:, i] = col_max + 1
                    vt[i, i] = np.nan
                else:
                    vt[i, target] = 0
            else:
                # Territorial foraging
                new_pos = np.clip(
                    pop[i, :-1] + np.random.randn() * dv * pop[i, :-1],
                    lo, hi)
                new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1

                vt[i] += 1
                vt[i, i] = np.nan
                if self._is_better(new_fit, pop[i, -1]):
                    pop[i] = np.append(new_pos, new_fit)
                    col_max = np.nanmax(vt, axis=1)
                    col_max[i] = np.nan
                    vt[:, i] = col_max + 1
                    vt[i, i] = np.nan

        # Migration
        step = state.payload.get("step", 0) + 1
        state.payload["step"] = step
        if step % (2 * n) == 0:
            worst = self._worst_index(pop[:, -1])
            new_pos = np.random.uniform(lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1
            pop[worst] = np.append(new_pos, new_fit)
            vt[worst] += 1
            col_max = np.nanmax(vt, axis=1)
            col_max[worst] = np.nan
            vt[:, worst] = col_max + 1
            vt[worst, worst] = np.nan

        state.payload["visit_table"] = vt
        return pop, evals, {}
