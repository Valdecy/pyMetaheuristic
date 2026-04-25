"""pyMetaheuristic src — Bonobo Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class BonobOEngine(PortedPopulationEngine):
    """Bonobo Optimizer — self-adjusting search with positive/negative phases."""
    algorithm_id   = "bono"
    algorithm_name = "Bonobo Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10489-021-02830-0"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return dict(npc=0, ppc=0, p_xgm=0.001, tsgs_factor=0.025, p_p=0.5, p_d=0.5,
                    best_pos=pop[self._best_index(pop[:,-1]), :-1].copy(),
                    best_cost=float(pop[self._best_index(pop[:,-1]), -1]))

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        pl = state.payload
        scab, scsb = 1.25, 1.3
        rcpp = 0.0035
        tsgs_max_limit = 0.05

        best_pos = pl["best_pos"].copy()
        best_cost = pl["best_cost"]
        p_xgm = pl["p_xgm"]
        tsgs_factor = pl["tsgs_factor"]
        p_p = pl["p_p"]
        p_d = pl["p_d"]
        npc = pl["npc"]
        ppc = pl["ppc"]

        tsgs_max = max(2, int(np.ceil(n * tsgs_factor)))

        for i in range(n):
            others = [k for k in range(n) if k != i]
            tsg = max(2, np.random.randint(2, max(3, tsgs_max+1)))
            q = np.random.choice(others, size=min(tsg, len(others)), replace=False)
            best_q = q[np.argmin(pop[q, -1])]

            if pop[i, -1] < pop[best_q, -1]:
                p_idx = q[np.random.randint(len(q))]
                flag = 1
            else:
                p_idx = best_q
                flag = -1

            new_pos = np.empty(d)
            if np.random.random() <= p_p:
                r1 = np.random.random(d)
                new_pos = pop[i, :-1] + scab * r1 * (best_pos - pop[i, :-1]) + \
                          flag * scsb * (1-r1) * (pop[i, :-1] - pop[p_idx, :-1])
            else:
                for j in range(d):
                    if np.random.random() <= p_xgm:
                        rv = np.random.random()
                        if best_pos[j] >= pop[i, j]:
                            if np.random.random() <= p_d:
                                beta1 = np.exp(rv**2 + rv - 2/max(rv, 1e-10))
                                new_pos[j] = pop[i, j] + beta1 * (hi[j] - pop[i, j])
                            else:
                                beta2 = np.exp(-rv**2 + 2*rv - 2/max(rv, 1e-10))
                                new_pos[j] = pop[i, j] - beta2 * (pop[i, j] - lo[j])
                        else:
                            if np.random.random() <= p_d:
                                beta1 = np.exp(rv**2 + rv - 2/max(rv, 1e-10))
                                new_pos[j] = pop[i, j] - beta1 * (pop[i, j] - lo[j])
                            else:
                                beta2 = np.exp(-rv**2 + 2*rv - 2/max(rv, 1e-10))
                                new_pos[j] = pop[i, j] + beta2 * (hi[j] - pop[i, j])
                    else:
                        if flag == 1 or np.random.random() <= p_d:
                            new_pos[j] = pop[i, j] + flag * np.exp(-np.random.random()) * (pop[i, j] - pop[p_idx, j])
                        else:
                            new_pos[j] = pop[p_idx, j]

            new_pos = np.clip(new_pos, lo, hi)
            new_fit = float(self._evaluate_population(new_pos[None])[0]); evals += 1

            if self._is_better(new_fit, pop[i, -1]) or np.random.random() <= p_xgm:
                pop[i, :-1] = new_pos
                pop[i, -1] = new_fit
                if self._is_better(new_fit, best_cost):
                    best_cost = new_fit
                    best_pos = new_pos.copy()

        prev_best = pl["best_cost"]
        if self._is_better(best_cost, prev_best):
            npc = 0; ppc += 1
            cp = min(0.5, ppc * rcpp)
            p_xgm = 0.001
            p_p = 0.5 + cp; p_d = p_p
            tsgs_factor = min(tsgs_max_limit, 0.025 + ppc * rcpp**2)
        else:
            npc += 1; ppc = 0
            cp = -min(0.5, npc * rcpp)
            p_xgm = min(0.5, 0.001 + npc * rcpp**2)
            tsgs_factor = max(0, 0.025 - npc * rcpp**2)
            p_p = 0.5 + cp; p_d = 0.5

        state.payload.update(dict(npc=npc, ppc=ppc, p_xgm=p_xgm, tsgs_factor=tsgs_factor,
                                   p_p=p_p, p_d=p_d, best_pos=best_pos, best_cost=best_cost))
        return pop, evals, {}
