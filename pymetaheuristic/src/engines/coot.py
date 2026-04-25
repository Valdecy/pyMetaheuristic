"""pyMetaheuristic src — COOT Bird Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class COOTEngine(PortedPopulationEngine):
    algorithm_id   = "coot"
    algorithm_name = "COOT Bird Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2021.115352"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step
        max_iter = self._params.get("max_iterations", 1000)
        n_leader = max(1, int(np.ceil(0.1 * n)))
        n_coot  = n - n_leader
        B = 2 - t * (1/max_iter)
        A = 1 - t * (1/max_iter)
        order = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        coot_pos  = pop[order[n_leader:], :-1].copy()
        coot_fit  = pop[order[n_leader:], -1].copy()
        lead_pos  = pop[order[:n_leader], :-1].copy()
        lead_fit  = pop[order[:n_leader], -1].copy()
        for i in range(n_coot):
            k = i % n_leader
            if np.random.random() < 0.5:
                R  = (-1 + 2*np.random.random()) if np.random.random()<0.5 else (-1 + 2*np.random.random(d))
                R1 = np.random.random() if np.random.random()<0.5 else np.random.random(d)
                coot_pos[i] = 2*R1*np.cos(2*np.pi*R)*(lead_pos[k]-coot_pos[i]) + lead_pos[k]
            else:
                if np.random.random()<0.5 and i>0:
                    coot_pos[i] = (coot_pos[i] + coot_pos[i-1]) / 2
                else:
                    Q = np.random.uniform(lo, hi)
                    coot_pos[i] = coot_pos[i] + A * np.random.random() * (Q - coot_pos[i])
            coot_pos[i] = np.clip(coot_pos[i], lo, hi)
        for i in range(n_coot):
            k = i % n_leader
            new_fit = float(self._evaluate_population(coot_pos[i][None])[0]); evals += 1
            if self._is_better(new_fit, lead_fit[k]):
                lead_pos[k], coot_pos[i] = coot_pos[i].copy(), lead_pos[k].copy()
                lead_fit[k], coot_fit[i] = new_fit, lead_fit[k]
            else:
                coot_fit[i] = new_fit
        for i in range(n_leader):
            R  = np.full(d, -1+2*np.random.random()) if np.random.random()<0.5 else -1+2*np.random.random(d)
            R3 = np.random.random() if np.random.random()<0.5 else np.random.random(d)
            if np.random.random()<0.5:
                tmp = np.clip(B*R3*np.cos(2*np.pi*R)*(best_pos-lead_pos[i])+best_pos, lo, hi)
            else:
                tmp = np.clip(B*R3*np.cos(2*np.pi*R)*(best_pos-lead_pos[i])-best_pos, lo, hi)
            tmp_fit = float(self._evaluate_population(tmp[None])[0]); evals += 1
            if self._is_better(tmp_fit, pop[order[0], -1]):
                lead_fit[i] = pop[order[0], -1]
                lead_pos[i] = best_pos.copy()
                pop[order[0], -1] = tmp_fit
                pop[order[0], :-1] = tmp
                best_pos = tmp.copy()
        all_pos = np.vstack([lead_pos, coot_pos])
        all_fit = np.concatenate([lead_fit, coot_fit])
        pop = np.hstack([all_pos, all_fit[:, None]])
        return pop, evals, {}
