"""pyMetaheuristic src — Spider Monkey Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SMOEngine(PortedPopulationEngine):
    """Spider Monkey Optimization — fission-fusion social dynamics with local and global leaders."""
    algorithm_id   = "smo"
    algorithm_name = "Spider Monkey Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s12293-013-0128-0"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, max_groups=5, perturbation_rate=0.7)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n  = pop.shape[0]
        mg = max(2, int(self._params.get("max_groups", 5)))
        # Start with one group
        groups = [list(range(n))]
        return {"groups": groups, "local_leaders": [0], "mg": mg,
                "local_limits": np.zeros(n, int), "global_limit": 0}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        pr      = float(self._params.get("perturbation_rate", 0.7))
        mg      = int(state.payload.get("mg", 5))

        groups  = state.payload.get("groups", [list(range(n))])
        ll_idx  = state.payload.get("local_leaders", [0])   # local leader index per group
        g_limit = int(state.payload.get("global_limit", 0))
        loc_lim = np.asarray(state.payload.get("local_limits", np.zeros(n, int)), dtype=int)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        # Phase 1: Local Leader Phase
        for gi, grp in enumerate(groups):
            if not grp: continue
            li = ll_idx[gi] if gi < len(ll_idx) else grp[0]
            li_pos = pop[li, :-1].copy()
            for i in grp:
                j   = np.random.choice([x for x in grp if x != i] or [i])
                pos = pop[i, :-1].copy()
                rd  = np.random.random(dim) < pr
                pos[rd] = pop[i, :-1][rd] + np.random.uniform(-1, 1) * (li_pos[rd] - pop[i, :-1][rd]) \
                                          + np.random.uniform(-1, 1) * (pop[j, :-1][rd] - pop[i, :-1][rd])
                pos = np.clip(pos, self._lo, self._hi)
                fit = float(self.problem.evaluate(pos)); evals += 1
                if self._is_better(fit, float(pop[i, -1])):
                    pop[i, :-1] = pos; pop[i, -1] = fit
                    loc_lim[i]  = 0
                else:
                    loc_lim[i] += 1

        # Phase 2: Global Leader Phase
        for gi, grp in enumerate(groups):
            if not grp: continue
            for i in grp:
                pos = pop[i, :-1].copy()
                rd  = np.random.random(dim) < pr
                pos[rd] = pop[i, :-1][rd] + np.random.uniform(-1, 1) * (best_pos[rd] - pop[i, :-1][rd])
                pos = np.clip(pos, self._lo, self._hi)
                fit = float(self.problem.evaluate(pos)); evals += 1
                if self._is_better(fit, float(pop[i, -1])):
                    pop[i, :-1] = pos; pop[i, -1] = fit

        # Phase 3: Local Leader Decision Phase — random reset if stagnant
        limit_thresh = int(0.6 * dim * len(groups))
        for gi, grp in enumerate(groups):
            if not grp: continue
            li = ll_idx[gi] if gi < len(ll_idx) else grp[0]
            if loc_lim[li] >= limit_thresh:
                pos = np.clip(
                    pop[li, :-1] + np.random.random(dim) * (best_pos - pop[li, :-1]),
                    self._lo, self._hi)
                fit = float(self.problem.evaluate(pos)); evals += 1
                pop[li, :-1] = pos; pop[li, -1] = fit
                loc_lim[li]  = 0

        # Global limit: fission/fusion
        g_limit += 1
        g_thresh = int(0.6 * dim * len(groups))
        if g_limit >= g_thresh and len(groups) < mg:
            # Fission: split largest group
            largest = max(range(len(groups)), key=lambda x: len(groups[x]))
            grp     = groups[largest]
            if len(grp) >= 4:
                mid = len(grp) // 2
                groups[largest] = grp[:mid]
                groups.append(grp[mid:])
                ll_idx = [grp[self._best_index(pop[g, -1])] if g else 0 for g in groups]
            g_limit = 0
        elif g_limit >= g_thresh and len(groups) > 1:
            # Fusion: merge two smallest groups
            sizes   = [(len(g), i) for i, g in enumerate(groups)]
            sizes.sort()
            i1, i2  = sizes[0][1], sizes[1][1]
            groups[i1].extend(groups[i2])
            groups.pop(i2)
            ll_idx  = [g[self._best_index(pop[g, -1])] if g else 0 for g in groups]
            g_limit = 0

        return pop, evals, {"groups": groups, "local_leaders": ll_idx,
                            "global_limit": g_limit, "local_limits": loc_lim, "mg": mg}
