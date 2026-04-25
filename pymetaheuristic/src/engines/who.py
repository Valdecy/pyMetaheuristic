"""pyMetaheuristic src — Wildebeest Herd Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WHOEngine(PortedPopulationEngine):
    """Wildebeest Herd Optimization — five social behaviours of migrating herds."""
    algorithm_id   = "who"
    algorithm_name = "Wildebeest Herd Optimization"
    family         = "bio"
    _REFERENCE     = {"doi": "10.3233/JIFS-190495"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_explore=3, n_exploit=3,
                     eta=0.15, p_hi=0.9,
                     local_alpha=0.9, local_beta=0.3,
                     global_alpha=0.2, global_beta=0.8,
                     delta_w=2.0, delta_c=2.0)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim       = pop.shape[0], self.problem.dimension
        eta          = float(self._params.get("eta", 0.15))
        p_hi         = float(self._params.get("p_hi", 0.9))
        l_alpha      = float(self._params.get("local_alpha", 0.9))
        l_beta       = float(self._params.get("local_beta", 0.3))
        g_alpha      = float(self._params.get("global_alpha", 0.2))
        g_beta       = float(self._params.get("global_beta", 0.8))
        delta_w      = float(self._params.get("delta_w", 2.0))
        delta_c      = float(self._params.get("delta_c", 2.0))
        n_explore    = int(self._params.get("n_explore", 3))
        n_exploit    = int(self._params.get("n_exploit", 3))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0],  :-1].copy()
        worst_pos= pop[order[-1], :-1].copy()
        evals    = 0

        # 1. Local movement (milling)
        for i in range(n):
            local_pos = np.empty((n_explore, dim))
            for j in range(n_explore):
                pos = pop[i, :-1] + eta * np.random.random() * np.random.uniform(self._lo, self._hi)
                local_pos[j] = np.clip(pos, self._lo, self._hi)
            local_fit = self._evaluate_population(local_pos); evals += n_explore
            best_local = local_pos[self._best_index(local_fit)]
            candidate  = np.clip(l_alpha * best_local + l_beta * (pop[i, :-1] - best_local),
                                 self._lo, self._hi)
            fit_cand   = float(self.problem.evaluate(candidate)); evals += 1
            if self._is_better(fit_cand, float(pop[i, -1])):
                pop[i, :-1] = candidate; pop[i, -1] = fit_cand

        # 2. Herd instinct
        for i in range(n):
            idr = np.random.randint(n)
            if self._is_better(float(pop[idr, -1]), float(pop[i, -1])) and np.random.random() < p_hi:
                candidate = np.clip(g_alpha * pop[i, :-1] + g_beta * pop[idr, :-1],
                                    self._lo, self._hi)
                fit_cand  = float(self.problem.evaluate(candidate)); evals += 1
                if self._is_better(fit_cand, float(pop[i, -1])):
                    pop[i, :-1] = candidate; pop[i, -1] = fit_cand

        # 3. Starvation avoidance + 4. Population pressure + 5. Social memory
        order2    = self._order(pop[:, -1])
        best_pos  = pop[order2[0],  :-1].copy()
        worst_pos = pop[order2[-1], :-1].copy()
        extra_pos, extra_fit = [], []
        for i in range(n):
            d_worst = np.linalg.norm(pop[i, :-1] - worst_pos)
            d_best  = np.linalg.norm(pop[i, :-1] - best_pos)
            if d_worst < delta_w:                         # starvation avoidance
                pos = pop[i, :-1] + np.random.random() * self._span * np.random.uniform(self._lo, self._hi)
                extra_pos.append(np.clip(pos, self._lo, self._hi))
            if 1.0 < d_best < delta_c:                   # population pressure
                pos = best_pos + eta * np.random.uniform(self._lo, self._hi)
                extra_pos.append(np.clip(pos, self._lo, self._hi))
            for _ in range(n_exploit):                    # social memory
                pos = best_pos + 0.1 * np.random.uniform(self._lo, self._hi)
                extra_pos.append(np.clip(pos, self._lo, self._hi))

        if extra_pos:
            ep   = np.vstack(extra_pos)
            ef   = self._evaluate_population(ep); evals += len(ep)
            comb = np.vstack([pop, np.hstack([ep, ef[:, None]])])
            ord3 = self._order(comb[:, -1])
            pop  = comb[ord3[:n]]

        return pop, evals, {}
