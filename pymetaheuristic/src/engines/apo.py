"""pyMetaheuristic src — Artificial Protozoa Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class APOEngine(PortedPopulationEngine):
    """Artificial Protozoa Optimizer — dormancy, reproduction, foraging behaviors."""
    algorithm_id   = "apo"
    algorithm_name = "Artificial Protozoa Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2024.111737"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, pf_max=0.1)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        pf_max = float(self._params.get("pf_max", 0.1))
        t = state.step
        max_iter = self._params.get("max_iterations", 1000)

        order = self._order(pop[:, -1])
        pop = pop[order]

        pf = pf_max * np.random.random()
        ri = np.random.choice(n, size=max(1, int(np.ceil(n * pf))), replace=False)
        ri_set = set(ri)

        new_pop = pop[:, :-1].copy()

        pah = 0.5 * (1 + np.cos(t / max_iter * np.pi))
        f = np.random.random() * (1 + np.cos(t / max_iter * np.pi))

        for i in range(n):
            Mf = np.zeros(d); Mf[np.random.permutation(d)[:max(1, int(np.ceil(d * (i+1) / n)))]] = 1
            if i in ri_set:
                pdr = 0.5 * (1 + np.cos((1 - (i+1)/n) * np.pi))
                if np.random.random() < pdr:
                    new_pop[i] = np.random.uniform(lo, hi)
                else:
                    flag = np.random.choice([-1, 1])
                    Mr = np.zeros(d); Mr[np.random.permutation(d)[:max(1, int(np.ceil(np.random.random()*d)))]] = 1
                    new_pop[i] = np.clip(
                        pop[i, :-1] + flag * np.random.random() * np.random.uniform(lo, hi) * Mr,
                        lo, hi)
            else:
                j = np.random.randint(n)
                if np.random.random() < pah:
                    km = max(0, i-1); kp = min(n-1, i+1)
                    if n > 1:
                        km = np.random.randint(0, max(1, i)) if i > 0 else 0
                        kp = i + np.random.randint(1, max(2, n-i)) if i < n-1 else n-1
                    wa = np.exp(-abs(pop[km, -1] / (pop[kp, -1] + 1e-300)))
                    epn = wa * (pop[km, :-1] - pop[kp, :-1])
                    new_pop[i] = np.clip(
                        pop[i, :-1] + f * (pop[j, :-1] - pop[i, :-1] + epn) * Mf, lo, hi)
                else:
                    km = max(0, i-1); kp = min(n-1, i+1)
                    wh = np.exp(-abs(pop[km, -1] / (pop[kp, -1] + 1e-300)))
                    epn = wh * (pop[km, :-1] - pop[kp, :-1])
                    flag = np.random.choice([-1, 1])
                    Xnear = (1 + flag * np.random.randint(1, max(2, d)) * (1 - t/max_iter)) * pop[i, :-1]
                    new_pop[i] = np.clip(
                        pop[i, :-1] + f * (Xnear - pop[i, :-1] + epn) * Mf, lo, hi)

        new_fits = self._evaluate_population(new_pop); evals += n
        mask = self._better_mask(new_fits, pop[:, -1])
        pop[mask] = np.hstack([new_pop, new_fits[:, None]])[mask]
        return pop, evals, {}
