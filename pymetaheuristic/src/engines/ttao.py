"""pyMetaheuristic src — Triangulation Topology Aggregation Optimizer Engine"""

from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class TTAOEngine(PortedPopulationEngine):
    algorithm_id   = "ttao"
    algorithm_name = "Triangulation Topology Aggregation Optimizer"
    family         = "math"

    _REFERENCE = {"doi": "10.1016/j.eswa.2023.121744"}

    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )

    _DEFAULTS = dict(population_size=30)
    _MIN_P_SIZE = 6

    def __init__(self, problem, config):
        """Initialize TTAO with a minimum viable population size.

        PortedPopulationEngine expects only (problem, config). Algorithm-specific
        parameters must be passed through config.params, so we normalize
        population_size there before delegating to the base class.
        """
        from dataclasses import replace

        params = dict(config.params)
        population_size = params.get("population_size", self._DEFAULTS["population_size"])
        params["population_size"] = max(int(population_size), self._MIN_P_SIZE)

        super().__init__(problem, replace(config, params=params))

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0

        t = state.step
        max_iter = self._params.get("max_iterations", 1000)

        N = max(1, n // 3)

        X1 = pop[:N, :-1].copy()

        l = 9 * np.exp(-t / max_iter)

        X2 = np.zeros((N, d))
        X3 = np.zeros((N, d))

        for i in range(N):
            theta = np.random.random(d) * np.pi
            X2[i] = np.clip(X1[i] + l * np.cos(theta), lo, hi)
            X3[i] = np.clip(X1[i] + l * np.cos(theta + np.pi / 3), lo, hi)

        r1, r2 = np.random.random(), np.random.random()
        X4 = np.clip(r1 * X1 + r2 * X2 + (1 - r1 - r2) * X3, lo, hi)

        X1_fit = self._evaluate_population(X1)
        evals += N

        X2_fit = self._evaluate_population(X2)
        evals += N

        X3_fit = self._evaluate_population(X3)
        evals += N

        X4_fit = self._evaluate_population(X4)
        evals += N

        Xall = np.vstack([X1, X2, X3, X4])
        fall = np.concatenate([X1_fit, X2_fit, X3_fit, X4_fit])

        order = np.argsort(fall)

        X_best1 = Xall[order[:N]]
        f_best1 = fall[order[:N]]

        X_best2 = Xall[order[1:N + 1]]
        f_best2 = fall[order[1:N + 1]]

        # Greedy crossover
        XG = np.zeros((N, d))
        fG = np.zeros(N)

        for i in range(N):
            others = [j for j in range(N) if j != i]

            idx = np.random.randint(len(others))
            j = others[idx]

            r = np.random.random(d)

            XG[i] = np.clip(
                r * X_best1[i] + (1 - r) * X_best1[j],
                lo,
                hi,
            )

            fG[i] = float(self._evaluate_population(XG[i][None])[0])
            evals += 1

            if fG[i] < f_best1[i]:
                X_best1[i] = XG[i]
                f_best1[i] = fG[i]
            elif fG[i] < f_best2[i]:
                X_best2[i] = XG[i]
                f_best2[i] = fG[i]

        # Contraction
        XC = np.zeros((N, d))
        fC = np.zeros(N)

        denom = max(1, max_iter - 1)
        a_coef = (np.e - np.e**3) / denom
        b_coef = np.e**3 - a_coef

        for i in range(N):
            al = np.log(a_coef * t + b_coef)

            XC[i] = np.clip(
                X_best1[i] + al * (X_best1[i] - X_best2[i]),
                lo,
                hi,
            )

            fC[i] = float(self._evaluate_population(XC[i][None])[0])
            evals += 1

            if fC[i] < f_best1[i]:
                X_best1[i] = XC[i]
                f_best1[i] = fC[i]

        rest = n - N

        if rest > 0:
            X00 = np.random.uniform(lo, hi, (rest, d))
            f00 = self._evaluate_population(X00)
            evals += rest

            comb = np.vstack([X_best1, X00])
            fc = np.concatenate([f_best1, f00])

            ord2 = np.argsort(fc)[:N]

            X_best1 = comb[ord2]
            f_best1 = fc[ord2]

        pop = np.hstack([X_best1, f_best1[:, None]])

        diff = n - N

        if diff > 0:
            extra = np.random.uniform(lo, hi, (diff, d))
            extra_fit = self._evaluate_population(extra)
            evals += diff

            pop = np.vstack([
                pop,
                np.hstack([extra, extra_fit[:, None]]),
            ])

        return pop, evals, {}
