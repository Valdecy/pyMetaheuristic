"""pyMetaheuristic src — Nelder-Mead Method Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class NMMEngine(PortedPopulationEngine):
    """Nelder-Mead Method — simplex reflection, expansion, contraction and shrink."""
    algorithm_id = "nmm"
    algorithm_name = "Nelder-Mead Method"
    family = "trajectory"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=None, alpha=0.1, gamma=0.3, rho=-0.2, sigma=-0.2)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        ps = self._params.get("population_size", None)
        self._n = max(problem.dimension + 1, int(ps) if ps is not None else problem.dimension + 1)

    def _step_impl(self, state, pop):
        order = self._order(pop[:, -1]); pop = pop[order].copy()
        centroid = np.mean(pop[:-1, :-1], axis=0)
        worst = pop[-1, :-1]
        alpha = float(self._params.get("alpha", 0.1)); gamma = float(self._params.get("gamma", 0.3)); rho = float(self._params.get("rho", -0.2)); sigma = float(self._params.get("sigma", -0.2))
        evals = 0
        xr = np.clip(centroid + alpha * (centroid - worst), self._lo, self._hi); fr = float(self.problem.evaluate(xr)); evals += 1
        best_fit, second_worst_fit, worst_fit = pop[0, -1], pop[-2, -1], pop[-1, -1]
        between = (self._is_better(fr, second_worst_fit) or fr == second_worst_fit) and (self._is_better(best_fit, fr) or fr == best_fit)
        if between:
            pop[-1, :-1], pop[-1, -1] = xr, fr
        elif self._is_better(fr, best_fit):
            xe = np.clip(centroid + gamma * (centroid - worst), self._lo, self._hi); fe = float(self.problem.evaluate(xe)); evals += 1
            pop[-1, :-1], pop[-1, -1] = (xe, fe) if self._is_better(fe, fr) else (xr, fr)
        else:
            xc = np.clip(centroid + rho * (centroid - worst), self._lo, self._hi); fc = float(self.problem.evaluate(xc)); evals += 1
            if self._is_better(fc, worst_fit):
                pop[-1, :-1], pop[-1, -1] = xc, fc
            else:
                newpos = np.clip(pop[0, :-1] + sigma * (pop[1:, :-1] - pop[0, :-1]), self._lo, self._hi)
                fit = self._evaluate_population(newpos); evals += newpos.shape[0]
                pop[1:, :-1], pop[1:, -1] = newpos, fit
        return pop, evals, {}
