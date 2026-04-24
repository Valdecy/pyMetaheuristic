"""pyMetaheuristic src — Bacterial Foraging Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class BFOEngine(PortedPopulationEngine):
    """Bacterial Foraging Optimization — tumble/swim with reproduction and elimination."""
    algorithm_id = "bfo"
    algorithm_name = "Bacterial Foraging Optimization"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, step_size=0.1, n_swim=4, reproduction_interval=10, elimination_prob=0.25)

    def _initialize_payload(self, pop):
        return {"health": np.zeros(pop.shape[0], dtype=float)}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        step_size = float(self._params.get("step_size", 0.1)) * self._span
        n_swim = max(1, int(self._params.get("n_swim", 4)))
        health = np.asarray(state.payload.get("health", np.zeros(n)), dtype=float)
        if health.shape[0] != n:
            health = np.zeros(n)
        evals = 0
        for i in range(n):
            direction = np.random.normal(size=dim)
            norm = float(np.linalg.norm(direction)) or 1.0
            direction = direction / norm
            best_pos = pop[i, :-1].copy()
            best_fit = float(pop[i, -1])
            for _ in range(n_swim):
                trial = np.clip(best_pos + direction * step_size, self._lo, self._hi)
                fit = float(self.problem.evaluate(trial)); evals += 1
                if self._is_better(fit, best_fit):
                    best_pos, best_fit = trial, fit
                else:
                    break
            pop[i, :-1], pop[i, -1] = best_pos, best_fit
            health[i] += best_fit if self.problem.objective == "min" else -best_fit
        if (state.step + 1) % max(1, int(self._params.get("reproduction_interval", 10))) == 0:
            order = np.argsort(health)
            survivors = order[:max(1, n // 2)]
            duplicated = np.resize(survivors, n)
            pop = pop[duplicated].copy()
            pop[:, :-1] = np.clip(pop[:, :-1] + np.random.normal(0.0, 0.01 * self._span, (n, dim)), self._lo, self._hi)
            pop[:, -1] = self._evaluate_population(pop[:, :-1]); evals += n
            health = np.zeros(n)
        elim = np.random.rand(n) < float(self._params.get("elimination_prob", 0.25)) / max(1, n)
        if np.any(elim):
            repl = self._pop_from_positions(self._new_positions(int(elim.sum())))
            evals += int(elim.sum())
            pop[elim] = repl
        return pop, evals, {"health": health}
