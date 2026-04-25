"""pyMetaheuristic src — Dwarf Mongoose Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class DMOAEngine(PortedPopulationEngine):
    """Dwarf Mongoose Optimization Algorithm — alpha group, scouts and babysitter foraging."""
    algorithm_id   = "dmoa"
    algorithm_name = "Dwarf Mongoose Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.cma.2022.114570"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_baby_sitter=3, peep=2.0)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n  = pop.shape[0]
        L  = int(0.6 * self.problem.dimension * int(self._params.get("n_baby_sitter", 3)))
        return {"C": np.zeros(n, dtype=int), "tau": 0.0, "L": L}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        n_bs        = min(int(self._params.get("n_baby_sitter", 3)), n - 1)
        peep        = float(self._params.get("peep", 2.0))
        C           = np.asarray(state.payload.get("C", np.zeros(n, int)), dtype=int)
        tau         = float(state.payload.get("tau", 0.0))
        L           = int(state.payload.get("L", max(1, int(0.6 * dim * n_bs))))

        CF = (1.0 - t / T) ** (2.0 * t / T)
        fit = pop[:, -1].copy()
        mean_cost = float(np.mean(np.abs(fit))) + 1e-30
        fi = np.exp(-np.abs(fit) / mean_cost)
        fi_sum = fi.sum() + 1e-30
        prob = fi / fi_sum
        evals = 0

        # 1. Alpha group foraging (roulette selection)
        for i in range(n):
            alpha = np.random.choice(n, p=prob)
            k     = np.random.choice([x for x in range(n) if x != i and x != alpha])
            phi   = (peep / 2.0) * np.random.uniform(-1, 1, dim)
            pos   = pop[alpha, :-1] + phi * (pop[alpha, :-1] - pop[k, :-1])
            pos   = np.clip(pos, self._lo, self._hi)
            fit_new = float(self.problem.evaluate(pos)); evals += 1
            if self._is_better(fit_new, float(pop[i, -1])):
                pop[i, :-1] = pos; pop[i, -1] = fit_new
            else:
                C[i] += 1

        # 2. Scout foraging
        SM = np.zeros(n)
        for i in range(n):
            k   = np.random.choice([x for x in range(n) if x != i])
            phi = (peep / 2.0) * np.random.uniform(-1, 1, dim)
            pos = pop[i, :-1] + phi * (pop[i, :-1] - pop[k, :-1])
            pos = np.clip(pos, self._lo, self._hi)
            fit_new = float(self.problem.evaluate(pos)); evals += 1
            denom = max(abs(fit_new), abs(float(pop[i, -1]))) + 1e-30
            SM[i] = (fit_new - float(pop[i, -1])) / denom
            if self._is_better(fit_new, float(pop[i, -1])):
                pop[i, :-1] = pos; pop[i, -1] = fit_new
            else:
                C[i] += 1

        # 3. Baby-sitter eviction
        for i in range(n_bs):
            if C[i] >= L:
                pop[i, :-1] = np.random.uniform(self._lo, self._hi)
                pop[i, -1]  = float(self.problem.evaluate(pop[i, :-1])); evals += 1
                C[i] = 0

        # 4. Next position update
        new_tau = float(np.mean(SM))
        for i in range(n):
            M   = SM[i] * np.ones(dim)  # scalar broadcast
            phi = (peep / 2.0) * np.random.uniform(-1, 1, dim)
            if new_tau > tau:
                pos = pop[i, :-1] - CF * phi * np.random.random() * (pop[i, :-1] - M)
            else:
                pos = pop[i, :-1] + CF * phi * np.random.random() * (pop[i, :-1] - M)
            pop[i, :-1] = np.clip(pos, self._lo, self._hi)
            pop[i, -1]  = float(self.problem.evaluate(pop[i, :-1])); evals += 1

        return pop, evals, {"C": C, "tau": new_tau, "L": L}
