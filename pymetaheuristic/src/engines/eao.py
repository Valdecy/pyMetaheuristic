"""pyMetaheuristic src — Enzyme Activity Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EAOEngine(PortedPopulationEngine):
    """Enzyme Activity Optimizer — three substrate candidates with adaptation factor."""
    algorithm_id   = "eao"
    algorithm_name = "Enzyme Activity Optimizer"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1007/s11227-025-07052-w"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, ec=0.1)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        ec      = float(self._params.get("ec", 0.1))
        AF      = np.sqrt(t / T)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        evals    = 0

        for i in range(n):
            others = [k for k in range(n) if k != i]
            j1, j2 = np.random.choice(others, 2, replace=False)

            # Candidate 1: sinusoidal attraction to best
            r1   = np.random.random(dim)
            pos1 = (best_pos - pop[i, :-1]) + r1 * np.sin(AF * pop[i, :-1])
            pos1 = np.clip(pos1, self._lo, self._hi)
            fit1 = float(self.problem.evaluate(pos1)); evals += 1

            # Candidate A: vector-scaled differential
            scA  = ec + (1.0 - ec) * np.random.random(dim)
            exA  = AF * (ec + (1.0 - ec) * np.random.random(dim))
            posA = pop[i, :-1] + scA * (pop[j1, :-1] - pop[j2, :-1]) + exA * (best_pos - pop[i, :-1])
            posA = np.clip(posA, self._lo, self._hi)
            fitA = float(self.problem.evaluate(posA)); evals += 1

            # Candidate B: scalar-scaled differential
            scB  = ec + (1.0 - ec) * np.random.random()
            exB  = AF * (ec + (1.0 - ec) * np.random.random())
            posB = pop[i, :-1] + scB * (pop[j1, :-1] - pop[j2, :-1]) + exB * (best_pos - pop[i, :-1])
            posB = np.clip(posB, self._lo, self._hi)
            fitB = float(self.problem.evaluate(posB)); evals += 1

            # Keep best of 4 (current + 3 candidates)
            candidates = [(float(pop[i, -1]), pop[i, :-1]),
                          (fit1, pos1), (fitA, posA), (fitB, posB)]
            best_cand  = min(candidates, key=lambda x: x[0]) if self.problem.objective == "min" \
                         else max(candidates, key=lambda x: x[0])
            pop[i, :-1] = best_cand[1]; pop[i, -1] = best_cand[0]

        return pop, evals, {}
