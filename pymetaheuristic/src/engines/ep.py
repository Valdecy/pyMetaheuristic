"""pyMetaheuristic src — Evolutionary Programming Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EPEngine(PortedPopulationEngine):
    """Evolutionary Programming — self-adaptive strategy mutation with tournament selection."""
    algorithm_id   = "ep"
    algorithm_name = "Evolutionary Programming"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, bout_size=0.05)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        # Each individual carries a strategy vector (step sizes)
        strategies = np.abs(np.random.normal(0, 1, (pop.shape[0], self.problem.dimension))) + 1e-6
        return {"strategies": strategies}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        bout_size   = max(1, int(self._params.get("bout_size", 0.05) * n))
        strategies  = np.asarray(state.payload.get("strategies",
                      np.ones((n, dim)) * 0.1), dtype=float)

        # Generate offspring
        off_pos = np.empty_like(pop[:, :-1])
        off_strat = np.empty_like(strategies)
        for i in range(n):
            off_pos[i]   = np.clip(pop[i, :-1] + strategies[i] * np.random.normal(0, 1, dim),
                                   self._lo, self._hi)
            off_strat[i] = np.abs(strategies[i] + np.random.normal(0, 1, dim) *
                                  np.sqrt(np.abs(strategies[i]))) + 1e-6

        off_fit = self._evaluate_population(off_pos)

        # Merge parent + offspring
        all_pos   = np.vstack([pop[:, :-1], off_pos])
        all_fit   = np.concatenate([pop[:, -1], off_fit])
        all_strat = np.vstack([strategies, off_strat])

        # Tournament selection: count wins
        total = 2 * n
        wins  = np.zeros(total, dtype=int)
        for i in range(total):
            opponents = np.random.choice(total, bout_size, replace=False)
            for opp in opponents:
                if self._is_better(float(all_fit[i]), float(all_fit[opp])):
                    wins[i] += 1
                else:
                    wins[opp] += 1

        # Keep top-n by wins
        top_idx   = np.argsort(-wins)[:n]
        new_pop   = np.hstack([all_pos[top_idx], all_fit[top_idx, None]])
        new_strat = all_strat[top_idx]
        return new_pop, n, {"strategies": new_strat}
