"""pyMetaheuristic src — Cheetah Based Optimization (CDDO) Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CDDOEngine(PortedPopulationEngine):
    """Cheetah Based Optimization — learning/skill rate drawing-inspired update."""
    algorithm_id   = "cddo"
    algorithm_name = "Cheetah Based Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1038/s41598-022-14338-z"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pattern_size=10, creativity_rate=0.1)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        gr = np.random.uniform(1.5, 2.0, pop.shape[0])  # growth rate per individual
        return {"local_best": pop.copy(), "gr": gr,
                "LR": np.random.uniform(0.1, 1.0),
                "SR": np.random.uniform(0.1, 1.0)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        pat_size    = min(int(self._params.get("pattern_size", 10)), n)
        cr          = float(self._params.get("creativity_rate", 0.1))

        lb   = np.asarray(state.payload.get("local_best", pop.copy()), dtype=float)
        gr   = np.asarray(state.payload.get("gr", np.ones(n) * 1.75), dtype=float)
        LR   = float(state.payload.get("LR", 0.5))
        SR   = float(state.payload.get("SR", 0.5))

        order   = self._order(pop[:, -1])
        pattern = pop[order[:pat_size], :-1]   # top-k as pattern
        best_pos= pop[order[0], :-1].copy()

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            hp = np.random.uniform(self._lo[0], self._hi[0])
            pp = np.random.randint(dim)
            if pop[i, pp] <= hp:
                pos = (gr[i] + SR * np.random.random(dim) * (lb[i, :-1] - pop[i, :-1])
                       + LR * np.random.random(dim) * (best_pos - pop[i, :-1]))
                LR = np.random.uniform(0.6, 1.1)
                SR = np.random.uniform(0.6, 1.1)
            elif 1.5 < gr[i] < 2.0:
                pat_choice = pattern[np.random.randint(pat_size)]
                pos = pat_choice - cr * lb[i, :-1]
                LR = np.random.uniform(0.0, 0.6)
                SR = np.random.uniform(0.0, 0.6)
            else:
                pos = pop[i, :-1] + np.random.normal(0, 1, dim) * 0.1 * self._span
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        # Update local bests
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(lb[i, -1])):
                lb[i] = new_pop[i]
        mask = self._better_mask(new_fit, pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, n, {"local_best": lb, "gr": gr, "LR": LR, "SR": SR}
