"""pyMetaheuristic src — Fox Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FOXEngine(PortedPopulationEngine):
    """Fox Optimizer — jump-distance hunting model of foxes."""
    algorithm_id   = "fox"
    algorithm_name = "Fox Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10489-022-03533-0"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, c1=0.18, c2=0.82)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"mint": 1e7}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        c1      = float(self._params.get("c1", 0.18))
        c2      = float(self._params.get("c2", 0.82))
        mint    = float(state.payload.get("mint", 1e7))

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        aa       = 2.0 * (1.0 - 1.0 / T)

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            if np.random.random() >= 0.5:
                t1   = np.random.random(dim)
                sps  = best_pos / (t1 + 1e-30)
                dis  = 0.5 * sps * t1
                tt   = float(np.mean(t1))
                t_   = tt / 2.0
                jump = 0.5 * 9.81 * t_ ** 2
                if np.random.random() > 0.18:
                    pos = dis * jump * c1
                else:
                    pos = dis * jump * c2
                if mint > tt:
                    mint = tt
            else:
                pos = best_pos * np.random.random(dim) * (mint * aa)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        pop     = np.hstack([new_pos, new_fit[:, None]])   # replaces all (original)
        return pop, n, {"mint": mint}
