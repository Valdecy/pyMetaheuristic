"""pyMetaheuristic src — Bacterial Chemotaxis Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BCOEngine(PortedPopulationEngine):
    """Bacterial Chemotaxis Optimizer — tumble/swim with personal+global direction blending."""
    algorithm_id   = "bco"
    algorithm_name = "Bacterial Chemotaxis Optimizer"
    family         = "nature"
    _REFERENCE     = {"doi": "10.1109/MCS.2002.1004010"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, c_min=0.01, c_max=0.2,
                     n_chemotaxis=1, max_swim_steps=4)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"local_best": pop.copy()}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim      = pop.shape[0], self.problem.dimension
        T           = max(1, self.config.max_steps or 500)
        t           = state.step + 1
        c_min       = float(self._params.get("c_min", 0.01))
        c_max       = float(self._params.get("c_max", 0.2))
        nc          = int(self._params.get("n_chemotaxis", 1))
        swim_steps  = int(self._params.get("max_swim_steps", 4))

        lb = np.asarray(state.payload.get("local_best", pop.copy()), dtype=float)
        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        step     = c_min + (c_max - c_min) * (1.0 - t / T) ** nc

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            fi  = np.random.random()
            pd  = lb[i, :-1] - pop[i, :-1]
            gd  = best_pos    - pop[i, :-1]
            # Tumble with turbulence
            turb = np.random.normal(0, 0.1, dim)
            move = fi * gd + (1.0 - fi) * pd
            # Swim (refine without turbulence)
            for _ in range(swim_steps):
                move = fi * gd + (1.0 - fi) * pd
            pos = pop[i, :-1] + step * (move + turb)
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i, -1])):
                pop[i] = new_pop[i]
                if self._is_better(float(new_fit[i]), float(lb[i, -1])):
                    lb[i] = new_pop[i]

        # Neighbour exchange
        for i in range(n):
            nb = i + 1 if i < n - 1 else i - 1
            if self._is_better(float(pop[nb, -1]), float(pop[i, -1])):
                pop[i] = pop[nb].copy()

        return pop, n, {"local_best": lb}
