"""pyMetaheuristic src — Crayfish Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class CrayfishOAEngine(PortedPopulationEngine):
    """Crayfish Optimization Algorithm (COA).

    Native NumPy port of the MATLAB reference algorithm: summer-resort,
    competition, and foraging stages driven by the adaptive coefficient C and
    temperature term.
    """

    algorithm_id = "crayfish_oa"
    algorithm_name = "Crayfish Optimization Algorithm"
    family = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10462-023-10567-4"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    @staticmethod
    def _p_obj(temp: float) -> float:
        return float(0.2 * (1.0 / (np.sqrt(2.0 * np.pi) * 3.0)) * np.exp(-((temp - 25.0) ** 2) / (2.0 * 3.0 ** 2)))

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        idx = self._best_index(pop[:, -1])
        return {"global_position": pop[idx, :-1].copy(), "global_fitness": float(pop[idx, -1])}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 1000)))
        C = 2.0 - (t / max_iter)
        temp = float(np.random.random() * 15.0 + 20.0)
        best_pos = np.asarray(state.best_position, dtype=float)
        global_pos = np.asarray(state.payload.get("global_position", best_pos), dtype=float)
        xf = 0.5 * (best_pos + global_pos)
        xfood = best_pos.copy()
        eps = 1e-12

        trial = pop[:, :-1].copy()
        for i in range(n):
            if temp > 30.0:
                if np.random.random() < 0.5:
                    trial[i] = pop[i, :-1] + C * np.random.random(dim) * (xf - pop[i, :-1])
                else:
                    z = np.random.randint(0, n, size=dim)
                    trial[i] = pop[i, :-1] - pop[z, np.arange(dim)] + xf
            else:
                denom = abs(float(state.best_fitness)) + eps
                P = 3.0 * np.random.random() * (abs(float(pop[i, -1])) + eps) / denom
                p = self._p_obj(temp)
                if P > 2.0:
                    xfood = np.exp(-1.0 / max(P, eps)) * xfood
                    theta = 2.0 * np.pi * np.random.random(dim)
                    trial[i] = pop[i, :-1] + (np.cos(theta) - np.sin(theta)) * xfood * p
                else:
                    trial[i] = (pop[i, :-1] - xfood) * p + p * np.random.random(dim) * pop[i, :-1]

        trial = np.clip(trial, self._lo, self._hi)
        fit = self._evaluate_population(trial)
        evals = n

        new_best_idx = self._best_index(fit)
        global_position = trial[new_best_idx].copy()
        global_fitness = float(fit[new_best_idx])

        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1] = trial[mask]
        pop[mask, -1] = fit[mask]

        return pop, evals, {"global_position": global_position, "global_fitness": global_fitness}
