"""pyMetaheuristic src — Rock Hyraxes Swarm Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class RHSOEngine(PortedPopulationEngine):
    """
    Rock Hyraxes Swarm Optimization.

    """

    algorithm_id = "rhso"
    algorithm_name = "Rock Hyraxes Swarm Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.32604/cmc.2021.013648",
        "title": "Rock Hyraxes Swarm Optimization: A New Nature-Inspired Metaheuristic Optimization Algorithm",
        "authors": "Belal Al-Khateeb, Kawther Ahmed, Maha Mahmood, Dac-Nhuong Le",
        "year": 2021,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("rhso requires population_size >= 2.")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        dim = self.problem.dimension
        return {"angle": np.random.uniform(0.0, 360.0, dim)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        angle = np.asarray(state.payload.get("angle"), dtype=float).copy()
        if angle.shape != (dim,):
            angle = np.random.uniform(0.0, 360.0, dim)

        best_idx = self._best_index(pop[:, -1])
        leader_old = pop[best_idx, :-1].copy()
        leader_new = np.clip(np.random.rand(dim) * leader_old, lo, hi)

        positions = pop[:, :-1].copy()
        r2 = np.random.rand(n, dim)
        circ = np.sqrt((r2 * np.cos(np.deg2rad(angle))) ** 2 + (r2 * np.sin(np.deg2rad(angle))) ** 2)
        positions = positions - (circ * positions + leader_new)
        positions = np.clip(positions, lo, hi)
        positions[best_idx] = leader_new

        fit = self._evaluate_population(positions)
        new_pop = np.hstack((positions, fit[:, None]))

        delta = np.random.uniform(lo, hi)
        angle = np.clip(angle + delta, 0.0, 360.0)

        return new_pop, n, {"angle": angle}
