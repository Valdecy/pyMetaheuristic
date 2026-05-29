"""pyMetaheuristic src — Dhole Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, EngineConfig, ProblemSpec
from ._ported_common import PortedPopulationEngine


class DholeOAEngine(PortedPopulationEngine):
    """Dhole Optimization Algorithm.

    Implements the paper's vocalization-driven choice between searching,
    encircling, and attacking, with dynamic pack-member number and prey-size
    logic. The registry id is ``dhole_oa`` because ``doa`` is already used by
    Deer Hunting Optimization in this package.
    """

    algorithm_id = "dhole_oa"
    algorithm_name = "Dhole Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-05005-1",
        "title": "Dhole optimization algorithm: a new metaheuristic algorithm for solving optimization problems",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        c1=1.0,
        l=25.0,
        k=0.5,
        c3=3.0,
        environmental_factor=None,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("population_size must be at least 2 for Dhole Optimization Algorithm.")
        if float(self._params.get("c1", 1.0)) <= 0.0:
            raise ValueError("c1 must be positive.")
        if float(self._params.get("k", 0.5)) <= 0.0:
            raise ValueError("k must be positive.")
        if float(self._params.get("c3", 3.0)) <= 0.0:
            raise ValueError("c3 must be positive.")
        ef = self._params.get("environmental_factor", None)
        if ef is not None and not (0.0 <= float(ef) <= 1.0):
            raise ValueError("environmental_factor must be None or a value in [0, 1].")

    @staticmethod
    def _safe_expit(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + float(np.exp(-x)))

    def _hunting_suitability(self, pack_member_number: int) -> float:
        # Eq. (4). The printed equation places the square over the full term;
        # this also keeps the suitable-time coefficient non-negative.
        c1 = float(self._params.get("c1", 1.0))
        l = float(self._params.get("l", 25.0))
        k = float(self._params.get("k", 0.5))
        ef_param = self._params.get("environmental_factor", None)
        ef = np.random.rand() if ef_param is None else float(ef_param)
        logistic = self._safe_expit(k * (float(pack_member_number) - l))
        return float((c1 * logistic - ef) ** 2)

    def _prey_size(self, individual_fitness: float, prey_fitness: float) -> float:
        # Eq. (10). Fitness values may be negative or zero in generic benchmarks;
        # size is interpreted as a magnitude ratio to keep S non-negative.
        c3 = float(self._params.get("c3", 3.0))
        denom = abs(float(prey_fitness)) + 1.0e-12
        return float(c3 * np.random.rand() * (abs(float(individual_fitness)) / denom))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, state.step + 1)
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 1000)))
        c2 = max(0.0, 1.0 - float(t) / float(max_iter))  # Eq. (7)
        c3 = float(self._params.get("c3", 3.0))

        best_idx = self._best_index(pop[:, -1])
        prey_local = pop[best_idx, :-1].copy()
        prey_global = np.asarray(state.best_position, dtype=float)
        prey = 0.5 * (prey_local + prey_global)  # Eq. (5)
        prey_fitness = float(self.problem.evaluate(prey))
        evals = 1

        pack_member_number = int(np.round(np.random.rand() * 15.0 + 5.0))  # Eq. (3)
        pack_member_number = max(5, min(20, pack_member_number))
        ps = self._hunting_suitability(pack_member_number)

        new_positions = pop[:, :-1].copy()
        operator_labels = ["carryover"] * n
        for i in range(n):
            x = pop[i, :-1]
            vocalization = np.random.rand()

            if vocalization < 0.5:
                if pack_member_number < 10:
                    # Searching stage — Eq. (6).
                    trial = x + c2 * np.random.rand(dim) * (prey - x)
                    operator_labels[i] = "dhole_oa.searching_stage"
                else:
                    # Encircling stage — Eqs. (8)–(9).
                    z = int(self._rand_indices(n, i, 1)[0])
                    trial = x - pop[z, :-1] + prey
                    operator_labels[i] = "dhole_oa.encircling_stage"
            else:
                size = self._prey_size(float(pop[i, -1]), prey_fitness)
                if size > (c3 + 1.0) / 2.0:
                    # Large prey: weaken and repeatedly attack — Eqs. (11)–(12).
                    weak_prey = np.exp(-1.0 / max(size, 1.0e-12)) * prey_local
                    angle = 2.0 * np.pi * np.random.rand(dim)
                    oscillation = np.cos(angle) - np.sin(angle)
                    trial = x + (weak_prey * ps) * oscillation * (weak_prey * ps)
                    operator_labels[i] = "dhole_oa.large_prey_attack"
                else:
                    # Small/weak prey: immediate kill — Eq. (13).
                    trial = (x - prey_global) * ps + ps * np.random.rand(dim) * x
                    operator_labels[i] = "dhole_oa.small_prey_kill"

            new_positions[i] = np.clip(trial, self._lo, self._hi)

        new_fitness = self._evaluate_population(new_positions)
        evals += n
        pop[:, :-1] = new_positions
        pop[:, -1] = new_fitness
        return pop, evals, {"pack_member_number": pack_member_number, "hunting_suitability": ps, "operator_labels": operator_labels}
