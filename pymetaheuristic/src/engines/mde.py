"""pyMetaheuristic src — Multi-population based Differential Evolution Engine.

Native NumPy port of MDE (Karkinli, 2023), contributed by the algorithm's
author and checked against the original MATLAB implementation
(https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm).

MDE keeps a full population ``A`` of ``3N`` vectors. In every macro-step ``N``
distinct vectors are drawn at random from ``A`` to form a sub-population
``B``. Trial vectors are built with a random scale factor, random ``dx``/``dy``
difference vectors around the base ``B[r2]``, a parameter-free random
crossover map and a tiny noise term carried over from the previous step.
Out-of-bounds variables are re-sampled uniformly inside the box (Eq. 15),
improved trials replace their parents (greedy selection), and ``B`` is
written back to ``A``. The algorithm has no control parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile, EngineState, OptimizationResult
from ._ported_common import PortedPopulationEngine

_N_SUBPOPULATIONS = 3


class MDEEngine(PortedPopulationEngine):
    """Multi-population based Differential Evolution (MDE), a parameter-free DE variant.

    Parameters (``config.params``)
    ------------------------------
    population_size : int, default 90
        Size of the full population ``A`` (``3N`` vectors, must be >= 9).
        The sub-population size is ``N = population_size // 3``; when
        ``population_size`` is not a multiple of 3, ``3N`` vectors are used.

    Budget semantics
    ----------------
    Initialization evaluates the ``3N`` vectors of ``A``. One native step
    (``max_steps`` unit) evaluates the ``N`` trial vectors of one
    sub-population, i.e. ``population_size // 3`` function evaluations.
    """

    algorithm_id = "mde"
    algorithm_name = "Multi-population based Differential Evolution"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1007/s00521-022-07969-w",
        "authors": "Ahmet Emin Karkinli",
        "title": "Detection of object boundary from point cloud by using multi-population based differential evolution algorithm",
        "journal": "Neural Computing and Applications, 35, 5193-5206",
        "year": 2023,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(population_size=90)

    _OPERATOR_LABELS = (
        "mde.subpopulation_sampling",
        "mde.random_scale_differential_mutation",
        "mde.parameter_free_crossover",
        "mde.noise_perturbation",
        "mde.random_bound_repair",
        "mde.greedy_selection",
        "mde.subpopulation_write_back",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        size = self._params.get("population_size")
        if size in (None, 0, "paper", "auto"):
            size = self._DEFAULTS["population_size"]
        self._N = int(size) // _N_SUBPOPULATIONS
        if self._N < 3:
            raise ValueError("mde population_size must be >= 9 (full population of 3*N vectors, N >= 3).")
        self._n = _N_SUBPOPULATIONS * self._N
        self._params["population_size"] = self._n
        self._last_operator_counts = {label: 0 for label in self._OPERATOR_LABELS}
        self._last_operator_contributions = {label: 0.0 for label in self._OPERATOR_LABELS}
        self._last_crossover_mode = "none"
        self._last_acceptance_rate = 0.0
        self._last_bound_repairs = 0

    # ------------------------------------------------------------------
    # Initialization: full population A of 3N vectors
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        dim = self.problem.dimension
        positions = np.random.uniform(self._lo, self._hi, (self._n, dim))
        pop = self._pop_from_positions(positions)
        idx = self._best_index(pop[:, -1])
        return EngineState(
            step=0,
            evaluations=int(pop.shape[0]),
            best_position=pop[idx, :-1].tolist(),
            best_fitness=float(pop[idx, -1]),
            initialized=True,
            payload={
                "population": pop,
                "noise": np.zeros((self._N, dim), dtype=float),
            },
        )

    # ------------------------------------------------------------------
    # One native macro-step: evolve one random sub-population
    # ------------------------------------------------------------------

    @staticmethod
    def _scale() -> float:
        # scale = |a - rand^b| * randn^d,  a in {0, 1}, b in {1..10}, d in {1..5}
        return abs(np.random.randint(0, 2) - np.random.rand() ** np.random.randint(1, 11)) * (
            np.random.randn() ** np.random.randint(1, 6)
        )

    def _step_impl(self, state, pop):
        N, dim = self._N, self.problem.dimension
        lo, hi = self._lo, self._hi
        noise = np.asarray(state.payload.get("noise", np.zeros((N, dim))), dtype=float)
        if noise.shape != (N, dim):
            noise = np.zeros((N, dim), dtype=float)

        counts = {label: 0 for label in self._OPERATOR_LABELS}
        contrib = {label: 0.0 for label in self._OPERATOR_LABELS}

        best = pop[self._best_index(pop[:, -1]), :-1].copy()

        # random sub-population B of N distinct members of A
        j0 = np.random.permutation(pop.shape[0])[:N]
        B = pop[j0, :-1].copy()
        fitB = pop[j0, -1].copy()
        counts["mde.subpopulation_sampling"] = N

        # mutation: base B[r2], difference dx - dy with random components
        temp = np.empty((N, dim), dtype=float)
        for i in range(N):
            scale = self._scale()
            while True:
                r1, r2 = np.random.choice(N, 2, replace=False)
                if r1 != i and r2 != i:
                    break
            dx = np.where(np.random.rand(dim) < 0.5, B[r1], best)
            dy = np.where(np.random.rand(dim) < 0.5, B[i], B[r2])
            temp[i] = B[r2] + scale * (dx - dy)
        counts["mde.random_scale_differential_mutation"] = N

        # parameter-free crossover map: c = 1 (one map per vector) or c = dim
        c = 1 if np.random.rand() ** np.random.randint(1, 6) < 0.5 else dim
        cmap = np.abs(
            np.random.randint(0, 2, (N, c)) - np.random.rand(N, dim) ** np.random.randint(1, 6, (N, c))
        ) < 0.5
        trials = B + cmap * (temp + noise - B)
        counts["mde.parameter_free_crossover"] = N
        counts["mde.noise_perturbation"] = N

        # Eq. 15: re-sample out-of-bounds variables uniformly in [lo, hi]
        out = (trials < lo) | (trials > hi)
        bound_repairs = int(np.count_nonzero(np.any(out, axis=1)))
        if out.any():
            trials = np.where(out, lo + np.random.rand(N, dim) * (hi - lo), trials)
        counts["mde.random_bound_repair"] = bound_repairs

        # greedy selection
        fitT = self._evaluate_population(trials)
        better = self._better_mask(fitT, fitB)
        gains = np.abs(fitB[better] - fitT[better])
        B[better] = trials[better]
        fitB[better] = fitT[better]
        counts["mde.greedy_selection"] = int(np.count_nonzero(better))
        accepted_gain = float(np.sum(gains)) if gains.size else 0.0
        if accepted_gain > 0.0 and np.isfinite(accepted_gain):
            share = accepted_gain / 3.0
            contrib["mde.random_scale_differential_mutation"] += share
            contrib["mde.parameter_free_crossover"] += share
            contrib["mde.greedy_selection"] += share

        # write the sub-population back to the full population
        pop[j0, :-1] = B
        pop[j0, -1] = fitB
        counts["mde.subpopulation_write_back"] = N

        # noise for the next step (Eq. 13): B .* 10^phi .* (rand(N,1) - 0.5), phi in {-12..-9}
        new_noise = B * 10.0 ** np.random.randint(-12, -8, (N, dim)) * (np.random.rand(N, 1) - 0.5)

        self._last_operator_counts = counts
        self._last_operator_contributions = contrib
        self._last_crossover_mode = "per_vector" if c == 1 else "per_variable"
        self._last_acceptance_rate = float(np.count_nonzero(better) / N)
        self._last_bound_repairs = bound_repairs
        return pop, N, {"noise": new_noise, "subpopulation_indices": j0}

    # ------------------------------------------------------------------
    # Telemetry / result
    # ------------------------------------------------------------------

    def observe(self, state) -> dict[str, Any]:
        obs = super().observe(state)
        active = [label for label, count in self._last_operator_counts.items() if int(count) > 0]
        obs.update(
            population_size=int(self._n),
            subpopulation_size=int(self._N),
            crossover_mode=self._last_crossover_mode,
            acceptance_rate=float(self._last_acceptance_rate),
            bound_repairs=int(self._last_bound_repairs),
            operator_contributions=dict(self._last_operator_contributions),
            operator_counts=dict(self._last_operator_counts),
            evomapx_operator_labels=active or list(self._OPERATOR_LABELS),
            evomapx_delta_f="improvement_positive",
            evomapx_fidelity="native",
        )
        return obs

    def finalize(self, state) -> OptimizationResult:
        result = super().finalize(state)
        result.metadata.update(
            population_size=int(self._n),
            subpopulation_size=int(self._N),
            n_subpopulations=_N_SUBPOPULATIONS,
            evaluations_per_step=int(self._N),
            bounds_policy="uniform_resample_eq15",
            evomapx_operator_labels=list(self._OPERATOR_LABELS),
        )
        return result
