"""pyMetaheuristic src — MadDE Engine.

Faithful native NumPy port of the Multiple Adaptation Differential Evolution
(MadDE) algorithm of Biswas et al. (CEC 2021).  It combines three adaptive
mutation strategies, probabilistic q-best/binomial crossover, SHADE-style
success memories, an external archive, and linear population-size reduction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

_EPS = 1.0e-30
_TERMINAL_CR = -1.0


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.5
    total = float(np.sum(weights))
    weights = weights / (total + _EPS)
    denominator = float(np.sum(weights * values))
    if abs(denominator) <= _EPS or not np.isfinite(denominator):
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))
    return float(np.clip(np.sum(weights * values * values) / denominator, 0.0, 1.0))


class MadDEEngine(PortedPopulationEngine):
    """Multiple Adaptation Differential Evolution (MadDE)."""

    algorithm_id = "madde"
    algorithm_name = "Multiple Adaptation Differential Evolution"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC45853.2021.9504725",
        "title": "Improving Differential Evolution through Bayesian Hyperparameter Optimization",
        "authors": "Subhodip Biswas, Debanjan Saha, Shuvodeep De, Adam D. Cobb, Swagatam Das, and Brian A. Jalaian",
        "year": 2021,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        population_size=None,
        initial_population_multiplier=2.0,
        min_population_size=4,
        qbx_probability=0.01,
        pbest_rate=0.18,
        archive_rate=2.30,
        memory_multiplier=10.0,
        memory_size=None,
        memory_f_init=0.20,
        memory_cr_init=0.20,
        cauchy_scale=0.10,
        cr_sigma=0.10,
        sobol_initialization=True,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size")
        if pop_param in (None, 0, "paper", "auto"):
            multiplier = float(self._params.get("initial_population_multiplier", 2.0))
            self._n = max(4, int(round(multiplier * self.problem.dimension ** 2)))
            self._params["population_size"] = self._n
        else:
            self._n = max(4, int(pop_param))
            self._params["population_size"] = self._n
        mem_param = self._params.get("memory_size")
        if mem_param in (None, 0, "paper", "auto"):
            self._params["memory_size"] = max(
                1, int(round(float(self._params.get("memory_multiplier", 10.0)) * self.problem.dimension))
            )
        else:
            self._params["memory_size"] = max(1, int(mem_param))
        self._validate_parameters()
        self._operator_labels = [
            "madde.parameter_sampling",
            "madde.current_to_pbest_archive_mutation",
            "madde.current_to_rand_archive_mutation",
            "madde.weighted_rand_to_qbest_mutation",
            "madde.midpoint_bound_repair",
            "madde.binomial_crossover",
            "madde.qbest_binomial_crossover",
            "madde.greedy_selection",
            "madde.external_archive_update",
            "madde.success_history_update",
            "madde.mutation_probability_adaptation",
            "madde.linear_population_size_reduction",
        ]
        self._last_operator_contributions = self._blank_contribs()
        self._last_operator_counts = self._blank_counts()

    def _validate_parameters(self) -> None:
        for key in ("qbx_probability", "pbest_rate"):
            value = float(self._params[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must lie in [0, 1].")
        if float(self._params["archive_rate"]) < 0.0:
            raise ValueError("archive_rate must be non-negative.")
        if int(self._params["min_population_size"]) < 4:
            raise ValueError("min_population_size must be at least 4.")

    def _new_positions(self, n: int | None = None) -> np.ndarray:
        count = self._n if n is None else int(n)
        if bool(self._params.get("sobol_initialization", True)):
            try:
                from scipy.stats import qmc

                sampler = qmc.Sobol(
                    d=self.problem.dimension,
                    scramble=True,
                    seed=self.config.seed,
                )
                unit = sampler.random(count)
                return self._lo + unit * self._span
            except Exception:
                pass
        return super()._new_positions(count)

    def _blank_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._operator_labels}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._operator_labels}

    def _sample_f(self, center: float) -> float:
        scale = float(self._params.get("cauchy_scale", 0.1))
        while True:
            value = float(center + scale * np.tan(np.pi * (np.random.random() - 0.5)))
            if value > 0.0:
                return min(value, 1.0)

    def _sample_cr(self, center: float) -> float:
        if float(center) < 0.0:
            return 0.0
        return float(np.clip(np.random.normal(center, float(self._params.get("cr_sigma", 0.1))), 0.0, 1.0))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, int]:
        donor = np.asarray(donor, dtype=float).copy()
        below = donor < self._lo
        above = donor > self._hi
        count = int(np.count_nonzero(below) + np.count_nonzero(above))
        donor[below] = 0.5 * (parent[below] + self._lo[below])
        donor[above] = 0.5 * (parent[above] + self._hi[above])
        return donor, count

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = int(self._params["memory_size"])
        return {
            "M_F": np.full(h, float(self._params["memory_f_init"]), dtype=float),
            "M_CR": np.full(h, float(self._params["memory_cr_init"]), dtype=float),
            "k": 0,
            "archive": np.empty((0, self.problem.dimension + 1), dtype=float),
            "initial_n": int(pop.shape[0]),
            "mutation_probabilities": np.full(3, 1.0 / 3.0, dtype=float),
        }

    def _progress(self, state) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return float(np.clip(state.evaluations / self.config.max_evaluations, 0.0, 1.0))
        horizon = max(1, int(self.config.max_steps or 100))
        return float(np.clip(state.step / horizon, 0.0, 1.0))

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, step: int) -> int:
        minimum = max(4, int(self._params["min_population_size"]))
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            progress = np.clip(evaluations_after_step / self.config.max_evaluations, 0.0, 1.0)
        else:
            progress = np.clip((step + 1) / max(1, int(self.config.max_steps or 100)), 0.0, 1.0)
        return max(minimum, int(round(initial_n + (minimum - initial_n) * progress)))

    def _select_population_index(self, n: int, excluded: set[int]) -> int:
        candidates = np.asarray([j for j in range(n) if j not in excluded], dtype=int)
        if candidates.size == 0:
            candidates = np.arange(n, dtype=int)
        return int(np.random.choice(candidates))

    def _select_combined_index(self, n: int, archive_n: int, excluded_population: set[int]) -> int:
        total = n + archive_n
        candidates = [j for j in range(total) if not (j < n and j in excluded_population)]
        if not candidates:
            candidates = list(range(total))
        return int(np.random.choice(candidates))

    def _step_impl(self, state, pop):
        n, dim = int(pop.shape[0]), self.problem.dimension
        if n < 4:
            raise ValueError("MadDE requires at least four population members.")

        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        archive = np.asarray(state.payload.get("archive"), dtype=float).reshape(-1, dim + 1)
        combined = np.vstack((pop, archive)) if archive.size else pop.copy()
        combined_order = self._order(combined[:, -1])
        M_F = np.asarray(state.payload["M_F"], dtype=float).copy()
        M_CR = np.asarray(state.payload["M_CR"], dtype=float).copy()
        k = int(state.payload.get("k", 0)) % M_F.size
        initial_n = int(state.payload.get("initial_n", n))
        probabilities = np.asarray(state.payload.get("mutation_probabilities", np.full(3, 1 / 3)), dtype=float)
        progress = self._progress(state)
        p_rate = float(self._params["pbest_rate"])
        q_rate = float(np.clip(2.0 * p_rate - p_rate * progress, _EPS, 1.0))
        attraction = 0.5 + 0.5 * progress
        pnum = min(n, max(2, int(round(p_rate * n))))
        qnum = min(n, max(2, int(round(q_rate * n))))
        qcombined_num = min(combined.shape[0], max(2, int(round(q_rate * combined.shape[0]))))

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        trials = np.empty((n, dim), dtype=float)
        Fs = np.empty(n, dtype=float)
        CRs = np.empty(n, dtype=float)
        strategy = np.empty(n, dtype=int)

        for i in range(n):
            memory_index = int(np.random.randint(M_F.size))
            F = self._sample_f(float(M_F[memory_index]))
            CR = self._sample_cr(float(M_CR[memory_index]))
            Fs[i], CRs[i] = F, CR
            m = int(np.random.choice(3, p=probabilities / np.sum(probabilities)))
            strategy[i] = m
            parent = pop[i, :-1]
            r1 = self._select_population_index(n, {i})
            r2_combined = self._select_combined_index(n, archive.shape[0], {i, r1})
            r3 = self._select_population_index(n, {i, r1})

            if m == 0:
                pbest = pop[int(np.random.randint(pnum)), :-1]
                donor = parent + F * (pbest - parent + pop[r1, :-1] - combined[r2_combined, :-1])
                label = "madde.current_to_pbest_archive_mutation"
            elif m == 1:
                donor = parent + F * (pop[r1, :-1] - combined[r2_combined, :-1])
                label = "madde.current_to_rand_archive_mutation"
            else:
                qbest = pop[int(np.random.randint(qnum)), :-1]
                donor = F * (pop[r1, :-1] + attraction * (qbest - pop[r3, :-1]))
                label = "madde.weighted_rand_to_qbest_mutation"
            counts[label] += 1

            donor, repaired = self._repair_bound(donor, parent)
            counts["madde.midpoint_bound_repair"] += repaired
            use_qbx = bool(np.random.random() <= float(self._params["qbx_probability"]))
            base = parent
            crossover_label = "madde.binomial_crossover"
            if use_qbx:
                base = combined[combined_order[int(np.random.randint(qcombined_num))], :-1]
                crossover_label = "madde.qbest_binomial_crossover"
            mask = np.random.random(dim) <= CR
            mask[int(np.random.randint(dim))] = True
            trials[i] = np.where(mask, donor, base)
            counts[crossover_label] += 1
            counts["madde.parameter_sampling"] += 1

        trial_pop = self._pop_from_positions(trials)
        strict = self._better_mask(trial_pop[:, -1], pop[:, -1])
        survive = strict | np.isclose(trial_pop[:, -1], pop[:, -1], rtol=0.0, atol=0.0)
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[strict] = pop[strict, -1] - trial_pop[strict, -1]
        else:
            gains[strict] = trial_pop[strict, -1] - pop[strict, -1]
        gains = np.maximum(gains, 0.0)
        counts["madde.greedy_selection"] = int(np.count_nonzero(survive))

        for m, suffix in enumerate((
            "current_to_pbest_archive_mutation",
            "current_to_rand_archive_mutation",
            "weighted_rand_to_qbest_mutation",
        )):
            mask = strict & (strategy == m)
            if np.any(mask):
                share = float(np.sum(gains[mask])) / 3.0
                contrib[f"madde.{suffix}"] += share
        accepted_gain = float(np.sum(gains))
        if accepted_gain > 0.0:
            bx_count = max(1, counts["madde.binomial_crossover"] + counts["madde.qbest_binomial_crossover"])
            contrib["madde.binomial_crossover"] += accepted_gain / 3.0 * counts["madde.binomial_crossover"] / bx_count
            contrib["madde.qbest_binomial_crossover"] += accepted_gain / 3.0 * counts["madde.qbest_binomial_crossover"] / bx_count
            contrib["madde.greedy_selection"] += accepted_gain / 3.0

        if np.any(strict):
            archived = pop[strict].copy()
            archive = np.vstack((archive, archived)) if archive.size else archived
            counts["madde.external_archive_update"] = int(archived.shape[0])
            diffs = np.abs(pop[strict, -1] - trial_pop[strict, -1])
            weights = diffs / (float(np.sum(diffs)) + _EPS)
            good_f = Fs[strict]
            good_cr = CRs[strict]
            M_F[k] = _weighted_lehmer(good_f, weights)
            if M_CR[k] < 0.0 or float(np.max(good_cr)) == 0.0:
                M_CR[k] = _TERMINAL_CR
            else:
                M_CR[k] = _weighted_lehmer(good_cr, weights)
            k = (k + 1) % M_F.size
            counts["madde.success_history_update"] = 1
        else:
            M_F[k] = 0.5
            M_CR[k] = 0.5
            counts["madde.success_history_update"] = 1

        scores = np.zeros(3, dtype=float)
        denominator = np.maximum(np.abs(pop[:, -1]), _EPS)
        relative_improvement = gains / denominator
        for m in range(3):
            mask = strategy == m
            scores[m] = float(np.mean(relative_improvement[mask])) if np.any(mask) else 0.0
        if float(np.sum(scores)) > 0.0 and np.all(np.isfinite(scores)):
            probabilities = np.clip(scores / np.sum(scores), 0.1, 0.9)
            probabilities = probabilities / np.sum(probabilities)
        else:
            probabilities[:] = 1.0 / 3.0
        counts["madde.mutation_probability_adaptation"] = 1

        pop[survive] = trial_pop[survive]
        target_n = self._target_population_size(state.evaluations + n, initial_n, state.step)
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            counts["madde.linear_population_size_reduction"] = int(pop.shape[0] - target_n)
            pop = pop[keep]

        max_archive = max(0, int(round(float(self._params["archive_rate"]) * pop.shape[0])))
        if archive.shape[0] > max_archive:
            if max_archive == 0:
                archive = np.empty((0, dim + 1), dtype=float)
            else:
                archive = archive[np.random.choice(archive.shape[0], max_archive, replace=False)]

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        return pop, n, {
            "M_F": M_F,
            "M_CR": M_CR,
            "k": k,
            "archive": archive,
            "initial_n": initial_n,
            "mutation_probabilities": probabilities,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        obs["mutation_probabilities"] = np.asarray(
            state.payload.get("mutation_probabilities", np.full(3, 1.0 / 3.0)), dtype=float
        ).tolist()
        return obs
