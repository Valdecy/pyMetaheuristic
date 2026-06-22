"""pyMetaheuristic src — L-SRTDE Engine.

Linear population size reduction Success-Rate-based Differential Evolution
(L-SRTDE), after Stanovov and Semenkin's CEC 2024 algorithm.  The port keeps
its two-population structure: a newest population used as replacement target
and a top population used as the elite reservoir.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


_EPS = 1.0e-30


class LSRTDEEngine(PortedPopulationEngine):
    """Linear population size reduction Success-Rate-based Differential Evolution."""

    algorithm_id = "l_srtde"
    algorithm_name = "L-SRTDE"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC60901.2024.10611907",
        "title": "Success Rate-based Adaptive Differential Evolution L-SRTDE for CEC 2024 Competition",
        "authors": "Vladimir Stanovov and Eugene Semenkin",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        PortedPopulationEngine._DEFAULTS,
        # Paper setting: NPmax = 20D.  None/0/"paper"/"auto" activates it.
        population_size=None,
        min_population_size=4,
        hist_mem_size=5,
        memory_cr_init=1.0,
        initial_success_rate=0.5,
        f_base=0.4,
        f_amplitude=0.25,
        f_success_slope=5.0,
        f_sigma=0.02,
        cr_sigma=0.05,
        cr_memory_mixing=0.5,
        pbest_base=0.7,
        pbest_success_slope=7.0,
        rank_pressure_coefficient=3.0,
        resampling_attempts=100,
        bound_handling="resample",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        pop_param = self._params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto"):
            self._n = max(4, 20 * int(self.problem.dimension))
            self._params["population_size"] = self._n
        else:
            self._n = max(4, int(pop_param))
            self._params["population_size"] = self._n
        self._operator_labels = self._make_operator_labels()
        self._last_operator_contributions = {label: 0.0 for label in self._operator_labels}
        self._last_operator_counts = {label: 0 for label in self._operator_labels}

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def _make_operator_labels(self) -> list[str]:
        return [
            f"{self.algorithm_id}.success_rate_f_adaptation",
            f"{self.algorithm_id}.success_rate_pbest_control",
            f"{self.algorithm_id}.rank_selective_pressure",
            f"{self.algorithm_id}.r_new_to_ptop_mutation",
            f"{self.algorithm_id}.binomial_crossover",
            f"{self.algorithm_id}.bound_resampling",
            f"{self.algorithm_id}.selection",
            f"{self.algorithm_id}.newest_population_update",
            f"{self.algorithm_id}.top_population_update",
            f"{self.algorithm_id}.crossover_memory_update",
            f"{self.algorithm_id}.linear_population_reduction",
        ]

    def _blank_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._operator_labels}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._operator_labels}

    def _add(self, contrib: dict[str, float], counts: dict[str, int], suffix: str, value: float = 0.0, count: int = 0) -> None:
        label = f"{self.algorithm_id}.{suffix}"
        if label in contrib:
            contrib[label] += float(value)
            counts[label] += int(count)

    # ------------------------------------------------------------------
    # Core L-SRTDE primitives
    # ------------------------------------------------------------------
    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        h = max(1, int(self._params.get("hist_mem_size", 5)))
        order = self._order(pop[:, -1])
        top = pop[order].copy()
        return {
            "new_population": top.copy(),
            "M_CR": np.full(h, float(self._params.get("memory_cr_init", 1.0)), dtype=float),
            "memory_index": 0,
            "success_rate": float(self._params.get("initial_success_rate", 0.5)),
            "initial_n": int(top.shape[0]),
            "pf_index": 0,
        }

    def _gain(self, old_fit: float, new_fit: float) -> float:
        old_fit = float(old_fit)
        new_fit = float(new_fit)
        return max(0.0, old_fit - new_fit) if self.problem.objective == "min" else max(0.0, new_fit - old_fit)

    def _not_worse(self, new_fit: float, old_fit: float) -> bool:
        if self.problem.objective == "min":
            return float(new_fit) <= float(old_fit)
        return float(new_fit) >= float(old_fit)

    def _weighted_lehmer(self, values: np.ndarray, improvements: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        improvements = np.asarray(improvements, dtype=float)
        if values.size == 0:
            return 1.0
        total = float(np.sum(improvements))
        if total <= _EPS:
            return 1.0
        weights = improvements / total
        denom = float(np.sum(weights * values))
        if abs(denom) <= 1.0e-8:
            return 1.0
        return float(np.sum(weights * values * values) / denom)

    def _sample_f(self, success_rate: float) -> tuple[float, float]:
        mean_f = float(self._params.get("f_base", 0.4)) + float(self._params.get("f_amplitude", 0.25)) * np.tanh(
            float(self._params.get("f_success_slope", 5.0)) * float(success_rate)
        )
        sigma = max(_EPS, float(self._params.get("f_sigma", 0.02)))
        attempts = max(1, int(self._params.get("resampling_attempts", 100)))
        for _ in range(attempts):
            f = float(np.random.normal(mean_f, sigma))
            if 0.0 <= f <= 1.0:
                return f, mean_f
        return float(np.clip(mean_f, 0.0, 1.0)), mean_f

    def _sample_cr(self, memory: np.ndarray) -> float:
        idx = int(np.random.randint(memory.size))
        cr = float(np.random.normal(float(memory[idx]), float(self._params.get("cr_sigma", 0.05))))
        return float(np.clip(cr, 0.0, 1.0))

    def _pbest_count(self, n: int, success_rate: float) -> int:
        raw = float(self._params.get("pbest_base", 0.7)) * np.exp(-float(self._params.get("pbest_success_slope", 7.0)) * float(success_rate))
        return int(np.clip(max(2, int(float(n) * raw)), 1, int(n)))

    def _choose_from_order(self, order: np.ndarray, limit: int, exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        candidates = [int(i) for i in np.asarray(order[:max(1, int(limit))], dtype=int).tolist() if int(i) not in exclude]
        if not candidates:
            candidates = [int(i) for i in np.asarray(order, dtype=int).tolist() if int(i) not in exclude]
        if not candidates:
            candidates = [int(np.asarray(order, dtype=int)[0])]
        return int(candidates[int(np.random.randint(len(candidates)))])

    def _rank_select_index(self, order: np.ndarray, n: int, exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        ranked = [int(i) for i in np.asarray(order, dtype=int).tolist() if int(i) not in exclude]
        if not ranked:
            ranked = [int(i) for i in np.asarray(order, dtype=int).tolist()]
        ranks = np.arange(len(ranked), dtype=float)
        pressure = float(self._params.get("rank_pressure_coefficient", 3.0))
        weights = np.exp(-pressure * ranks / float(max(1, n)))
        weights = weights / (float(np.sum(weights)) + _EPS)
        return int(np.random.choice(np.asarray(ranked, dtype=int), p=weights))

    def _repair_bounds(self, trial: np.ndarray, contrib: dict[str, float], counts: dict[str, int]) -> np.ndarray:
        repaired = np.asarray(trial, dtype=float).copy()
        below = repaired < self._lo
        above = repaired > self._hi
        mask = below | above
        if np.any(mask):
            handling = str(self._params.get("bound_handling", "resample")).lower()
            if handling in {"clip", "clamp"}:
                repaired = np.clip(repaired, self._lo, self._hi)
            elif handling in {"midpoint", "target_midpoint"}:
                repaired[below] = 0.5 * (self._lo[below] + repaired[below])
                repaired[above] = 0.5 * (self._hi[above] + repaired[above])
                repaired = np.clip(repaired, self._lo, self._hi)
            else:
                repaired[mask] = np.random.uniform(self._lo[mask], self._hi[mask])
            self._add(contrib, counts, "bound_resampling", 0.0, int(np.count_nonzero(mask)))
        return self.problem.apply_variable_types(repaired)

    def _target_population_size(self, evaluations_after_step: int, initial_n: int, current_n: int, step: int) -> int:
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            progress = min(1.0, max(0.0, float(evaluations_after_step) / float(self.config.max_evaluations)))
        else:
            horizon = max(1, int(self.config.max_steps or 100))
            progress = min(1.0, max(0.0, float(step + 1) / float(horizon)))
        target = int(float(initial_n) + float(min_n - initial_n) * progress)
        return int(np.clip(target, min_n, max(min_n, int(current_n))))

    def _reduce_population(self, pop: np.ndarray, target_n: int) -> tuple[np.ndarray, int]:
        if pop.shape[0] <= target_n:
            return pop, 0
        keep = self._order(pop[:, -1])[:target_n]
        removed = pop.shape[0] - int(target_n)
        return pop[keep].copy(), int(removed)

    def _step_impl(self, state, pop: np.ndarray):
        top_pop = np.asarray(pop, dtype=float).copy()
        new_pop = np.asarray(state.payload.get("new_population", top_pop), dtype=float).copy()
        if new_pop.ndim != 2 or new_pop.shape[1] != self.problem.dimension + 1:
            new_pop = top_pop.copy()
        n = int(min(top_pop.shape[0], new_pop.shape[0]))
        if n < 4:
            raise ValueError("L-SRTDE requires at least four population members.")
        top_pop = top_pop[:n].copy()
        new_pop = new_pop[:n].copy()

        memory_cr = np.asarray(state.payload.get("M_CR", np.ones(5)), dtype=float).copy()
        if memory_cr.size == 0:
            memory_cr = np.ones(5, dtype=float)
        memory_index = int(state.payload.get("memory_index", 0)) % memory_cr.size
        success_rate = float(state.payload.get("success_rate", self._params.get("initial_success_rate", 0.5)))
        initial_n = int(state.payload.get("initial_n", n))
        pf_index = int(state.payload.get("pf_index", 0)) % n

        order_top = self._order(top_pop[:, -1])
        order_new = self._order(new_pop[:, -1])
        pbest_count = self._pbest_count(n, success_rate)

        contrib = self._blank_contribs()
        counts = self._blank_counts()
        self._add(contrib, counts, "success_rate_f_adaptation", 0.0, n)
        self._add(contrib, counts, "success_rate_pbest_control", 0.0, n)
        self._add(contrib, counts, "rank_selective_pressure", 0.0, n)

        successes: list[np.ndarray] = []
        success_cr: list[float] = []
        success_delta: list[float] = []
        total_gain = 0.0
        dim = int(self.problem.dimension)

        for _ in range(n):
            chosen = int(np.random.randint(n))
            pbest = self._choose_from_order(order_top, pbest_count, exclude={chosen})
            r_new = self._rank_select_index(order_new, n, exclude={pbest})
            r_top = self._choose_from_order(order_top, n, exclude={pbest, r_new})
            f, _mean_f = self._sample_f(success_rate)
            cr = self._sample_cr(memory_cr)

            target = new_pop[chosen, :-1]
            mutant = target + f * (top_pop[pbest, :-1] - target) + f * (new_pop[r_new, :-1] - top_pop[r_top, :-1])
            mask = np.random.rand(dim) < cr
            mask[int(np.random.randint(dim))] = True
            trial = np.where(mask, mutant, target)
            actual_cr = float(np.count_nonzero(mask) / float(dim))
            trial = self._repair_bounds(trial, contrib, counts)
            trial_fit = float(self.problem.evaluate(trial))

            old_fit = float(new_pop[chosen, -1])
            if self._not_worse(trial_fit, old_fit):
                gain = self._gain(old_fit, trial_fit)
                total_gain += gain
                trial_row = np.concatenate((np.asarray(trial, dtype=float), np.array([trial_fit], dtype=float)))
                successes.append(trial_row)
                success_cr.append(actual_cr)
                success_delta.append(gain)
                new_pop[pf_index, :] = trial_row
                pf_index = (pf_index + 1) % n
                self._add(contrib, counts, "selection", gain / 5.0 if gain > 0.0 else 0.0, 1)
                self._add(contrib, counts, "r_new_to_ptop_mutation", gain / 5.0 if gain > 0.0 else 0.0, 1)
                self._add(contrib, counts, "binomial_crossover", gain / 5.0 if gain > 0.0 else 0.0, 1)
                self._add(contrib, counts, "newest_population_update", gain / 5.0 if gain > 0.0 else 0.0, 1)
            else:
                self._add(contrib, counts, "selection", 0.0, 1)
                self._add(contrib, counts, "r_new_to_ptop_mutation", 0.0, 1)
                self._add(contrib, counts, "binomial_crossover", 0.0, 1)

        new_success_rate = float(len(successes) / float(n))
        evals_after_step = int(state.evaluations) + n
        target_n = self._target_population_size(evals_after_step, initial_n, n, int(state.step))

        new_pop, removed_new = self._reduce_population(new_pop, target_n)
        if removed_new:
            self._add(contrib, counts, "linear_population_reduction", 0.0, removed_new)

        if successes:
            success_block = np.vstack(successes)
            top_candidates = np.vstack((top_pop, success_block))
        else:
            top_candidates = top_pop
        top_pop, removed_top = self._reduce_population(top_candidates, target_n)
        self._add(contrib, counts, "top_population_update", total_gain / 5.0 if total_gain > 0.0 else 0.0, len(successes))
        if removed_top and not removed_new:
            self._add(contrib, counts, "linear_population_reduction", 0.0, removed_top)

        if successes:
            old_cr = float(memory_cr[memory_index])
            mean_cr = self._weighted_lehmer(np.asarray(success_cr, dtype=float), np.asarray(success_delta, dtype=float))
            mixing = float(np.clip(self._params.get("cr_memory_mixing", 0.5), 0.0, 1.0))
            memory_cr[memory_index] = (1.0 - mixing) * old_cr + mixing * mean_cr
            memory_cr[memory_index] = float(np.clip(memory_cr[memory_index], 0.0, 1.0))
            memory_index = (memory_index + 1) % memory_cr.size
            self._add(contrib, counts, "crossover_memory_update", 0.0, 1)

        pf_index = int(pf_index % max(1, new_pop.shape[0]))
        self._last_operator_contributions = {label: float(value) for label, value in contrib.items()}
        self._last_operator_counts = {label: int(value) for label, value in counts.items()}
        return top_pop, n, {
            "new_population": new_pop,
            "M_CR": memory_cr,
            "memory_index": memory_index,
            "success_rate": new_success_rate,
            "initial_n": initial_n,
            "pf_index": pf_index,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["success_rate"] = float(state.payload.get("success_rate", 0.0))
        obs["mean_crossover_memory"] = float(np.mean(np.asarray(state.payload.get("M_CR", [1.0]), dtype=float)))
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        return obs
