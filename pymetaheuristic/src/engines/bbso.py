"""pyMetaheuristic src — Boxelder Bug Search Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, EngineState
from ._ported_common import PortedPopulationEngine


class BBSOEngine(PortedPopulationEngine):
    """Boxelder Bug Search Optimization with coordinated following and population reduction."""

    algorithm_id = "bbso"
    algorithm_name = "Boxelder Bug Search Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "",
        "authors": "Iraj Faraji Davoudkhani, Hossein Shayeghi, Abdollah Younesi",
        "year": 2025,
        "note": "DOI was not assigned in the provided reference implementation.",
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, population_reduction=0.5, min_population_size=None)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("BBSO requires population_size >= 2.")
        self._initial_population_size = self._n
        self._max_evaluations_hint = self._resolve_max_evaluations_hint()

    def _resolve_max_evaluations_hint(self) -> int:
        if self.config.max_evaluations is not None:
            return max(1, int(self.config.max_evaluations))
        max_steps = max(1, int(self.config.max_steps or 500))
        return max(1, int(self._n + max_steps * max(1, self._n * (self._n - 1) // 2)))

    def _temperature_ratio(self, fitness: np.ndarray, index: int) -> float:
        """Positive cost-ratio used by the BBSO perturbation equation.

        The source equation uses Cost/Sfr and assumes a positive minimization
        landscape. This shifted form preserves the minimization ordering while
        avoiding invalid powers on functions such as Easom, whose optimum is
        negative.
        """
        fit = np.asarray(fitness, dtype=float)
        if self.problem.objective == "min":
            scaled = fit - np.min(fit)
        else:
            scaled = np.max(fit) - fit
        eps = 1.0e-12
        scaled = scaled + eps
        mean_scaled = float(np.mean(scaled)) + eps
        return float(max(scaled[index] / mean_scaled, eps))

    def _min_population_size(self, current_n: int) -> int:
        configured = self._params.get("min_population_size", None)
        if configured is None:
            configured = self.problem.dimension
        value = max(2, int(configured))
        return min(max(2, current_n), value)

    def initialize(self) -> EngineState:
        state = super().initialize()
        pop = np.asarray(state.payload["population"], dtype=float)
        pop = pop[self._order(pop[:, -1])]
        state.payload["population"] = pop
        state.best_position = pop[0, :-1].tolist()
        state.best_fitness = float(pop[0, -1])
        return state

    def _step_impl(self, state: EngineState, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        pop = pop[self._order(pop[:, -1])].copy()
        reduction = float(self._params.get("population_reduction", 0.5))
        if not 0.0 <= reduction <= 1.0:
            raise ValueError("population_reduction must be in [0, 1].")

        new_rows: list[np.ndarray] = []
        evals = 0
        safe_hi = np.where(np.abs(self._hi) > 1.0e-12, self._hi, self._span)

        for i in range(n):
            best_trial = None
            best_trial_fitness = self.problem.worst_fitness()

            # MATLAB source uses for j = 2:i (1-based). In Python this is
            # j = 1, ..., i, so the first bug has no coordinated trial.
            for j in range(1, i + 1):
                fr = self._temperature_ratio(pop[:, -1], i)
                progress = max(1.0, (state.evaluations + evals) / max(1, n))
                # Compute fr**(-progress) in log-space to avoid overflow on
                # very flat or shifted-cost landscapes.
                log_gain = np.clip(-progress * np.log(fr), -18.0, 18.0)
                perturbation_gain = float(np.exp(log_gain))

                trial = (
                    pop[j, :-1]
                    + np.random.random(dim) * (pop[j, :-1] - pop[i, :-1])
                    + np.random.random(dim) * (pop[j - 1, :-1] - pop[j, :-1])
                    + perturbation_gain * np.random.randn(dim) * (pop[j, :-1] / safe_hi)
                )
                trial = np.clip(trial, self._lo, self._hi)
                trial_fitness = float(self.problem.evaluate(trial))
                evals += 1

                if self._is_better(trial_fitness, best_trial_fitness):
                    best_trial = trial.copy()
                    best_trial_fitness = trial_fitness

            if best_trial is not None:
                new_rows.append(np.append(best_trial, best_trial_fitness))

        if new_rows:
            candidates = np.vstack([pop, np.vstack(new_rows)])
        else:
            candidates = pop

        candidates = candidates[self._order(candidates[:, -1])]
        progress_ratio = min(1.0, (state.evaluations + evals) / self._max_evaluations_hint)
        n_new = int(round(n - reduction * n * progress_ratio))
        n_new = max(self._min_population_size(n), min(n, n_new))
        pop = candidates[:n_new].copy()
        self._n = pop.shape[0]
        return pop, evals, {}
