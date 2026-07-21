"""pyMetaheuristic src — Delta Plus Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class DPEngine(PortedPopulationEngine):
    """Delta Plus — population optimizer driven by the native Delta operation."""

    algorithm_id = "dp"
    algorithm_name = "Delta Plus"
    family = "math"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-05094-y",
        "source": (
            "Gao, Wang, Qin, Zhang, and Wang (2025), "
            "Freedom from inspiration! Achieving efficient metaheuristic "
            "optimization with delta plus."
        ),
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=False,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    # The paper reports w1=w2=1 for its general benchmark experiments. It does
    # not prescribe one universal population size; 50 is the first/main CEC2017
    # setting, while 100 is used for CEC2022 and the engineering experiments.
    _DEFAULTS = dict(
        population_size=50,
        w1=1.0,
        w2=1.0,
        epsilon=float(np.finfo(float).eps),
    )

    _DIRECT_OPERATOR = "dp.delta_operation"

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._w1 = float(self._params.get("w1", 1.0))
        self._w2 = float(self._params.get("w2", 1.0))
        self._eps = float(self._params.get("epsilon", np.finfo(float).eps))
        if not np.isfinite(self._w1) or not np.isfinite(self._w2):
            raise ValueError("Delta Plus parameters w1 and w2 must be finite.")
        if not np.isfinite(self._eps) or self._eps <= 0.0:
            raise ValueError("Delta Plus epsilon must be finite and strictly positive.")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n, dim = pop.shape[0], self.problem.dimension
        return {
            "previous_delta": np.zeros((n, dim), dtype=float),
            "operator_labels": ["initialization"] * n,
            "operator_counts": {},
            "operator_contributions": {},
            "dp_acceptances": 0,
            "dp_trials": 0,
            "dp_eta": 1.0,
        }

    def _schedule_horizon(self, population_size: int) -> int:
        """Return T in eta=1-t/T from the configured stopping budget."""
        if self.config.max_steps is not None:
            return max(1, int(self.config.max_steps))
        if self.config.max_evaluations is not None:
            # Initialization consumes n evaluations and every complete DP
            # macro-iteration consumes another n evaluations.
            remaining = max(0, int(self.config.max_evaluations) - int(population_size))
            return max(1, remaining // max(1, int(population_size)))
        # The paper's exploration/exploitation illustration uses 500 iterations.
        # This is only a schedule fallback; callers still need a stopping rule.
        return 500

    def _positive_improvement(self, old_fitness: float, new_fitness: float) -> float:
        if self.problem.objective == "max":
            return max(0.0, float(new_fitness) - float(old_fitness))
        return max(0.0, float(old_fitness) - float(new_fitness))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        x = np.asarray(pop[:, :-1], dtype=float).copy()
        fit = np.asarray(pop[:, -1], dtype=float).copy()

        previous_delta = np.asarray(
            state.payload.get("previous_delta", np.zeros_like(x)), dtype=float
        )
        if previous_delta.shape != x.shape or not np.all(np.isfinite(previous_delta)):
            previous_delta = np.zeros_like(x)

        t = max(1, int(state.step) + 1)
        total = self._schedule_horizon(n)
        eta = float(np.clip(1.0 - float(t) / float(total), 0.0, 1.0))
        population_mean = np.mean(x, axis=0)

        # All Delta vectors are calculated from the same generation x^t. The
        # greedy replacement is committed only after each trial is evaluated.
        new_pop = pop.copy()
        current_delta = np.zeros_like(x)
        operator_labels = ["carryover"] * n
        accepted = 0
        total_improvement = 0.0
        step_norm_sum = 0.0
        realtime_norm_sum = 0.0
        inertial_norm_sum = 0.0

        for i in range(n):
            learning_sum = np.zeros(dim, dtype=float)
            for k in range(n):
                if k == i:
                    continue

                # Eq. (2): r_n is a scalar random number, so every coordinate
                # of the pairwise difference receives the same random scale.
                information_diversity = np.random.rand() * (x[k] - x[i])

                # Eq. (3): r_v is a 1xd vector. A fresh vector is sampled for
                # each peer contribution in the summation.
                learning_sum += np.random.rand(dim) * information_diversity

            realtime = learning_sum / (np.linalg.norm(learning_sum) + self._eps)

            # Eq. (4): at t=1 the inertial vector equals the realtime vector;
            # subsequently it is the normalized Delta vector from t-1.
            if t == 1:
                inertial = realtime.copy()
            else:
                inertial = previous_delta[i] / (
                    np.linalg.norm(previous_delta[i]) + self._eps
                )

            # Eq. (5): r_a, r_b, r_k, and r_m are scalar random numbers in the
            # paper. r_m follows N(0,1) and broadcasts across the mean vector.
            r_a = np.random.rand()
            r_b = np.random.rand()
            r_k = np.random.rand()
            r_m = np.random.normal()
            step_size = (eta * r_k + self._w1) * np.linalg.norm(
                population_mean + self._w2 * eta * r_m - x[i]
            )
            delta = step_size * (r_a * realtime + r_b * inertial)
            current_delta[i] = delta

            trial = np.clip(x[i] + delta, self._lo, self._hi)
            trial_fitness = float(self.problem.evaluate(trial))

            # Eq. (6): strict greedy replacement. ProblemSpec.evaluate may
            # project/repair the trial in-place, so the stored position is the
            # exact candidate whose fitness was evaluated.
            if self._is_better(trial_fitness, fit[i]):
                new_pop[i, :-1] = trial
                new_pop[i, -1] = trial_fitness
                operator_labels[i] = self._DIRECT_OPERATOR
                accepted += 1
                total_improvement += self._positive_improvement(fit[i], trial_fitness)

            step_norm_sum += float(np.linalg.norm(delta))
            realtime_norm_sum += float(np.linalg.norm(realtime))
            inertial_norm_sum += float(np.linalg.norm(inertial))

        return new_pop, n, {
            # The paper reuses the generated Delta vector, not merely the
            # accepted displacement, in the next iteration's inertial term.
            "previous_delta": current_delta,
            "operator_labels": operator_labels,
            "native_evomapx_operator_labels": True,
            "operator_counts": {self._DIRECT_OPERATOR: int(n)},
            "operator_contributions": {
                self._DIRECT_OPERATOR: float(total_improvement)
            },
            "operator_acceptances": {self._DIRECT_OPERATOR: int(accepted)},
            "dp_trials": int(n),
            "dp_acceptances": int(accepted),
            "dp_acceptance_rate": float(accepted / max(1, n)),
            "dp_eta": eta,
            "dp_schedule_horizon": int(total),
            "dp_mean_delta_norm": float(step_norm_sum / max(1, n)),
            "dp_mean_realtime_norm": float(realtime_norm_sum / max(1, n)),
            "dp_mean_inertial_norm": float(inertial_norm_sum / max(1, n)),
        }

    def observe(self, state):
        obs = super().observe(state)
        payload = state.payload
        obs.update(
            {
                "delta_eta": float(payload.get("dp_eta", 0.0)),
                "delta_schedule_horizon": int(
                    payload.get("dp_schedule_horizon", self._schedule_horizon(self._n))
                ),
                "delta_trials": int(payload.get("dp_trials", 0)),
                "delta_acceptances": int(payload.get("dp_acceptances", 0)),
                "delta_acceptance_rate": float(payload.get("dp_acceptance_rate", 0.0)),
                "delta_mean_norm": float(payload.get("dp_mean_delta_norm", 0.0)),
                "realtime_learning_mean_norm": float(
                    payload.get("dp_mean_realtime_norm", 0.0)
                ),
                "inertial_learning_mean_norm": float(
                    payload.get("dp_mean_inertial_norm", 0.0)
                ),
                "operator_counts": dict(payload.get("operator_counts", {})),
                "operator_contributions": dict(
                    payload.get("operator_contributions", {})
                ),
                "operator_acceptances": dict(
                    payload.get("operator_acceptances", {})
                ),
                "evomapx_fidelity": "native_operator_contributions",
                "evomapx_delta_f": "positive_accepted_parent_child",
            }
        )
        return obs
