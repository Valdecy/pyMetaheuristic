"""pyMetaheuristic src — Delta Plus Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class DPEngine(PortedPopulationEngine):
    """Delta Plus — inspiration-independent population optimizer."""

    algorithm_id = "dp"
    algorithm_name = "Delta Plus"
    family = "math"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-05094-y",
        "source": "Gao et al. (2025), Freedom from inspiration! Achieving efficient metaheuristic optimization with Delta Plus.",
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
    _DEFAULTS = dict(population_size=30, w1=1.0, w2=1.0, epsilon=1.0e-12)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._w1 = float(self._params.get("w1", 1.0))
        self._w2 = float(self._params.get("w2", 1.0))
        self._eps = float(self._params.get("epsilon", 1.0e-12))

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"previous_delta": np.zeros((pop.shape[0], self.problem.dimension), dtype=float)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        x = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        previous_delta = np.asarray(state.payload.get("previous_delta", np.zeros_like(x)), dtype=float)
        if previous_delta.shape != x.shape:
            previous_delta = np.zeros_like(x)

        t = max(1, int(state.step) + 1)
        total = max(1, int(self.config.max_steps or 500))
        eta = max(0.0, 1.0 - float(t) / float(total))
        mean = np.mean(x, axis=0)
        new_pop = pop.copy()
        new_delta = np.zeros_like(x)
        operator_labels = ["carryover"] * n
        evals = 0

        for i in range(n):
            # Eq. (2): information diversity against all other candidates.
            li = np.zeros(dim, dtype=float)
            for k in range(n):
                if k == i:
                    continue
                nt = np.random.rand(dim) * (x[k] - x[i])
                li += np.random.rand(dim) * nt

            # Eq. (3): realtime learning vector.
            vi = li / (np.linalg.norm(li) + self._eps)

            # Eq. (4): inertial learning vector from the previous delta.
            if t <= 1:
                ci = vi.copy()
            else:
                ci = previous_delta[i] / (np.linalg.norm(previous_delta[i]) + self._eps)

            # Eq. (5): Delta operation. The paper writes lambda_i as a scalar
            # norm-based step-size; rm is sampled as a standard-normal vector.
            rm = np.random.normal(0.0, 1.0, dim)
            lam = (eta * np.random.rand() + self._w1) * np.linalg.norm(mean + self._w2 * eta * rm - x[i])
            delta = lam * (np.random.rand() * vi + np.random.rand() * ci)
            new_delta[i] = delta

            trial = np.clip(x[i] + delta, self._lo, self._hi)
            trial_fit = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(trial_fit, fit[i]):
                new_pop[i, :-1] = trial
                new_pop[i, -1] = trial_fit
                operator_labels[i] = "dp.delta_operation"

        return new_pop, evals, {
            "previous_delta": new_delta,
            "operator_labels": operator_labels,
            "native_evomapx_operator_labels": True,
            "dp_eta": float(eta),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["delta_eta"] = float(state.payload.get("dp_eta", 0.0))
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_fidelity", "native_labels")
        return obs
