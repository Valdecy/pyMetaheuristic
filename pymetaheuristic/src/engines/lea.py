"""pyMetaheuristic src — Love Evolution Algorithm Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class LEAEngine(PortedPopulationEngine):
    """Love Evolution Algorithm using stimulus, value, and role phases."""

    algorithm_id = "lea"
    algorithm_name = "Love Evolution Algorithm"
    family = "human"
    _REFERENCE = {
        "doi": "10.1007/s11227-024-05905-4",
        "source": "Gao et al. (2024), Love Evolution Algorithm: a stimulus-value-role theory-inspired evolutionary algorithm.",
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
    _DEFAULTS = dict(
        population_size=30,
        hmax=0.7,
        hmin=0.0,
        lambda_c=0.5,
        lambda_p=0.5,
        epsilon=1.0e-12,
    )

    def __init__(self, problem, config):
        super().__init__(problem, config)
        if self._n % 2:
            self._n += 1
        self._hmax = float(self._params.get("hmax", 0.7))
        self._hmin = float(self._params.get("hmin", 0.0))
        self._lambda_c = float(self._params.get("lambda_c", 0.5))
        self._lambda_p = float(self._params.get("lambda_p", 0.5))
        self._eps = float(self._params.get("epsilon", 1.0e-12))

    def _mod_bounds(self, x: np.ndarray) -> np.ndarray:
        return self._lo + np.mod(x - self._lo, self._span)

    def _reflect_pair(self, a: np.ndarray, b: np.ndarray, g: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
        dim = self.problem.dimension
        za = int(np.random.randint(dim))
        kb = int(np.random.randint(dim))
        delta = 0.5 * ((a[za] - self._lo[za]) / self._span[za] + (b[kb] - self._lo[kb]) / self._span[kb])
        alpha_a = np.random.uniform(-1.0, 1.0, dim)
        alpha_b = np.random.uniform(-1.0, 1.0, dim)
        sa = alpha_a * a / (b + np.sign(b) * self._eps + (b == 0.0) * self._eps)
        sb = alpha_b * b / (a + np.sign(a) * self._eps + (a == 0.0) * self._eps)
        return g + mu * sa * delta, g + mu * sb * delta

    def _role_pair(self, a: np.ndarray, b: np.ndarray, g: np.ndarray, mu: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        xi = a * b
        denom = float(np.max(xi) - np.min(xi)) + self._eps
        xi = (xi - np.min(xi)) / denom + h
        gamma_a = np.random.uniform(-1.0, 1.0, a.shape)
        gamma_b = np.random.uniform(-1.0, 1.0, b.shape)
        return g + gamma_a * mu * xi, g + gamma_b * mu * xi

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        if n % 2:
            pop = pop[:-1].copy()
            n -= 1
        x = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        old_x = x.copy()
        best = x[self._best_index(fit)].copy()
        mu = float(np.mean(np.abs(x - best))) + self._eps
        h = self._hmin + (self._hmax - self._hmin) * np.random.rand()

        order = np.random.permutation(n)
        a_idx = order[: n // 2]
        b_idx = order[n // 2 :]
        new_x = x.copy()
        labels = ["carryover"] * n

        for ia, ib in zip(a_idx, b_idx):
            a = new_x[ia].copy()
            b = new_x[ib].copy()
            rc = np.random.rand()
            c_vec = rc * (a - b) ** 2
            c = float(np.mean(c_vec / (np.max(c_vec) + np.min(c_vec) + self._eps)))

            if c < self._lambda_c:
                # Value phase: inter-variable convolution/division around G.
                tau_a = np.random.uniform(-1.0, 1.0, dim)
                tau_b = np.random.uniform(-1.0, 1.0, dim)
                omega_a = np.random.uniform(-1.0, 1.0, dim)
                omega_b = np.random.uniform(-1.0, 1.0, dim)
                phi1 = best * a
                phi2 = best * best + a * b
                phi3 = best * b
                a = a * tau_a + omega_a * np.abs(phi2 - phi1)
                b = b * tau_b + omega_b * np.abs(phi2 - phi3)
                a = self._mod_bounds(a)
                b = self._mod_bounds(b)

                d = float(np.linalg.norm(a - b)) + self._eps
                p = float(np.random.rand() * c * np.sum(np.abs(a - b)) / (d * mu + self._eps))
                if p > self._lambda_p:
                    a, b = self._reflect_pair(a, b, best, mu)
                    pair_label = "lea.value_phase_reflection_operation"
                else:
                    a, b = self._role_pair(a, b, best, mu, h)
                    pair_label = "lea.value_phase_role_phase"
            else:
                a, b = self._reflect_pair(a, b, best, mu)
                pair_label = "lea.reflection_operation"

            new_x[ia] = np.clip(a, self._lo, self._hi)
            new_x[ib] = np.clip(b, self._lo, self._hi)
            labels[int(ia)] = pair_label if np.linalg.norm(new_x[ia] - old_x[ia]) > 0.0 else "carryover"
            labels[int(ib)] = pair_label if np.linalg.norm(new_x[ib] - old_x[ib]) > 0.0 else "carryover"

        new_fit = self._evaluate_population(new_x)
        evals = n
        # LEA is generational; pymetaheuristic keeps the incumbent best in EngineState.
        out = np.hstack((new_x, new_fit[:, None]))
        return out, evals, {
            "operator_labels": labels,
            "native_evomapx_operator_labels": True,
            "lea_h": float(h),
            "lea_mu": float(mu),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["lea_h"] = float(state.payload.get("lea_h", 0.0))
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_fidelity", "native_labels")
        return obs
