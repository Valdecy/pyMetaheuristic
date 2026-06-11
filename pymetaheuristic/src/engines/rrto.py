"""pyMetaheuristic src — RRT-based Optimizer Engine."""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class RRTOEngine(PortedPopulationEngine):
    """RRT-based Optimizer — adaptive step-size strategies inspired by RRT."""

    algorithm_id = "rrto"
    algorithm_name = "RRT-based Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1109/ACCESS.2025.3547537",
        "source": "Lai, Li and Shi (2025), RRT-based Optimizer.",
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
    _DEFAULTS = dict(population_size=30, step_penalty=10.0)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._C = float(self._params.get("step_penalty", 10.0))
        if self._C <= 0.0:
            raise ValueError("step_penalty must be positive.")

    def _k_e(self, t: int) -> tuple[float, float]:
        total = max(2, int(self.config.max_steps or 500))
        if t >= total:
            k = 0.0
        else:
            k = math.log(max(float(total - t), 1.0)) / math.log(float(total))
            k = float(np.clip(k, 0.0, 1.0))
        e = float((max(0.0, min(1.0, float(t) / float(total)))) ** (1.0 / 3.0))
        return k, e

    def _evaluate_strategy(self, pop: np.ndarray, labels: list[str], positions: np.ndarray, changed: np.ndarray, label: str) -> tuple[np.ndarray, list[str], int]:
        positions = np.clip(positions, self._lo, self._hi)
        fit = self._evaluate_population(positions)
        out = np.hstack((positions, fit[:, None]))
        for i in range(out.shape[0]):
            if bool(changed[i]):
                labels[i] = label
        return out, labels, out.shape[0]

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = max(1, int(state.step) + 1)
        total = max(2, int(self.config.max_steps or 500))
        ratio = min(1.0, float(t) / float(total))
        k, e = self._k_e(t)
        m1 = e / 10.0
        m2 = e / 50.0
        labels = ["carryover"] * n
        evals = 0

        # Current best agent before this macro-step.
        best_idx = self._best_index(pop[:, -1])
        qbest = pop[best_idx, :-1].copy()

        # Strategy 1: adaptive step-size wandering, Eqs. (13)-(14).
        pos = pop[:, :-1].copy()
        changed = np.zeros(n, dtype=bool)
        if k > 0.0:
            r1 = np.random.rand(n, dim)
            mask = r1 < k
            s1 = (r1 - k / 2.0) * k * (self._span / self._C)
            pos = np.where(mask, pos + s1, pos)
            changed = np.any(mask, axis=1)
        if np.any(changed):
            pop, labels, inc = self._evaluate_strategy(pop, labels, pos, changed, "rrto.adaptive_step_size_wandering")
            evals += inc
            best_idx = self._best_index(pop[:, -1])
            qbest = pop[best_idx, :-1].copy()

        # Strategy 2: absolute-difference adaptive step-size, Eqs. (15)-(19).
        pos = pop[:, :-1].copy()
        changed = np.zeros(n, dtype=bool)
        if m1 > 0.0:
            r2 = np.random.rand(n, dim)
            mask = r2 < m1
            b = math.e ** math.cos(math.pi - math.pi / max(float(t), 1.0))
            alpha1 = 5.0 * (r2 - m1 / 2.0) * np.cos(2.0 * math.pi * r2) * b
            s2 = alpha1 * np.abs(qbest[None, :] - pos)
            pos = np.where(mask, qbest[None, :] + s2, pos)
            changed = np.any(mask, axis=1)
        if np.any(changed):
            pop, labels, inc = self._evaluate_strategy(pop, labels, pos, changed, "rrto.absolute_difference_step")
            evals += inc
            best_idx = self._best_index(pop[:, -1])
            qbest = pop[best_idx, :-1].copy()

        # Strategy 3: boundary-based adaptive step-size, Eqs. (20)-(24).
        pos = pop[:, :-1].copy()
        changed = np.zeros(n, dtype=bool)
        if m2 > 0.0:
            r3 = np.random.rand(n, dim)
            mask = r3 < m2
            beta = 10.0 * math.pi * ratio
            alpha2 = r3 * (r3 - m2 / 2.0) * k * (1.0 - ratio)
            s3 = self._span * math.cos(beta) * alpha2
            pos = np.where(mask, qbest[None, :] + s3, pos)
            changed = np.any(mask, axis=1)
        if np.any(changed):
            pop, labels, inc = self._evaluate_strategy(pop, labels, pos, changed, "rrto.boundary_based_step")
            evals += inc

        return pop, evals, {
            "operator_labels": labels,
            "native_evomapx_operator_labels": True,
            "rrto_k": float(k),
            "rrto_e": float(e),
            "rrto_m1": float(m1),
            "rrto_m2": float(m2),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs.update({
            "rrto_k": float(state.payload.get("rrto_k", 0.0)),
            "rrto_e": float(state.payload.get("rrto_e", 0.0)),
            "rrto_m1": float(state.payload.get("rrto_m1", 0.0)),
            "rrto_m2": float(state.payload.get("rrto_m2", 0.0)),
        })
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_fidelity", "native_labels")
        return obs
