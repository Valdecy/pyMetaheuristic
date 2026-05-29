"""pyMetaheuristic src — Birds-of-Paradise Search Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class BPSEngine(PortedPopulationEngine):
    """Birds-of-Paradise Search.

    Assumptions documented for faithfulness:
    * The movement-control expression in Eq. (2) is OCR-damaged in the PDF. This
      implementation uses the intended absolute time-scaled random control
      ``abs((t / Tmax) * (2 * rand - 1))``.
    * The paper describes long-distance movement when a bird's current tree is not
      fruitful, but it does not specify a stagnation counter. A small per-bird
      patience counter is therefore used before switching the local branch to the
      long-distance branch.
    * The boundary equations in the paper are described as reflective re-entry.
      Because the printed formulas can push candidates outside the box again, a
      standard repeated reflection repair is used.
    """

    algorithm_id = "bps"
    algorithm_name = "Birds-of-Paradise Search"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s00521-026-11887-6",
        "title": "An innovative metaheuristic algorithm inspired by behavior of birds-of-paradise in tropical rainforests for global optimization",
        "authors": "Moh Nur Sholeh, Linda Karlina",
        "year": 2026,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        c0=0.5,
        beta_max=0.1,
        gamma=0.1,
        omega=3.0,
        chaos_alpha=4.0,
        chaos_warmup=5,
        stagnation_limit=3,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("bps requires population_size >= 2.")
        if not 0.0 <= float(self._params.get("c0", 0.5)) <= 1.0:
            raise ValueError("bps c0 must be in [0, 1].")
        if float(self._params.get("beta_max", 0.1)) <= 0.0:
            raise ValueError("bps beta_max must be positive.")
        if float(self._params.get("gamma", 0.1)) < 0.0:
            raise ValueError("bps gamma must be non-negative.")
        if float(self._params.get("omega", 3.0)) < 0.0:
            raise ValueError("bps omega must be non-negative.")
        if float(self._params.get("chaos_alpha", 4.0)) <= 0.0:
            raise ValueError("bps chaos_alpha must be positive.")
        if int(self._params.get("chaos_warmup", 5)) < 0:
            raise ValueError("bps chaos_warmup must be >= 0.")
        if int(self._params.get("stagnation_limit", 3)) < 1:
            raise ValueError("bps stagnation_limit must be at least 1.")

    def _reflect_into_box(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo, hi = self._lo, self._hi
        span = hi - lo
        out = x.copy()
        zero_mask = span <= 0.0
        if np.any(zero_mask):
            out[..., zero_mask] = lo[zero_mask]
        active = ~zero_mask
        if np.any(active):
            doubled = 2.0 * span[active]
            y = np.mod(out[..., active] - lo[active], doubled)
            out[..., active] = lo[active] + np.where(y <= span[active], y, doubled - y)
        return out

    def _new_positions(self, n: int | None = None) -> np.ndarray:
        n_points = self._n if n is None else int(n)
        dim = self.problem.dimension
        total = n_points * dim
        alpha = float(self._params.get("chaos_alpha", 4.0))
        warmup = int(self._params.get("chaos_warmup", 5))
        z = np.random.uniform(1.0e-6, 1.0 - 1.0e-6)
        seq = np.empty(total, dtype=float)
        for k in range(total + warmup):
            z = alpha * z * (1.0 - z)
            if k >= warmup:
                seq[k - warmup] = np.clip(z, 1.0e-12, 1.0 - 1.0e-12)
        return self._lo + seq.reshape(n_points, dim) * self._span

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"stagnation": np.zeros(pop.shape[0], dtype=int)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        current = pop[:, :-1].copy()
        current_fit = pop[:, -1].copy()
        stagnation = np.asarray(state.payload.get("stagnation", np.zeros(n, dtype=int)), dtype=int).copy()
        if stagnation.shape != (n,):
            stagnation = np.zeros(n, dtype=int)

        order = self._order(current_fit)
        best = current[order[0]].copy()
        pmean = np.mean(current, axis=0)
        beta_max = float(self._params.get("beta_max", 0.1))
        gamma = float(self._params.get("gamma", 0.1))
        omega = float(self._params.get("omega", 3.0))
        c0 = float(self._params.get("c0", 0.5))
        patience = int(self._params.get("stagnation_limit", 3))
        Tmax = max(1, int(self.config.max_steps or max(50, state.step + 1)))
        t = min(Tmax, state.step + 1)
        beta = beta_max * (1.0 - t / Tmax)

        trials = np.empty_like(current)
        attempted_labels = ["carryover"] * n
        for i in range(n):
            cm = abs((t / Tmax) * (2.0 * np.random.rand() - 1.0))
            if cm > c0:
                if stagnation[i] >= patience:
                    candidate = current[i] + np.random.uniform(-1.0, 1.0, dim) * (best - gamma * np.random.rand(dim) * self._span)
                    attempted_labels[i] = "bps.long_distance_flight"
                else:
                    j = self._rand_indices(n, i, 1)[0]
                    delta = np.random.uniform(-1.0, 1.0, dim)
                    candidate = current[i] + np.random.rand(dim) * (current[j] - current[i]) + beta * delta
                    attempted_labels[i] = "bps.local_tree_movement"
            else:
                candidate = current[i] + np.random.rand(dim) * (best - omega * np.random.rand(dim) * pmean)
                attempted_labels[i] = "bps.best_tree_attraction"
            trials[i] = self._reflect_into_box(candidate)

        trial_fit = self._evaluate_population(trials)
        improved = self._better_mask(trial_fit, current_fit)
        new_pop = pop.copy()
        new_pop[improved, :-1] = trials[improved]
        new_pop[improved, -1] = trial_fit[improved]
        stagnation[improved] = 0
        stagnation[~improved] += 1
        operator_labels = [attempted_labels[i] if bool(improved[i]) else "carryover" for i in range(n)]
        return new_pop, n, {"stagnation": stagnation, "operator_labels": operator_labels}
