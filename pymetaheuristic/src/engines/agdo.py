"""pyMetaheuristic src — Adam Gradient Descent Optimizer Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight


class AGDOEngine(PortedPopulationEngine):
    """Adam Gradient Descent Optimizer for derivative-free global search."""

    algorithm_id = "agdo"
    algorithm_name = "Adam Gradient Descent Optimizer"
    family = "math"
    _REFERENCE = {
        "doi": "10.1038/s41598-025-01678-9",
        "source": "Xia and Ji (2025), Application of a novel metaheuristic algorithm inspired by Adam gradient descent.",
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
        beta1=0.9,
        beta2=0.999,
        learning_rate=0.001,
        epsilon=1.0e-8,
        levy_beta=1.5,
        momentum_passes=None,
    )

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._beta1 = float(self._params.get("beta1", 0.9))
        self._beta2 = float(self._params.get("beta2", 0.999))
        self._eta = float(self._params.get("learning_rate", 0.001))
        self._eps = float(self._params.get("epsilon", 1.0e-8))
        self._levy_beta = float(self._params.get("levy_beta", 1.5))
        passes = self._params.get("momentum_passes", None)
        self._momentum_passes = None if passes is None else max(1, int(passes))

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        dim = self.problem.dimension
        return {
            "agdo_m": np.zeros(dim, dtype=float),
            "agdo_v": np.zeros(dim, dtype=float),
        }

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        x = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        old_x = x.copy()
        new_pop = pop.copy()
        labels = ["carryover"] * n
        evals = 0

        t = max(1, int(state.step) + 1)
        total = max(1, int(self.config.max_steps or 500))
        tau = float(t) / float(total)
        w_base = tau * tau - 2.0 * tau + 0.5
        decay = max(0.0, 1.0 - tau)

        best_idx = self._best_index(fit)
        best = x[best_idx].copy()
        mean = np.mean(x, axis=0)
        m = np.asarray(state.payload.get("agdo_m", np.zeros(dim)), dtype=float).copy()
        v = np.asarray(state.payload.get("agdo_v", np.zeros(dim)), dtype=float).copy()
        if m.shape != (dim,):
            m = np.zeros(dim, dtype=float)
        if v.shape != (dim,):
            v = np.zeros(dim, dtype=float)

        # Progressive gradient momentum integration. The paper's inner-loop
        # condition is printed ambiguously for integer k; by default we perform
        # one faithful stochastic momentum pass and allow users to raise it.
        passes = self._momentum_passes or 1
        x1 = x.copy()
        for _ in range(passes):
            w = np.random.rand(n, 1) * w_base
            alpha = np.cos(1.0 - np.random.rand(n, dim) * 2.0 * np.pi)
            candidate = w * x1 + alpha * x1
            if passes > 1:
                candidate = x1 + np.sin(2.0 * np.pi * dim * tau) * candidate
            x1 = np.clip(candidate, self._lo, self._hi)

        # Dynamic gradient interaction system with an Adam-like guide vector.
        idx1 = np.random.permutation(n)
        idx2 = np.random.permutation(n)
        a = decay * np.random.rand(n, dim)
        c = 1.0 - np.random.rand()
        p = mean - x1
        g = best - c * p.mean(axis=0)
        m = self._beta1 * m + (1.0 - self._beta1) * g
        v = self._beta2 * v + (1.0 - self._beta2) * (g * g)
        m_hat = m / (1.0 - self._beta1 ** t + self._eps)
        v_hat = v / (1.0 - self._beta2 ** t + self._eps)
        guide = best - self._eta * m_hat / (np.sqrt(v_hat) + self._eps)
        guide = np.clip(guide, self._lo, self._hi)

        denom = np.abs(fit[idx1] - fit) + self._eps
        xi = ((fit[idx1] - fit) / denom).reshape(n, 1)
        new2a = x1 + xi * a * (guide[None, :] - x[idx1]) - a * (x1 - x[idx2])
        new2b = x[idx1] + a * (guide[None, :] - x[idx2])
        # The paper uses a stochastic k-dependent switch; the bounded variant
        # below preserves the intended early/mid-stage alternation.
        switch = (np.random.rand(n, 1) / max(1.0, float(t))) > np.random.rand(n, 1)
        trial = np.where(switch, new2b, new2a)
        trial = np.clip(trial, self._lo, self._hi)
        trial_fit = self._evaluate_population(trial)
        evals += n
        mask = self._better_mask(trial_fit, fit)
        new_pop[mask, :-1] = trial[mask]
        new_pop[mask, -1] = trial_fit[mask]
        for i in np.where(mask)[0]:
            labels[int(i)] = "agdo.progressive_gradient_momentum_dynamic_interaction"

        # System optimization operator. It is activated according to the paper's
        # logistic probability and greedily replaces only improved individuals.
        if np.random.rand() <= self._sigmoid(18.0 * tau - 12.0):
            cur = new_pop[:, :-1].copy()
            cur_fit = new_pop[:, -1].copy()
            order = self._order(cur_fit)
            half = max(1, n // 2)
            superior = cur[np.random.choice(order[:half], size=n, replace=True)]
            delta = np.random.rand(n, 1) * w_base
            theta = 2.0 * tau
            levy = np.array([levy_flight(dim, self._levy_beta, 1.0) for _ in range(n)])
            trial2 = superior + levy * delta * (superior - cur * theta)
            trial2 = np.clip(trial2, self._lo, self._hi)
            fit2 = self._evaluate_population(trial2)
            evals += n
            mask2 = self._better_mask(fit2, cur_fit)
            new_pop[mask2, :-1] = trial2[mask2]
            new_pop[mask2, -1] = fit2[mask2]
            for i in np.where(mask2)[0]:
                labels[int(i)] = "agdo.system_optimization_operator"

        movement = np.linalg.norm(new_pop[:, :-1] - old_x, axis=1)
        labels = [lab if mv > 0.0 else "carryover" for lab, mv in zip(labels, movement)]
        return new_pop, evals, {
            "agdo_m": m,
            "agdo_v": v,
            "operator_labels": labels,
            "native_evomapx_operator_labels": True,
            "agdo_tau": float(tau),
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["agdo_tau"] = float(state.payload.get("agdo_tau", 0.0))
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_fidelity", "native_labels")
        return obs
