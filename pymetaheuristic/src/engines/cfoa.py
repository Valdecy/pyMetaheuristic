"""pyMetaheuristic src — Catch Fish Optimization Algorithm Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class CFOAEngine(PortedPopulationEngine):
    """Catch Fish Optimization Algorithm (CFOA).

    Native NumPy port of the MATLAB reference implementation. The native
    macro-step evaluates one candidate population and then applies the paper's
    independent-search / group-capture / collective-capture movement rules.
    """

    algorithm_id = "cfoa"
    algorithm_name = "Catch Fish Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10586-024-04618-w",
        "authors": "Heming Jia, Qixian Wen, Yuhao Wang, Seyedali Mirjalili",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("CFOA requires population_size >= 2.")

    def _cost_view(self, fitness: np.ndarray) -> np.ndarray:
        fitness = np.asarray(fitness, dtype=float)
        return fitness if self.problem.objective == "min" else -fitness

    def _evaluation_budget(self, state) -> int:
        if self.config.max_evaluations is not None:
            return max(1, int(self.config.max_evaluations))
        if self.config.max_steps is not None:
            return max(1, int(self.config.max_steps) * self._n)
        return max(1000, 1000 * self.problem.dimension)

    @staticmethod
    def _safe_group_weights(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        total = float(np.sum(values))
        if np.isfinite(total) and abs(total) > 1.0e-12:
            return values / total
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=float)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        budget = self._evaluation_budget(state)
        eval_ratio = min(1.0, float(state.evaluations) / float(budget))

        # The MATLAB code first evaluates newFisher and greedily accepts it.
        # In the pymh protocol, pop already stores the current evaluated Fisher.
        fisher = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        costs = self._cost_view(fit)
        best_pos = np.asarray(state.best_position, dtype=float)
        best_cost = float(np.min(costs))

        new_fisher = fisher.copy()
        pos_order = np.random.permutation(n)

        if state.evaluations < budget / 2.0:
            alpha_base = max(0.0, 1.0 - 3.0 * eval_ratio / 2.0)
            alpha_exp = max(0.0, 3.0 * eval_ratio / 2.0)
            alpha = float(alpha_base ** alpha_exp) if alpha_base > 0.0 else 0.0
            p = float(np.random.random())
            i = 0
            denom = float(np.max(costs) - best_cost)
            if abs(denom) <= 1.0e-12 or not np.isfinite(denom):
                denom = 1.0e-12

            while i < n:
                group_size = int(np.random.randint(3, 5))
                if p < alpha or i + group_size > n:
                    current = int(pos_order[i])
                    r = int(np.random.randint(0, n))
                    while r == current and n > 1:
                        r = int(np.random.randint(0, n))
                    exp_term = float((costs[current] - costs[r]) / denom)
                    rs = np.random.random(dim) * 2.0 - 1.0
                    rs_norm = float(np.linalg.norm(rs))
                    if rs_norm <= 1.0e-12:
                        rs = np.ones(dim, dtype=float) / np.sqrt(dim)
                        rs_norm = 1.0
                    radius = float(np.linalg.norm(fisher[r] - fisher[current]) * np.random.random() * (1.0 - eval_ratio))
                    rs = radius * rs / rs_norm
                    new_fisher[current] = (
                        fisher[current]
                        + (fisher[r] - fisher[current]) * exp_term
                        + np.sqrt(abs(exp_term)) * rs
                    )
                    i += 1
                else:
                    ids = pos_order[i : i + group_size]
                    weights = self._safe_group_weights(costs[ids]).reshape(-1, 1)
                    aim = np.sum(weights * fisher[ids], axis=0)
                    new_fisher[ids] = (
                        fisher[ids]
                        + np.random.random((group_size, 1)) * (aim - fisher[ids])
                        + (1.0 - 2.0 * eval_ratio) * (np.random.random((group_size, dim)) * 2.0 - 1.0)
                    )
                    i += group_size
        else:
            ratio = max(0.0, 1.0 - eval_ratio)
            sigma = float(np.sqrt(max(0.0, 2.0 * ratio / (ratio * ratio + 1.0))))
            center = np.mean(fisher, axis=0)
            for i in range(n):
                W = np.abs(best_pos - center) * (int(np.random.randint(1, 4)) / 3.0) * sigma
                new_fisher[i] = best_pos + np.random.normal(0.0, W, dim)

        new_fisher = np.clip(new_fisher, self._lo, self._hi)
        new_fit = self._evaluate_population(new_fisher)
        mask = self._better_mask(new_fit, fit)
        pop[mask, :-1] = new_fisher[mask]
        pop[mask, -1] = new_fit[mask]
        return pop, n, {}
