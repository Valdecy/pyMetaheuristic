"""pyMetaheuristic src — New Caledonian Crow Learning Algorithm Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class NCCLAEngine(PortedPopulationEngine):
    """New Caledonian Crow Learning Algorithm — social/individual crow learning."""

    algorithm_id = "nccla"
    algorithm_name = "New Caledonian Crow Learning Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.asoc.2020.106325",
        "source": "Al-Sorori, W.; Mohsen, A.M. (2020)New Caledonian crow learning algorithm: A new metaheuristic algorithm for solving continuous optimization problems, Applied Soft Computing, Volume 92, 106325, ISSN 1568-4946.",
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
    )
    _DEFAULTS = dict(
        population_size=30,
        social_learning_probability=0.75,
        vertical_learning_probability=0.50,
        trial_probability=0.25,
        learning_factor_min=0.10,
        learning_factor_max=1.00,
        juvenile_reinforcement_scale=0.05,
        parent_reinforcement_scale=0.10,
        attribute_copy_probability=0.50,
    )

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._sl_prob = self._validate_probability("social_learning_probability")
        self._vsl_prob = self._validate_probability("vertical_learning_probability")
        self._trial_prob = self._validate_probability("trial_probability")
        self._copy_prob = self._validate_probability("attribute_copy_probability")
        self._lf_min = float(self._params.get("learning_factor_min", 0.10))
        self._lf_max = float(self._params.get("learning_factor_max", 1.00))
        self._juvenile_scale = float(self._params.get("juvenile_reinforcement_scale", 0.05))
        self._parent_scale = float(self._params.get("parent_reinforcement_scale", 0.10))
        if self._lf_min < 0.0 or self._lf_max < 0.0:
            raise ValueError("learning_factor_min and learning_factor_max must be non-negative.")
        if self._lf_max < self._lf_min:
            raise ValueError("learning_factor_max must be greater than or equal to learning_factor_min.")
        if self._juvenile_scale < 0.0 or self._parent_scale < 0.0:
            raise ValueError("reinforcement scales must be non-negative.")

    def _validate_probability(self, name: str) -> float:
        value = float(self._params.get(name, self._DEFAULTS[name]))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

    def _copy_mask(self, dim: int) -> np.ndarray:
        mask = np.random.rand(dim) < self._copy_prob
        if not bool(np.any(mask)):
            mask[np.random.randint(dim)] = True
        return mask

    def _learning_factor(self, step: int) -> float:
        total = max(1, int(self.config.max_steps or 500))
        ratio = min(1.0, max(0.0, float(step) / float(total)))
        return self._lf_min + (self._lf_max - self._lf_min) * ratio

    def _juvenile_reinforcement(self, learned: np.ndarray, previous: np.ndarray, rank: int, n: int, lf: float, ratio: float) -> np.ndarray:
        """Scaled Eq. (14)-(18) reinforcement for arbitrary continuous box bounds."""
        span = self._span
        norm_mean = np.clip((learned - self._lo) / span, 0.0, 1.0)
        a = np.abs(learned - previous) / span
        b = np.exp(np.clip(-lf * ratio * np.random.rand(self.problem.dimension) * norm_mean, -50.0, 50.0))
        if rank < n / 2.0:
            rd = b - a
        else:
            rd = np.random.rand(self.problem.dimension) * (np.random.rand(self.problem.dimension) * b) - a
        direction = np.where(np.random.rand(self.problem.dimension) < 0.5, -1.0, 1.0)
        return learned + direction * self._juvenile_scale * span * rd

    def _parent_reinforcement(self, parent: np.ndarray, family_mean: np.ndarray, best: np.ndarray, parent_rank: int, lf: float, ratio: float) -> np.ndarray:
        """Bounded parent reinforcement guided by family mean and elite experience."""
        decay = float(np.exp(-lf * ratio))
        if parent_rank == 0:
            direction = family_mean - parent
        else:
            direction = 0.5 * (family_mean - parent) + 0.5 * (best - parent)
        return parent + self._parent_scale * decay * np.random.rand(self.problem.dimension) * direction

    def _improves_nearest_previous(self, trial: np.ndarray, trial_fit: float, previous: np.ndarray) -> bool:
        """Keep EvoMapX parent-child attribution non-negative under nearest-parent matching."""
        if previous.size == 0:
            return True
        distances = np.linalg.norm(previous[:, :-1] - trial[None, :], axis=1)
        parent_idx = int(np.argmin(distances))
        parent_fit = float(previous[parent_idx, -1])
        return self._is_better(float(trial_fit), parent_fit) or abs(float(trial_fit) - parent_fit) <= 1.0e-14

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        step = state.step + 1
        total = max(1, int(self.config.max_steps or 500))
        ratio = min(1.0, float(step) / float(total))
        lf = self._learning_factor(step)

        # Crow family (CF) is maintained in fitness order; top two crows are parents.
        pop = pop[self._order(pop[:, -1])].copy()
        previous = pop.copy()
        family_mean = pop[:, :-1].mean(axis=0)
        best_pos = pop[0, :-1].copy()
        operator_labels = ["carryover"] * n
        evals = 0

        # Parent reinforcement, Eq. (19), implemented as a bounded movement using
        # the mean-family term present in the paper. Trials are greedily accepted
        # to avoid attributing negative direct contributions.
        parent_count = min(2, n)
        for rank in range(parent_count):
            old_pos = pop[rank, :-1].copy()
            old_fit = float(pop[rank, -1])
            trial = self._parent_reinforcement(old_pos, family_mean, best_pos, rank, lf, ratio)
            trial = np.clip(trial, self._lo, self._hi)
            trial_fit = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(trial_fit, old_fit) and self._improves_nearest_previous(trial, trial_fit, previous):
                pop[rank, :-1] = trial
                pop[rank, -1] = trial_fit
                operator_labels[rank] = "nccla.parent_reinforcement"

        # Juvenile learning: social vertical/horizontal copying or individual
        # learning, followed by the juvenile reinforcement step, Eq. (12)-(18).
        for rank in range(2, n):
            old_pos = pop[rank, :-1].copy()
            old_fit = float(pop[rank, -1])
            learned = old_pos.copy()
            if np.random.rand() < self._sl_prob:
                if rank <= 2 or np.random.rand() < self._vsl_prob:
                    demonstrator = int(np.random.randint(0, parent_count))
                    operator = "nccla.vertical_social_learning_juvenile_reinforcement"
                else:
                    demonstrator = int(np.random.randint(2, rank))
                    operator = "nccla.horizontal_social_learning_juvenile_reinforcement"
                mask = self._copy_mask(dim)
                learned[mask] = pop[demonstrator, :-1][mask]
            else:
                operator = "nccla.individual_learning_juvenile_reinforcement"
                mask = self._copy_mask(dim)
                if np.random.rand() < self._trial_prob:
                    learned[mask] = np.random.uniform(self._lo, self._hi, dim)[mask]
                else:
                    jitter = (np.random.rand(dim) - 0.5) * (1.0 - ratio) * 0.10 * self._span
                    learned = old_pos + jitter

            trial = self._juvenile_reinforcement(learned, old_pos, rank, n, lf, ratio)
            trial = np.clip(trial, self._lo, self._hi)
            trial_fit = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(trial_fit, old_fit) and self._improves_nearest_previous(trial, trial_fit, previous):
                pop[rank, :-1] = trial
                pop[rank, -1] = trial_fit
                operator_labels[rank] = operator

        # Sorting and parent selection for the next iteration.
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        operator_labels = [operator_labels[int(i)] for i in order]

        return pop, evals, {
            "operator_labels": operator_labels,
            "native_evomapx_operator_labels": True,
            "nccla_learning_factor": float(lf),
            "nccla_parent_indices": [0, 1] if n > 1 else [0],
            "nccla_previous_population": previous,
        }

    def observe(self, state):
        pop = state.payload["population"]
        obs = super().observe(state)
        obs["learning_factor"] = float(state.payload.get("nccla_learning_factor", self._learning_factor(state.step)))
        obs["parent_selection"] = "top_two_after_sorting"
        # The passive EvoMapX probe replaces this with parent-child lineage
        # values when enabled. This fallback keeps operator telemetry visible
        # if users run with evomapx=False.
        labels = state.payload.get("operator_labels", []) or []
        counts = {label: labels.count(label) for label in sorted(set(labels)) if label != "carryover"}
        if counts:
            obs.setdefault("operator_counts", counts)
            obs.setdefault("operator_contributions", {label: 0.0 for label in counts})
            obs.setdefault("evomapx_delta_f", "diagnostic_counts_only")
            obs.setdefault("evomapx_fidelity", "native_labels")
        obs["mean_fitness"] = float(np.mean(pop[:, -1]))
        obs["std_fitness"] = float(np.std(pop[:, -1]))
        return obs
