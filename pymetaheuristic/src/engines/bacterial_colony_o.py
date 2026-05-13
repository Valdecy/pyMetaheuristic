"""pyMetaheuristic src — Bacterial Colony Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class BacterialColonyOEngine(PortedPopulationEngine):
    """Bacterial Colony Optimization.

    Notes
    -----
    This is the colony-lifecycle BCO of Niu and Wang, not the existing
    ``bco`` engine in this package, which is Bacterial Chemotaxis Optimizer.
    The non-conflicting registry id is therefore ``bacterial_colony_o``.

    The Mealpy LTS source includes several stale/undefined variables in the
    reproduction and migration tail. This port keeps the documented lifecycle
    stages—chemotaxis, communication, reproduction, elimination, and migration—
    and implements those stages with the package's population matrix protocol.
    """

    algorithm_id = "bacterial_colony_o"
    algorithm_name = "Bacterial Colony Optimization"
    family = "nature"
    _REFERENCE = {
        "doi": "10.1155/2012/698057",
        "title": "Bacterial Colony Optimization",
        "authors": "Ben Niu, Hong Wang",
        "year": 2012,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=50,
        c_min=0.01,
        c_max=0.2,
        n_chemotaxis=1,
        max_swim_steps=4,
        energy_threshold=0.5,
        migration_prob=0.1,
        turbulence_scale=0.1,
        offspring_scale=0.1,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 5:
            raise ValueError("bacterial_colony_o requires population_size >= 5.")
        c_min = float(self._params.get("c_min", 0.01))
        c_max = float(self._params.get("c_max", 0.2))
        if not 0.0 < c_min < c_max:
            raise ValueError("bacterial_colony_o requires 0 < c_min < c_max.")
        if c_max > 10.0:
            raise ValueError("bacterial_colony_o c_max must be <= 10.")
        if int(self._params.get("n_chemotaxis", 1)) < 1:
            raise ValueError("bacterial_colony_o n_chemotaxis must be >= 1.")
        if int(self._params.get("max_swim_steps", 4)) < 1:
            raise ValueError("bacterial_colony_o max_swim_steps must be >= 1.")
        for key in ("energy_threshold", "migration_prob"):
            value = float(self._params.get(key))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"bacterial_colony_o {key} must be in [0, 1].")
        for key in ("turbulence_scale", "offspring_scale"):
            if float(self._params.get(key)) < 0.0:
                raise ValueError(f"bacterial_colony_o {key} must be non-negative.")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"local_best": pop.copy()}

    def _energy(self, fit: np.ndarray) -> np.ndarray:
        """Return normalized bacterial energy, where larger means healthier."""
        return self._quality(fit)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, int(self.config.max_steps or 500))
        t = min(T, int(state.step) + 1)
        c_min = float(self._params.get("c_min", 0.01))
        c_max = float(self._params.get("c_max", 0.2))
        n_chemotaxis = int(self._params.get("n_chemotaxis", 1))
        swim_steps = int(self._params.get("max_swim_steps", 4))
        energy_threshold = float(self._params.get("energy_threshold", 0.5))
        migration_prob = float(self._params.get("migration_prob", 0.1))
        turbulence_scale = float(self._params.get("turbulence_scale", 0.1))
        offspring_scale = float(self._params.get("offspring_scale", 0.1))

        local_best = np.asarray(state.payload.get("local_best", pop.copy()), dtype=float).copy()
        if local_best.shape != pop.shape:
            local_best = pop.copy()

        order = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        step_size = c_min + (c_max - c_min) * (1.0 - t / T) ** n_chemotaxis
        evals = 0

        # Chemotaxis and communication: blend personal and global directions,
        # then greedily keep successful tumbles/swims.
        trial_pos = np.empty((n, dim), dtype=float)
        for i in range(n):
            fi = np.random.random()
            personal_direction = local_best[i, :-1] - pop[i, :-1]
            global_direction = best_pos - pop[i, :-1]
            direction = fi * global_direction + (1.0 - fi) * personal_direction
            turbulent = np.random.normal(0.0, turbulence_scale, dim) * self._span
            pos_new = pop[i, :-1] + step_size * (direction + turbulent)

            # Swim repeatedly in the non-turbulent direction, as in the source
            # implementation, but only as a bounded macro-step.
            for _ in range(max(0, swim_steps - 1)):
                pos_new = pos_new + step_size * direction
            trial_pos[i] = np.clip(pos_new, self._lo, self._hi)

        trial_fit = self._evaluate_population(trial_pos)
        evals += n
        improved = self._better_mask(trial_fit, pop[:, -1])
        pop[improved, :-1] = trial_pos[improved]
        pop[improved, -1] = trial_fit[improved]
        better_local = self._better_mask(pop[:, -1], local_best[:, -1])
        local_best[better_local] = pop[better_local]

        # Interactive exchange between bacteria.
        for i in range(n):
            if np.random.random() < 0.5:
                if np.random.random() < 0.5:
                    if i == 0:
                        neighbor = 1
                    elif i == n - 1:
                        neighbor = n - 2
                    else:
                        neighbor = i + 1 if np.random.random() < 0.5 else i - 1
                else:
                    neighbor = int(np.random.choice([j for j in range(n) if j != i]))
                if self._is_better(pop[neighbor, -1], pop[i, -1]):
                    pop[i] = pop[neighbor].copy()
            else:
                # Group exchange: move a weak bacterium slightly toward the
                # current colony best and accept only if it improves.
                if not self._is_better(pop[i, -1], pop[order[0], -1]):
                    candidate = pop[i, :-1] + 0.1 * (best_pos - pop[i, :-1])
                    candidate = np.clip(candidate, self._lo, self._hi)
                    fit = float(self.problem.evaluate(candidate))
                    evals += 1
                    if self._is_better(fit, pop[i, -1]):
                        pop[i, :-1] = candidate
                        pop[i, -1] = fit

        # Reproduction and elimination using normalized energy.
        energy = self._energy(pop[:, -1])
        sorted_indices = np.argsort(energy)[::-1]
        n_reproduce = max(1, n // 2)
        n_eliminate = max(1, n // 4)
        reproduction_candidates = sorted_indices[:n_reproduce]
        weak = sorted_indices[energy[sorted_indices] <= energy_threshold]
        elimination_candidates = weak[-n_eliminate:] if weak.size else sorted_indices[-n_eliminate:]
        for k, child_idx in enumerate(elimination_candidates[:n_eliminate]):
            parent_idx = reproduction_candidates[k % reproduction_candidates.size]
            child = pop[parent_idx, :-1] + np.random.normal(0.0, offspring_scale, dim) * self._span
            child = np.clip(child, self._lo, self._hi)
            fit = float(self.problem.evaluate(child))
            evals += 1
            pop[child_idx, :-1] = child
            pop[child_idx, -1] = fit

        # Migration is triggered by low positional diversity or random pressure.
        diversity = float(np.mean(np.var(pop[:, :-1], axis=0)))
        scale = float(np.mean(self._span ** 2)) + 1.0e-30
        if diversity / scale < 0.01 or np.random.random() < migration_prob:
            n_migrate = max(1, n // 10)
            migrate_idx = np.random.choice(n, size=n_migrate, replace=False)
            migrants = np.random.uniform(self._lo, self._hi, size=(n_migrate, dim))
            migrant_fit = self._evaluate_population(migrants)
            evals += n_migrate
            pop[migrate_idx, :-1] = migrants
            pop[migrate_idx, -1] = migrant_fit

        better_local = self._better_mask(pop[:, -1], local_best[:, -1])
        local_best[better_local] = pop[better_local]
        return pop, evals, {"local_best": local_best}
