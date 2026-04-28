"""pyMetaheuristic src — Artificial Algae Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class AAAEngine(PortedPopulationEngine):
    """
    Artificial Algae Algorithm — helical movement, reproduction, and adaptation.

    The implementation follows the main operators described by Uymaz, Tezel,
    and Yel: algal colonies move toward a selected light source by helical
    movement, colony sizes are updated through a Monod-style growth signal,
    the smallest colony inherits one cell from the largest colony, and the most
    starving colony may adapt toward the largest colony.
    """

    algorithm_id = "aaa"
    algorithm_name = "Artificial Algae Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.asoc.2015.03.003",
        "title": "Artificial algae algorithm (AAA) for nonlinear global optimization",
        "authors": "Sait Ali Uymaz, Gulay Tezel, Esra Yel",
        "year": 2015,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=40,
        energy_loss=0.3,
        shear_force=2.0,
        adaptation_probability=0.5,
        half_saturation=0.5,
        initial_energy=1.0,
        tournament_size=2,
        max_moves_per_colony=10,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("aaa requires population_size >= 2.")
        if float(self._params.get("energy_loss", 0.3)) <= 0.0:
            raise ValueError("aaa energy_loss must be positive.")
        if float(self._params.get("shear_force", 2.0)) <= 0.0:
            raise ValueError("aaa shear_force must be positive.")
        ap = float(self._params.get("adaptation_probability", 0.5))
        if not 0.0 <= ap <= 1.0:
            raise ValueError("aaa adaptation_probability must be in [0, 1].")
        if float(self._params.get("half_saturation", 0.5)) <= 0.0:
            raise ValueError("aaa half_saturation must be positive.")
        if float(self._params.get("initial_energy", 1.0)) <= 0.0:
            raise ValueError("aaa initial_energy must be positive.")
        if int(self._params.get("tournament_size", 2)) < 1:
            raise ValueError("aaa tournament_size must be at least 1.")
        if int(self._params.get("max_moves_per_colony", 10)) < 1:
            raise ValueError("aaa max_moves_per_colony must be at least 1.")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        quality = self._quality(pop[:, -1])
        colony_size = self._normalize_colony_size(0.25 + quality)
        return {
            "colony_size": colony_size,
            "starvation": np.zeros(n, dtype=float),
        }

    @staticmethod
    def _normalize_colony_size(colony_size: np.ndarray) -> np.ndarray:
        colony_size = np.asarray(colony_size, dtype=float).reshape(-1)
        colony_size = np.maximum(colony_size, 1.0e-12)
        max_size = float(np.max(colony_size))
        if max_size <= 0.0:
            return np.ones_like(colony_size)
        return np.clip(colony_size / max_size, 1.0e-12, 1.0)

    @staticmethod
    def _friction_surface(size: float) -> float:
        # Hemisphere surface derived from colony volume G:
        # tau(x_i) = 2*pi*((3*G_i)/(4*pi))^(2/3).
        radius = ((3.0 * max(float(size), 1.0e-12)) / (4.0 * np.pi)) ** (1.0 / 3.0)
        return float(2.0 * np.pi * radius * radius)

    def _select_light_source(self, pop: np.ndarray, index: int, tournament_size: int) -> int:
        n = pop.shape[0]
        pool = np.array([j for j in range(n) if j != index], dtype=int)
        if pool.size == 0:
            return index
        k = min(max(1, tournament_size), pool.size)
        candidates = np.random.choice(pool, size=k, replace=False)
        local_order = self._order(pop[candidates, -1])
        return int(candidates[local_order[0]])

    def _helical_trial(self, current: np.ndarray, source: np.ndarray, step_factor: float) -> np.ndarray:
        dim = self.problem.dimension
        trial = np.asarray(current, dtype=float).copy()
        direction = source - current

        dims = np.random.permutation(dim)
        p = np.random.uniform(-1.0, 1.0)
        beta = np.random.uniform(0.0, 2.0 * np.pi)

        # Eq. (16): linear component. Used for all dimensions, including D=1.
        m = int(dims[0])
        trial[m] = current[m] + direction[m] * step_factor * p

        if dim >= 2:
            # Eq. (18): sinusoidal component for 2-D and higher-dimensional cases.
            l = int(dims[1])
            trial[l] = current[l] + direction[l] * step_factor * np.sin(beta)

        if dim >= 3:
            # Eq. (17): cosine component for the third helical coordinate.
            alpha = np.random.uniform(0.0, 2.0 * np.pi)
            k = int(dims[2])
            trial[k] = current[k] + direction[k] * step_factor * np.cos(alpha)

        return np.clip(trial, self._lo, self._hi)

    def _step_impl(self, state, pop: np.ndarray):
        n = pop.shape[0]
        energy_loss = float(self._params.get("energy_loss", 0.3))
        shear_force = float(self._params.get("shear_force", 2.0))
        adaptation_probability = float(self._params.get("adaptation_probability", 0.5))
        half_saturation = float(self._params.get("half_saturation", 0.5))
        initial_energy = float(self._params.get("initial_energy", 1.0))
        tournament_size = int(self._params.get("tournament_size", 2))
        max_moves = int(self._params.get("max_moves_per_colony", 10))

        colony_size = self._normalize_colony_size(
            np.asarray(state.payload.get("colony_size", np.ones(n)), dtype=float)
        )
        starvation = np.asarray(state.payload.get("starvation", np.zeros(n)), dtype=float).copy()

        new_pop = pop.copy()
        quality = self._quality(pop[:, -1])
        evals = 0

        # Helical movement phase.
        for i in range(n):
            current = new_pop[i, :-1].copy()
            current_fit = float(new_pop[i, -1])
            energy = initial_energy * (0.1 + float(quality[i]))
            improved = False
            moves = 0

            friction = self._friction_surface(colony_size[i])
            step_factor = max(0.0, shear_force - min(friction, shear_force))

            while energy > 0.0 and moves < max_moves:
                source_idx = self._select_light_source(new_pop, i, tournament_size)
                source = new_pop[source_idx, :-1]
                trial = self._helical_trial(current, source, step_factor)
                trial_fit = float(self.problem.evaluate(trial))
                evals += 1
                moves += 1

                # Movement consumes half of the loss. Failed metabolism consumes
                # the other half, following the pseudo-code distinction.
                energy -= energy_loss / 2.0
                if self._is_better(trial_fit, current_fit):
                    current = trial
                    current_fit = trial_fit
                    improved = True
                else:
                    energy -= energy_loss / 2.0

            new_pop[i, :-1] = current
            new_pop[i, -1] = current_fit
            if improved:
                starvation[i] = max(0.0, 0.5 * starvation[i])
            else:
                starvation[i] += 1.0

        # Monod-style evolutionary growth signal. Raw objective values can be
        # negative or maximized, so the package's normalized quality signal is
        # used as the nutrient proxy.
        nutrient = self._quality(new_pop[:, -1])
        growth_rate = nutrient / (half_saturation + nutrient + 1.0e-12)
        colony_size = self._normalize_colony_size(colony_size * (1.0 + growth_rate))

        # Reproduction phase: one random cell/dimension of the smallest colony
        # is replaced by the corresponding cell of the biggest colony.
        biggest = int(np.argmax(colony_size))
        smallest = int(np.argmin(colony_size))
        if biggest != smallest:
            dim_idx = int(np.random.randint(self.problem.dimension))
            candidate = new_pop[smallest, :-1].copy()
            candidate[dim_idx] = new_pop[biggest, dim_idx]
            candidate = np.clip(candidate, self._lo, self._hi)
            candidate_fit = float(self.problem.evaluate(candidate))
            evals += 1
            new_pop[smallest, :-1] = candidate
            new_pop[smallest, -1] = candidate_fit
            starvation[smallest] += 0.5
            colony_size[smallest] = max(colony_size[smallest], 1.0e-12)

        # Adaptation phase: the most starving colony moves toward the biggest.
        if np.random.random() < adaptation_probability:
            starving = int(np.argmax(starvation))
            biggest = int(np.argmax(colony_size))
            if starving != biggest:
                candidate = new_pop[starving, :-1] + np.random.random(self.problem.dimension) * (
                    new_pop[biggest, :-1] - new_pop[starving, :-1]
                )
                candidate = np.clip(candidate, self._lo, self._hi)
                candidate_fit = float(self.problem.evaluate(candidate))
                evals += 1
                new_pop[starving, :-1] = candidate
                new_pop[starving, -1] = candidate_fit
                starvation[starving] = 0.0

        return new_pop, evals, {"colony_size": colony_size, "starvation": starvation}
