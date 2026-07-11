"""Python Snake Optimization Algorithm (PySOA).

Paper-faithful NumPy implementation of the population-based optimizer proposed
by Diab, Darwish, Oliva, and Hosny.  PySOA models three hunting outcomes:

1. Searching for prey when pheromone density is low (Eq. 11).
2. Attacking detected prey when density is high (Eq. 10).
3. Redirecting through a random python when the prey detects the hunter
   (Eq. 12).

The infrared submodel follows the authors' released MATLAB implementation.  In
particular, it uses a 32 micrometre wavelength and multiplies ambient
temperature by lighting in the receptor-transparency term.  These details differ
slightly from the prose surrounding Eqs. (8)-(9), but reproduce the executable
reference implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._ported_common import PortedPopulationEngine
from .protocol import CapabilityProfile


class PySOAEngine(PortedPopulationEngine):
    """PySOA, a python-snake hunting-inspired population optimizer."""

    algorithm_id = "pysoa"
    algorithm_name = "Python Snake Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10586-026-05958-5",
        "title": "PySOA: a novel bio-inspired python snake optimization algorithm",
        "authors": "Mahmoud S. Diab, Mohamed M. Darwish, Diego Oliva, and Khalid M. Hosny",
        "year": 2026,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        max_distance=100.0,
        prey_temperature_f=104.0,
        infrared_wavelength=32.0e-6,
        infrared_reference=745.4497,
        cosine_epsilon=1.0e-12,
    )

    _OPERATOR_LABELS = (
        "pysoa.searching_for_prey",
        "pysoa.attacking_prey",
        "pysoa.random_agent_redirection",
        "pysoa.sensory_scanning",
        "pysoa.temperature_cooling",
    )
    _DIRECT_OPERATORS = _OPERATOR_LABELS[:3]
    _DIAGNOSTIC_OPERATORS = _OPERATOR_LABELS[3:]

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._max_distance = float(self._params.get("max_distance", 100.0))
        self._prey_temperature_f = float(
            self._params.get("prey_temperature_f", 104.0)
        )
        self._infrared_wavelength = float(
            self._params.get("infrared_wavelength", 32.0e-6)
        )
        self._infrared_reference = float(
            self._params.get("infrared_reference", 745.4497)
        )
        self._cosine_epsilon = float(
            self._params.get("cosine_epsilon", 1.0e-12)
        )
        self._validate_parameters()
        self._last_operator_contributions = self._blank_contributions()
        self._last_operator_counts = self._blank_counts()
        self._last_sensor_statistics = {
            "temperature": 1.0,
            "mean_pheromone_density": 0.0,
            "mean_abs_sight": 0.0,
            "mean_infrared": 1.0,
        }

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("PySOA requires population_size >= 2.")
        values = {
            "max_distance": self._max_distance,
            "prey_temperature_f": self._prey_temperature_f,
            "infrared_wavelength": self._infrared_wavelength,
            "infrared_reference": self._infrared_reference,
            "cosine_epsilon": self._cosine_epsilon,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number.")

    def _blank_contributions(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _initialize_payload(self, pop: np.ndarray) -> dict[str, Any]:
        return {
            "operator_contributions": self._blank_contributions(),
            "operator_counts": self._blank_counts(),
            "operator_labels": ["pysoa.initialization"] * int(pop.shape[0]),
            "evomapx_fidelity": "native",
            "temperature": 1.0,
            "mean_pheromone_density": 0.0,
            "mean_abs_sight": 0.0,
            "mean_infrared": 1.0,
        }

    def _horizon(self) -> int:
        # The reference implementation uses maxIter in the temperature schedule.
        return max(1, int(self.config.max_steps or 500))

    def _remaining_evaluations(self, state) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(0, int(self.config.max_evaluations) - int(state.evaluations))

    def _infrared_radiation(self, lighting: float, temperature: float) -> float:
        """Return normalized infrared response from the released MATLAB code."""
        h = 6.62607015e-34
        c = 2.998e8
        # The authors convert Boltzmann's constant from kelvin to Fahrenheit.
        kb = 1.380649e-23 * (9.0 / 5.0)
        wavelength = self._infrared_wavelength
        c1 = 2.0 * h * c * c
        c2 = (h * c) / (wavelength * kb)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            denominator = np.expm1(c2 / self._prey_temperature_f)
            emitted = (c1 / wavelength**5) / denominator
            absorbed = float(temperature) * float(lighting) * emitted
            normalized = 1.0 + 9.0 * absorbed / self._infrared_reference
        if not np.isfinite(normalized) or normalized <= 0.0:
            return 1.0
        return float(normalized)

    def _sight(self, distance: float, transparency: float, lighting: float) -> float:
        if distance > self._max_distance:
            return 0.0
        cosine = float(np.cos(distance))
        if abs(cosine) < self._cosine_epsilon:
            cosine = self._cosine_epsilon if cosine >= 0.0 else -self._cosine_epsilon
        return float(transparency * lighting / cosine)

    def _gain(self, old_fitness: float, new_fitness: float) -> float:
        if self.problem.objective == "min":
            return float(max(0.0, old_fitness - new_fitness))
        return float(max(0.0, new_fitness - old_fitness))

    def _step_impl(self, state, pop: np.ndarray):
        n = int(pop.shape[0])
        dim = int(self.problem.dimension)
        remaining = self._remaining_evaluations(state)
        if remaining == 0:
            contributions = self._blank_contributions()
            counts = self._blank_counts()
            self._last_operator_contributions = contributions
            self._last_operator_counts = counts
            return pop, 0, {
                "operator_contributions": contributions,
                "operator_counts": counts,
                "operator_labels": ["carryover"] * n,
                "evomapx_fidelity": "native",
            }

        horizon = self._horizon()
        # The MATLAB loop begins with t=0, so the first update uses Temp=1.
        temperature = max(0.0, 1.0 - float(state.step) / float(horizon))
        old_positions = pop[:, :-1].copy()
        old_fitness = pop[:, -1].copy()
        best = np.asarray(state.best_position, dtype=float).copy()
        distances = np.abs(best[None, :] - old_positions)

        new_positions = old_positions.copy()
        operator_by_coordinate = np.full((n, dim), "carryover", dtype=object)
        active_n = n if remaining is None else min(n, int(max(0, remaining)))
        pheromone_values: list[float] = []
        sight_values: list[float] = []
        infrared_values: list[float] = []
        counts = self._blank_counts()
        contributions = self._blank_contributions()
        counts["pysoa.temperature_cooling"] = 1

        for i in range(active_n):
            humidity = float(np.random.random())
            transparency = float(np.random.random())
            lighting = float(np.random.random())
            air_pollution = float(np.random.random())
            infrared = self._infrared_radiation(lighting, temperature)
            log_ir = float(np.log10(infrared))
            infrared_values.append(infrared)

            for j in range(dim):
                hidden = int(np.random.randint(0, 2))
                distance = float(distances[i, j])
                pheromone = float(
                    temperature / (distance + np.exp(humidity + air_pollution))
                )
                sight = self._sight(distance, transparency, lighting)
                pheromone_values.append(pheromone)
                sight_values.append(abs(sight))
                counts["pysoa.sensory_scanning"] += 1

                if hidden == 0:
                    if abs(pheromone) < 0.5:
                        direction = 1.0 if int(np.random.randint(0, 2)) == 1 else -1.0
                        candidate = (
                            best[j]
                            + direction * float(np.random.random()) * old_positions[i, j]
                        )
                        label = "pysoa.searching_for_prey"
                    else:
                        candidate = (
                            best[j]
                            - pheromone * log_ir * sight * old_positions[i, j]
                        )
                        label = "pysoa.attacking_prey"
                else:
                    random_agent = int(
                        np.rint(1.0 + float(n - 1) * float(np.random.random()))
                    ) - 1
                    random_agent = int(np.clip(random_agent, 0, n - 1))
                    candidate = (
                        best[j]
                        - pheromone
                        * log_ir
                        * sight
                        * new_positions[random_agent, j]
                    )
                    label = "pysoa.random_agent_redirection"

                if not np.isfinite(candidate):
                    candidate = old_positions[i, j]
                new_positions[i, j] = candidate
                operator_by_coordinate[i, j] = label
                counts[label] += 1

        # Boundary handling follows the reference implementation's clipping.
        new_positions = np.clip(new_positions, self._lo, self._hi)
        evaluated = active_n
        new_fitness = old_fitness.copy()
        if evaluated > 0:
            new_fitness[:evaluated] = self._evaluate_population(
                new_positions[:evaluated]
            )

        for i in range(evaluated):
            gain = self._gain(float(old_fitness[i]), float(new_fitness[i]))
            if gain <= 0.0:
                continue
            labels, label_counts = np.unique(
                operator_by_coordinate[i], return_counts=True
            )
            for label, count in zip(labels.tolist(), label_counts.tolist()):
                if label in contributions:
                    contributions[label] += gain * float(count) / float(dim)

        # Diagnostic operators are deliberately assigned zero direct gain.
        contributions["pysoa.sensory_scanning"] = 0.0
        contributions["pysoa.temperature_cooling"] = 0.0

        new_pop = np.hstack((new_positions, new_fitness[:, None]))
        operator_labels: list[str] = []
        for i in range(n):
            labels, label_counts = np.unique(
                operator_by_coordinate[i], return_counts=True
            )
            operator_labels.append(str(labels[int(np.argmax(label_counts))]))

        statistics = {
            "temperature": float(temperature),
            "mean_pheromone_density": float(np.mean(pheromone_values))
            if pheromone_values
            else 0.0,
            "mean_abs_sight": float(np.mean(sight_values)) if sight_values else 0.0,
            "mean_infrared": float(np.mean(infrared_values))
            if infrared_values
            else 1.0,
        }
        self._last_operator_contributions = {
            key: float(value) for key, value in contributions.items()
        }
        self._last_operator_counts = {key: int(value) for key, value in counts.items()}
        self._last_sensor_statistics = statistics
        return new_pop, evaluated, {
            "operator_contributions": dict(self._last_operator_contributions),
            "operator_counts": dict(self._last_operator_counts),
            "operator_labels": operator_labels,
            "evomapx_fidelity": "native",
            **statistics,
        }

    def observe(self, state) -> dict[str, Any]:
        observation = super().observe(state)
        observation.update(
            {
                "temperature": float(
                    state.payload.get(
                        "temperature", self._last_sensor_statistics["temperature"]
                    )
                ),
                "mean_pheromone_density": float(
                    state.payload.get(
                        "mean_pheromone_density",
                        self._last_sensor_statistics["mean_pheromone_density"],
                    )
                ),
                "mean_abs_sight": float(
                    state.payload.get(
                        "mean_abs_sight",
                        self._last_sensor_statistics["mean_abs_sight"],
                    )
                ),
                "mean_infrared": float(
                    state.payload.get(
                        "mean_infrared",
                        self._last_sensor_statistics["mean_infrared"],
                    )
                ),
                "operator_contributions": dict(
                    state.payload.get(
                        "operator_contributions",
                        self._last_operator_contributions,
                    )
                ),
                "operator_counts": dict(
                    state.payload.get("operator_counts", self._last_operator_counts)
                ),
                "operator_labels": list(
                    state.payload.get("operator_labels", [])
                ),
                "evomapx_fidelity": "native",
            }
        )
        return observation

    def finalize(self, state):
        result = super().finalize(state)
        result.metadata.update(
            {
                "reference": dict(self._REFERENCE),
                "evomapx_fidelity": "native",
                "direct_operators": list(self._DIRECT_OPERATORS),
                "diagnostic_operators": list(self._DIAGNOSTIC_OPERATORS),
                "operator_contributions": dict(
                    state.payload.get(
                        "operator_contributions",
                        self._last_operator_contributions,
                    )
                ),
                "operator_counts": dict(
                    state.payload.get("operator_counts", self._last_operator_counts)
                ),
            }
        )
        return result
