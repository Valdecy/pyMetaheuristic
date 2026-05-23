"""pyMetaheuristic src — Animated Oat Optimization Algorithm Engine"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class AOOEngine(PortedPopulationEngine):
    """Animated Oat Optimization Algorithm.

    Population-based optimizer inspired by animated oat seed propagation. This
    port follows the public MATLAB AOOv4 equations: wind disturbance, rolling
    displacement, jumping displacement, Levy-flight perturbation, and direct
    boundary clipping after each macro-iteration.
    """

    algorithm_id = "aoo"
    algorithm_name = "Animated Oat Optimization Algorithm"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.knosys.2025.113589",
        "title": "The Animated Oat Optimization Algorithm: A nature-inspired metaheuristic for engineering optimization and a case study on wireless sensor networks",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, levy_beta=1.5)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        beta = float(self._params.get("levy_beta", 1.5))
        if not (1.0 < beta < 2.0):
            raise ValueError("AOO parameter 'levy_beta' must satisfy 1 < levy_beta < 2.")
        self._levy_beta = beta

    @staticmethod
    def _levy_matrix(n: int, dim: int, beta: float) -> np.ndarray:
        """Mantegna Levy step matrix used by the original MATLAB AOO source."""
        numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        denominator = math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
        sigma_u = (numerator / denominator) ** (1.0 / beta)
        u = np.random.normal(0.0, sigma_u, (int(n), int(dim)))
        v = np.random.normal(0.0, 1.0, (int(n), int(dim)))
        return u / (np.abs(v) ** (1.0 / beta) + 1.0e-12)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        dim = max(1, self.problem.dimension)
        mass = 0.5 * np.random.random(n) / dim
        return {
            "x_coeff": 3.0 * np.random.random(n) / dim,
            "mass": mass,
            "length": n * np.random.random(n) / dim,
            "elasticity": mass.copy(),
            "gravity": 9.8 / dim,
        }

    def _bound_amplitude(self) -> np.ndarray:
        # The MATLAB implementation assumes symmetric bounds and uses ub as an
        # absolute movement amplitude. For arbitrary box bounds, the closest
        # scale-preserving interpretation is the larger absolute bound per axis.
        amp = np.maximum(np.abs(self._lo), np.abs(self._hi))
        return np.where(amp <= 1.0e-12, self._span, amp)

    def _time_horizon(self, state) -> int:
        if self.config.max_steps is not None:
            return max(1, int(self.config.max_steps))
        if self.config.max_evaluations is not None:
            remaining = max(1, int(self.config.max_evaluations) - int(state.evaluations))
            return max(1, int(math.ceil(remaining / max(1, self._n))))
        return 500

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        horizon = self._time_horizon(state)
        progress = min(1.0, float(t) / float(horizon))
        c = (1.0 - progress) ** 3

        best = np.asarray(state.best_position, dtype=float)
        if best.size != dim:
            best = pop[self._best_index(pop[:, -1]), :-1].copy()

        x_coeff = np.asarray(state.payload["x_coeff"], dtype=float)
        mass = np.maximum(np.asarray(state.payload["mass"], dtype=float), 1.0e-12)
        length = np.asarray(state.payload["length"], dtype=float)
        elasticity = np.asarray(state.payload["elasticity"], dtype=float)
        gravity = max(float(state.payload["gravity"]), 1.0e-12)

        theta = math.pi * np.random.random(n)
        levy = self._levy_matrix(n, dim, self._levy_beta)
        amplitude = self._bound_amplitude()
        new_positions = np.empty((n, dim), dtype=float)

        period = max(1, int(round(n / 10.0)))
        x_mean = pop[:, :-1].mean(axis=0)

        for i in range(n):
            if np.random.random() > 0.5:
                wind = c / math.pi * (2.0 * np.random.random(dim) - 1.0) * amplitude
                one_based = i + 1
                if one_based % period == 0:
                    pos = x_mean + wind
                elif one_based % period == 1:
                    pos = best + wind
                else:
                    pos = pop[i, :-1] + wind
            elif np.random.random() > 0.5:
                a = amplitude - np.abs(amplitude * progress * math.sin(2.0 * math.pi * np.random.random()))
                roll = ((mass[i] * elasticity[i] + length[i] ** 2) / max(1, dim)) * np.random.uniform(-a, a, dim)
                pos = best + roll + c * levy[i, :] * best
            else:
                k = 0.5 + 0.5 * np.random.random()
                b = amplitude - np.abs(amplitude * progress * math.cos(2.0 * math.pi * np.random.random()))
                alpha = math.exp(float(np.random.randint(0, t + 1)) / float(horizon)) / math.pi
                jump = (
                    2.0
                    * k
                    * x_coeff[i] ** 2
                    * math.sin(2.0 * theta[i])
                    / mass[i]
                    / gravity
                    * (1.0 - alpha)
                    / max(1, dim)
                    * np.random.uniform(-b, b, dim)
                )
                pos = best + jump + c * levy[i, :] * best
            new_positions[i, :] = np.clip(pos, self._lo, self._hi)

        new_fitness = self._evaluate_population(new_positions)
        pop[:, :-1] = new_positions
        pop[:, -1] = new_fitness
        return pop, int(n), {}
