"""pyMetaheuristic src — Giant Pacific Octopus Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class GPOOEngine(PortedPopulationEngine):
    """
    Giant Pacific Octopus Optimizer.

    Notes on ambiguities handled conservatively:
    * Eq. (12) in the paper uses ``nextRand > 1`` even though ``nextRand`` is
      defined in [0, 1]. This implementation treats that branch as the intended
      exploration branch and uses a 0.5 threshold.
    * The paper describes tentacle points but does not provide a fully usable
      head/tentacle optimisation state transition. We therefore keep tentacles
      as auxiliary local probes around each octopus head and optimise the head
      positions directly.
    """

    algorithm_id = "gpoo"
    algorithm_name = "Giant Pacific Octopus Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s12065-024-00945-4",
        "title": "A novel giant pacific octopus optimizer for real-world engineering problem",
        "authors": "Pham Vu Hong Son, Luu Ngoc Quynh Khoi",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=100,
        convergence_parameter=1.0,
        max_tentacles=8,
        max_points=2,
        max_width=1.0,
        exploration_threshold=0.5,
        elite_fraction=0.2,
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 2:
            raise ValueError("gpoo requires population_size >= 2.")
        if float(self._params.get("convergence_parameter", 1.0)) <= 0.0:
            raise ValueError("gpoo convergence_parameter must be positive.")
        if int(self._params.get("max_tentacles", 8)) < 1:
            raise ValueError("gpoo max_tentacles must be at least 1.")
        if int(self._params.get("max_points", 2)) < 1:
            raise ValueError("gpoo max_points must be at least 1.")
        if float(self._params.get("max_width", 1.0)) <= 0.0:
            raise ValueError("gpoo max_width must be positive.")
        threshold = float(self._params.get("exploration_threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("gpoo exploration_threshold must be in [0, 1].")
        elite_fraction = float(self._params.get("elite_fraction", 0.2))
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("gpoo elite_fraction must be in (0, 1].")

    def _build_tentacles(self, heads: np.ndarray) -> np.ndarray:
        n, dim = heads.shape
        max_tentacles = int(self._params.get("max_tentacles", 8))
        max_points = int(self._params.get("max_points", 2))
        max_width = float(self._params.get("max_width", 1.0))
        tentacles = np.empty((n, max_tentacles, max_points, dim), dtype=float)
        tentacles[:, :, 0, :] = heads[:, None, :]
        for p in range(1, max_points):
            step = np.random.uniform(-max_width, max_width, (n, max_tentacles, dim))
            tentacles[:, :, p, :] = np.clip(tentacles[:, :, p - 1, :] + step, self._lo, self._hi)
        return tentacles

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"tentacles": self._build_tentacles(pop[:, :-1])}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        current = pop[:, :-1].copy()
        tentacles = state.payload.get("tentacles")
        if tentacles is None or np.asarray(tentacles).shape[0] != n:
            tentacles = self._build_tentacles(current)
        else:
            tentacles = np.asarray(tentacles, dtype=float)

        order = self._order(pop[:, -1])
        elite_count = max(1, int(np.ceil(float(self._params.get("elite_fraction", 0.2)) * n)))
        prey_mean = np.mean(current[order[:elite_count]], axis=0)
        tentacle_mean = np.mean(tentacles, axis=(1, 2))

        a = float(self._params.get("convergence_parameter", 1.0))
        threshold = float(self._params.get("exploration_threshold", 0.5))
        max_width = float(self._params.get("max_width", 1.0))

        new_pos = np.empty_like(current)
        for i in range(n):
            target = 0.5 * (prey_mean + tentacle_mean[i])
            r11 = max_width * np.random.rand(dim)
            r21 = np.random.rand(dim)
            a11 = 2.0 * a * r11 - a
            c11 = 2.0 * r21
            next_rand = float(np.random.rand())
            if next_rand > threshold:
                candidate = np.random.uniform(self._lo, self._hi, dim)
            else:
                dm = np.abs(c11 * target - current[i])
                candidate = target - a11 * dm
            new_pos[i] = np.clip(candidate, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack((new_pos, new_fit[:, None]))
        new_tentacles = self._build_tentacles(new_pos)
        return new_pop, n, {"tentacles": new_tentacles}
