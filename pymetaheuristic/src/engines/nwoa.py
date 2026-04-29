"""pyMetaheuristic src — Narwhal Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, safe_norm


class NWOAEngine(PortedPopulationEngine):
    """Narwhal Optimizer with wave-based exploration and suction exploitation."""

    algorithm_id = "nwoa"
    algorithm_name = "Narwhal Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1038/s41598-024-61278-8",
        "title": "Narwhal optimizer: a novel nature-inspired metaheuristic algorithm",
        "authors": "Ayman A. Aly et al.",
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
        population_size=30,
        amplitude=1.0,
        wave_number=2.0 * np.pi,
        angular_frequency=2.0 * np.pi,
        damping=0.01,
        prey_decay=0.001,
        prey_energy_start=1.0,
        exploration_ratio=0.7,
    )

    def _initialize_payload(self, pop):
        return {"prey_energy": float(self._params.get("prey_energy_start", 1.0))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        t = state.step
        progress = min(1.0, float(t) / float(T))
        prey_energy = float(state.payload.get("prey_energy", self._params.get("prey_energy_start", 1.0)))
        prey_energy = max(0.0, prey_energy * np.exp(-float(self._params.get("prey_decay", 0.001)) * max(1, t + 1)))
        a = 2.0 - 2.0 * progress
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()

        explore_ratio = float(self._params.get("exploration_ratio", 0.7))
        if prey_energy < 0.3:
            explore_ratio = 0.3
        elif prey_energy < 0.6:
            explore_ratio = 0.5

        A = float(self._params.get("amplitude", 1.0))
        k = float(self._params.get("wave_number", 2.0 * np.pi))
        omega = float(self._params.get("angular_frequency", 2.0 * np.pi))
        delta = float(self._params.get("damping", 0.01))

        trials = np.zeros((n, dim), dtype=float)
        for i in range(n):
            xi = pop[i, :-1]
            cos_sim = float(np.dot(xi, best) / (safe_norm(xi) * safe_norm(best)))
            h = 1.0 - np.clip(cos_sim, -1.0, 1.0)
            wave_strength = A * abs(np.sin(k * h - omega * progress)) * np.exp(-delta * t)
            r1, r2 = np.random.rand(), np.random.rand()
            if np.random.rand() < explore_ratio:
                Aexp = 2.0 * a * r1 - a
                trials[i] = xi + Aexp * (best - xi) + wave_strength * np.random.rand(dim)
            else:
                Aexp = a * r1 - a
                Cexp = 2.0 * r2
                suction_strength = prey_energy / (1.0 + safe_norm(best - xi))
                suction_force = suction_strength * prey_energy
                trials[i] = xi - Aexp * (best - xi) + Cexp * suction_force * wave_strength * np.random.rand(dim)
        trial_pop = self._pop_from_positions(np.clip(trials, self._lo, self._hi))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        return pop, n, {"prey_energy": prey_energy}
