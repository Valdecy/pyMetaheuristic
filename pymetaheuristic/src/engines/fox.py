"""pyMetaheuristic src — Fox Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class FOXEngine(PortedPopulationEngine):
    """Fox Optimizer — robust bounded implementation.

    The earlier port replaced the entire population by positions derived mostly
    from the absolute best coordinate.  That collapses diversity and performs
    poorly when coordinates can be negative or near zero.  This implementation
    keeps the FOX exploration/exploitation spirit but uses current-position
    movements, greedy replacement, and explicit elite preservation.
    """
    algorithm_id   = "fox"
    algorithm_name = "Fox Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s10489-022-03533-0"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, c1=0.18, c2=0.82, walk_scale=0.25, elite_fraction=0.05)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"mint": 1.0}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or int(self._params.get("max_iterations", 500)))
        t = state.step + 1
        progress = min(1.0, t / T)
        c1 = float(self._params.get("c1", 0.18))
        c2 = float(self._params.get("c2", 0.82))
        walk_scale = max(0.0, float(self._params.get("walk_scale", 0.25)))
        elite_fraction = min(max(float(self._params.get("elite_fraction", 0.05)), 0.0), 0.5)
        mint = float(state.payload.get("mint", 1.0))

        order = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        elite_n = max(1, int(round(elite_fraction * n)))
        span = self._span
        contraction = 1.0 - progress
        jump_decay = 0.15 + 0.85 * contraction

        trial = pop[:, :-1].copy()
        for i in range(n):
            xi = pop[i, :-1]
            r = np.random.random(dim)
            mean_time = float(np.mean(r))
            mint = min(mint, mean_time)

            if np.random.random() < 0.5:
                # Exploitation: jump from the current position toward the prey
                # (best), scaled by a FOX-like jump factor.
                gravity_jump = 0.5 * 9.81 * (0.5 * mean_time) ** 2
                step = (c1 + (c2 - c1) * np.random.random()) * gravity_jump
                candidate = xi + step * r * (best_pos - xi)
            else:
                # Exploration: bounded random walk around the best and current
                # positions.  The radius decreases with progress.
                direction = np.random.uniform(-1.0, 1.0, dim)
                candidate = best_pos + direction * span * walk_scale * jump_decay * (0.5 + mint)
                if np.random.random() < 0.5:
                    candidate = xi + np.random.random(dim) * (candidate - xi)

            trial[i] = np.clip(candidate, self._lo, self._hi)

        # Preserve the best few individuals explicitly.
        trial[order[:elite_n]] = pop[order[:elite_n], :-1]
        trial_fit = self._evaluate_population(trial)
        new_pop = pop.copy()
        mask = self._better_mask(trial_fit, pop[:, -1])
        new_pop[mask, :-1] = trial[mask]
        new_pop[mask, -1] = trial_fit[mask]
        return new_pop, n, {"mint": mint}
