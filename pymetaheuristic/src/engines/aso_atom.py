"""pyMetaheuristic src — Atom Search Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class ASOAtomEngine(PortedPopulationEngine):
    """Atom Search Optimization (ASO).

    Robust bounded port.  The previous force model used ``(-h)**negative`` for
    positive distances, producing unstable signed potentials and very large
    accelerations.  This version uses a stable mass-weighted attraction/repulsion
    model with velocity clipping and greedy selection.
    """

    algorithm_id = "aso_atom"
    algorithm_name = "Atom Search Optimization"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.knosys.2018.08.030",
        "title": "Atom Search Optimization: a metaheuristic algorithm for global optimization",
        "authors": "Zhao et al.",
        "year": 2019,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=40, alpha=20.0, beta=0.2, velocity_limit=0.25, elite_fraction=0.08, local_refinement_fraction=0.20, local_refinement_scale=0.12)

    def _initialize_payload(self, pop):
        return {"velocity": np.zeros((pop.shape[0], self.problem.dimension), dtype=float)}

    def _masses(self, fit):
        fit = np.asarray(fit, dtype=float)
        best = np.min(fit) if self.problem.objective == "min" else np.max(fit)
        worst = np.max(fit) if self.problem.objective == "min" else np.min(fit)
        denom = abs(worst - best) + 1.0e-12
        if self.problem.objective == "min":
            raw = (worst - fit) / denom
        else:
            raw = (fit - worst) / denom
        raw = np.maximum(raw, 0.0)
        if np.sum(raw) <= 1.0e-30:
            return np.full_like(raw, 1.0 / max(1, raw.size))
        return raw / (np.sum(raw) + 1.0e-12)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or int(self._params.get("max_iterations", 500)))
        t = state.step + 1
        progress = min(1.0, t / T)
        velocity = np.asarray(state.payload.get("velocity", np.zeros((n, dim))), dtype=float).copy()
        if velocity.shape != (n, dim):
            velocity = np.zeros((n, dim), dtype=float)

        masses = self._masses(pop[:, -1])
        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        worst = pop[order[-1], :-1].copy()

        # Number of active neighbours decreases over time; early iterations are
        # exploratory, late iterations emphasize the best atoms.
        K = max(2, int(round(n - (n - 2) * np.sqrt(progress))))
        alpha = float(self._params.get("alpha", 20.0)) * np.exp(-4.0 * progress)
        beta = float(self._params.get("beta", 0.2))
        vmax = max(1.0e-12, float(self._params.get("velocity_limit", 0.25))) * self._span
        elite_fraction = min(max(float(self._params.get("elite_fraction", 0.08)), 0.0), 0.5)
        elite_n = max(1, int(round(elite_fraction * n)))
        local_fraction = min(max(float(self._params.get("local_refinement_fraction", 0.20)), 0.0), 0.5)
        local_n = max(1, int(round(local_fraction * n))) if local_fraction > 0.0 else 0
        local_scale = max(0.0, float(self._params.get("local_refinement_scale", 0.12)))

        trial = pop[:, :-1].copy()
        mean_span = float(np.mean(self._span)) + 1.0e-12
        for i in range(n):
            xi = pop[i, :-1]
            force = np.zeros(dim, dtype=float)
            for j in order[:K]:
                if j == i:
                    continue
                xj = pop[j, :-1]
                diff = xj - xi
                dist = np.linalg.norm(diff) + 1.0e-12
                h = np.clip(dist / mean_span, 1.0e-6, 10.0)

                # Smooth bounded approximation of ASO interaction: repulsive at
                # very small distances, attractive otherwise, with decaying gain.
                if h < beta:
                    interaction = -alpha * (beta - h) / max(beta, 1.0e-12)
                else:
                    interaction = alpha / (h * h + 1.0e-12)
                force += np.random.random(dim) * interaction * diff / dist * masses[j]

            best_pull = np.random.random(dim) * (best - xi)
            diversity_push = 0.05 * (1.0 - progress) * np.random.random(dim) * (xi - worst)
            acc = force / (masses[i] + 1.0e-6) + best_pull + diversity_push
            velocity[i] = 0.65 * np.random.random(dim) * velocity[i] + acc
            velocity[i] = np.clip(velocity[i], -vmax, vmax)
            trial[i] = np.clip(xi + velocity[i], self._lo, self._hi)

        # Refine around the best by replacing a few worst trial points with a
        # shrinking Gaussian neighbourhood.  This gives ASO a stable exploitation
        # mechanism on narrow continuous optima.
        if local_n > 0:
            radius = (local_scale * (1.0 - progress) + 1.0e-4) * self._span
            worst_ids = order[-local_n:]
            trial[worst_ids] = np.clip(best + np.random.normal(0.0, radius, size=(local_n, dim)), self._lo, self._hi)

        # Do not move the current elites unless a strictly better trial appears.
        trial[order[:elite_n]] = pop[order[:elite_n], :-1]
        trial_fit = self._evaluate_population(trial)
        new_pop = pop.copy()
        mask = self._better_mask(trial_fit, pop[:, -1])
        new_pop[mask, :-1] = trial[mask]
        new_pop[mask, -1] = trial_fit[mask]
        return new_pop, n, {"velocity": velocity}
