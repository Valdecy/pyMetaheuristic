"""pyMetaheuristic src — Glowworm Swarm Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class GSOEngine(PortedPopulationEngine):
    """Glowworm Swarm Optimization — luciferin-guided local-range movement."""
    algorithm_id = "gso"
    algorithm_name = "Glowworm Swarm Optimization"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=25, l0=5.0, nt=5.0, rho=0.4, gamma=0.6, beta=0.08, s=0.03, max_radius=1.0)

    def _initialize_payload(self, pop):
        return {"luciferin": np.full(pop.shape[0], float(self._params.get("l0", 5.0))), "radius": np.full(pop.shape[0], float(self._params.get("max_radius", 1.0)))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        luc = np.asarray(state.payload.get("luciferin", np.ones(n)), dtype=float)
        radius = np.asarray(state.payload.get("radius", np.ones(n)), dtype=float)
        if luc.shape[0] != n: luc = np.ones(n) * float(self._params.get("l0", 5.0))
        if radius.shape[0] != n: radius = np.ones(n) * float(self._params.get("max_radius", 1.0))
        rho, gamma, beta, nt = [float(self._params.get(k, v)) for k, v in [("rho",0.4),("gamma",0.6),("beta",0.08),("nt",5.0)]]
        luc = (1.0 - rho) * luc + gamma * self._quality(pop[:, -1])
        new_pos = pop[:, :-1].copy()
        scaled = pop[:, :-1] / self._span
        for i in range(n):
            dist = np.linalg.norm(scaled - scaled[i], axis=1)
            neigh = np.where((dist < radius[i]) & (luc > luc[i]))[0]
            if neigh.size:
                diff = luc[neigh] - luc[i]
                probs = diff / (diff.sum() + 1e-30)
                j = int(np.random.choice(neigh, p=probs))
                direction = pop[j, :-1] - pop[i, :-1]
                norm = float(np.linalg.norm(direction)) or 1.0
                new_pos[i] = np.clip(pop[i, :-1] + float(self._params.get("s", 0.03)) * self._span * direction / norm, self._lo, self._hi)
            radius[i] = min(float(self._params.get("max_radius", 1.0)), max(0.0, radius[i] + beta * (nt - neigh.size)))
        fit = self._evaluate_population(new_pos)
        return np.hstack((new_pos, fit[:, None])), n, {"luciferin": luc, "radius": radius}
