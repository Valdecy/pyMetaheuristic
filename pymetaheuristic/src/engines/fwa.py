"""pyMetaheuristic src — Fireworks Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class FWAEngine(PortedPopulationEngine):
    """Fireworks Algorithm — explosion and Gaussian sparks with elitist selection."""
    algorithm_id = "fwa"
    algorithm_name = "Fireworks Algorithm"
    family = "swarm"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=5, num_sparks=50, a=0.04, b=0.8, max_amplitude=0.4, num_gaussian=5)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        q = self._quality(pop[:, -1]) + 1e-12
        invq = q.max() - q + 1e-12
        total_sparks = max(n, int(self._params.get("num_sparks", 50)))
        counts = np.floor(total_sparks * q / q.sum()).astype(int)
        counts = np.clip(counts, max(1, int(float(self._params.get("a", 0.04)) * total_sparks / n)), max(1, int(float(self._params.get("b", 0.8)) * total_sparks)))
        while counts.sum() < total_sparks:
            counts[np.random.randint(n)] += 1
        amps = float(self._params.get("max_amplitude", 0.4)) * self._span * invq[:, None] / invq.max()
        sparks = []
        for i in range(n):
            for _ in range(int(counts[i])):
                s = pop[i, :-1].copy()
                dims = np.random.rand(dim) < np.random.rand()
                if not np.any(dims):
                    dims[np.random.randint(dim)] = True
                s[dims] += np.random.uniform(-1, 1, dims.sum()) * amps[i, dims]
                sparks.append(np.clip(s, self._lo, self._hi))
        for _ in range(int(self._params.get("num_gaussian", 5))):
            i = np.random.randint(n)
            s = pop[i, :-1].copy()
            dims = np.random.rand(dim) < 0.5
            if not np.any(dims):
                dims[np.random.randint(dim)] = True
            s[dims] *= np.random.normal(1.0, 1.0, dims.sum())
            sparks.append(np.clip(s, self._lo, self._hi))
        spark_pop = self._pop_from_positions(np.asarray(sparks))
        combined = np.vstack((pop, spark_pop))
        order = self._order(combined[:, -1])
        chosen = [order[0]]
        while len(chosen) < n:
            remain = [i for i in range(combined.shape[0]) if i not in chosen]
            d = np.array([np.min(np.linalg.norm(combined[i, :-1] - combined[chosen, :-1], axis=1)) for i in remain])
            probs = d / (d.sum() + 1e-30)
            chosen.append(remain[int(np.random.choice(len(remain), p=probs))])
        return combined[chosen], spark_pop.shape[0], {}
