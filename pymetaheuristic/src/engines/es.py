"""pyMetaheuristic src — Evolution Strategy Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class ESEngine(PortedPopulationEngine):
    """Evolution Strategy (mu + lambda) with self-adaptive mutation scales."""
    algorithm_id = "es"
    algorithm_name = "Evolution Strategy (mu + lambda)"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=40, lam=45, sigma=0.15, tau=None, tau_prime=None)

    def _initialize_payload(self, pop):
        sigma = float(self._params.get("sigma", 0.15)) * np.tile(self._span, (pop.shape[0], 1))
        return {"sigma": sigma}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        lam = max(1, int(self._params.get("lam", 45)))
        sigmas = np.asarray(state.payload.get("sigma"), dtype=float)
        if sigmas.shape != (n, dim):
            sigmas = float(self._params.get("sigma", 0.15)) * np.tile(self._span, (n, 1))
        tau = self._params.get("tau")
        tau_prime = self._params.get("tau_prime")
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim)) if tau is None else float(tau)
        tau_prime = 1.0 / np.sqrt(2.0 * dim) if tau_prime is None else float(tau_prime)
        children, child_sigmas = [], []
        for _ in range(lam):
            i = np.random.randint(n)
            sigma = sigmas[i] * np.exp(tau_prime * np.random.normal() + tau * np.random.normal(size=dim))
            sigma = np.maximum(sigma, 1e-12)
            child = np.clip(pop[i, :-1] + np.random.normal(0.0, sigma), self._lo, self._hi)
            children.append(child); child_sigmas.append(sigma)
        child_pop = self._pop_from_positions(np.asarray(children))
        combined = np.vstack((pop, child_pop))
        combined_sigmas = np.vstack((sigmas, np.asarray(child_sigmas)))
        keep = self._order(combined[:, -1])[:n]
        return combined[keep], lam, {"sigma": combined_sigmas[keep]}
