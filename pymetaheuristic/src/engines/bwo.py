"""pyMetaheuristic src — Black Widow Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BWOEngine(PortedPopulationEngine):
    """Black Widow Optimization — procreation, cannibalism and mutation inspired by spiders."""
    algorithm_id   = "bwo"
    algorithm_name = "Black Widow Optimization"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1016/j.engappai.2019.103249"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, pp=0.6, cr=0.44, pm=0.4)

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        pp     = float(self._params.get("pp", 0.6))
        cr     = float(self._params.get("cr", 0.44))
        pm     = float(self._params.get("pm", 0.4))

        order    = self._order(pop[:, -1])
        sorted_p = pop[order].copy()
        n_parents = max(2, int(n * pp))

        offspring = []
        evals     = 0

        # Procreation
        for _ in range(n_parents):
            idx1, idx2 = np.random.choice(n_parents, 2, replace=False)
            p1, p2     = sorted_p[idx1, :-1], sorted_p[idx2, :-1]
            # Arithmetic crossover (two children)
            alpha = np.random.random()
            c1 = np.clip(alpha * p1 + (1 - alpha) * p2, self._lo, self._hi)
            c2 = np.clip((1 - alpha) * p1 + alpha * p2, self._lo, self._hi)
            fits = self._evaluate_population(np.vstack([c1, c2])); evals += 2
            children = list(np.hstack([np.vstack([c1, c2]), fits[:, None]]))
            children.sort(key=lambda r: float(r[-1]) if self.problem.objective=="min" else -float(r[-1]))
            # Cannibalism: keep ≥1 child
            n_keep = max(1, int(np.random.binomial(len(children), 1 - cr)))
            offspring.extend(children[:n_keep])

        # Mutation
        n_mutate = max(0, int(n * pm))
        for _ in range(n_mutate):
            src = sorted_p[np.random.randint(n_parents), :-1].copy()
            i   = np.random.randint(dim)
            src[i] = np.random.uniform(self._lo[i], self._hi[i])
            fit = float(self.problem.evaluate(src)); evals += 1
            offspring.append(np.append(src, fit))

        if not offspring:
            return pop, evals, {}

        all_pop = np.vstack([sorted_p] + [r[np.newaxis, :] for r in offspring])
        order2  = self._order(all_pop[:, -1])
        return all_pop[order2[:n]], evals, {}
