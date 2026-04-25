"""pyMetaheuristic src — Search And Rescue Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SAROEngine(PortedPopulationEngine):
    """Search And Rescue Optimization — social and individual search phases with memory archive."""
    algorithm_id   = "saro"
    algorithm_name = "Search And Rescue Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1155/2019/2482543"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, se=0.5, mu=15)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        # memory archive = copy of initial population
        mem = pop.copy()
        return {"mem": mem, "USN": np.zeros(pop.shape[0], dtype=int)}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        se      = float(self._params.get("se", 0.5))
        mu      = int(self._params.get("mu", 15))

        mem = np.asarray(state.payload.get("mem", pop.copy()), dtype=float)
        USN = np.asarray(state.payload.get("USN", np.zeros(n, int)), dtype=int)
        combined = np.vstack([pop, mem])
        evals = 0

        # Social phase
        new_pop = np.empty_like(pop[:, :-1])
        for i in range(n):
            k  = np.random.choice([x for x in range(2 * n) if x != i])
            sd = pop[i, :-1] - combined[k, :-1]
            r1 = np.random.uniform(-1, 1)
            j_rand = np.random.randint(dim)
            pos = pop[i, :-1].copy()
            for j in range(dim):
                if np.random.random() < se or j == j_rand:
                    if self._is_better(float(combined[k, -1]), float(pop[i, -1])):
                        pos[j] = combined[k, j] + r1 * sd[j]
                    else:
                        pos[j] = pop[i, j] + r1 * sd[j]
                    # boundary repair
                    if pos[j] < self._lo[j]: pos[j] = (pop[i, j] + self._lo[j]) / 2.0
                    if pos[j] > self._hi[j]: pos[j] = (pop[i, j] + self._hi[j]) / 2.0
            new_pop[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pop); evals += n
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i, -1])):
                mem[np.random.randint(n)] = pop[i].copy()
                pop[i, :-1] = new_pop[i]; pop[i, -1] = new_fit[i]
                USN[i] = 0
            else:
                USN[i] += 1

        # Individual phase
        combined2 = np.vstack([pop, mem])
        ind_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            k, m = np.random.choice([x for x in range(2 * n) if x != i], 2, replace=False)
            pos  = pop[i, :-1] + np.random.random() * (combined2[k, :-1] - combined2[m, :-1])
            for j in range(dim):
                if pos[j] < self._lo[j]: pos[j] = (pop[i, j] + self._lo[j]) / 2.0
                if pos[j] > self._hi[j]: pos[j] = (pop[i, j] + self._hi[j]) / 2.0
            ind_pos[i] = np.clip(pos, self._lo, self._hi)

        ind_fit = self._evaluate_population(ind_pos); evals += n
        for i in range(n):
            if self._is_better(float(ind_fit[i]), float(pop[i, -1])):
                mem[np.random.randint(n)] = pop[i].copy()
                pop[i, :-1] = ind_pos[i]; pop[i, -1] = ind_fit[i]
                USN[i] = 0
            else:
                USN[i] += 1

        # Reinitialize individuals with too many unsuccessful searches
        for i in range(n):
            if USN[i] >= mu:
                pop[i, :-1] = np.random.uniform(self._lo, self._hi)
                pop[i, -1]  = float(self.problem.evaluate(pop[i, :-1])); evals += 1
                USN[i] = 0

        return pop, evals, {"mem": mem, "USN": USN}
