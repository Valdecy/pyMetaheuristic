"""pyMetaheuristic src — Coronavirus Herd Immunity Optimization Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CHIOEngine(PortedPopulationEngine):
    """Coronavirus Herd Immunity Optimization — susceptible/infected/immune epidemiological model."""
    algorithm_id   = "chio"
    algorithm_name = "Coronavirus Herd Immunity Optimization"
    family         = "human"
    _REFERENCE     = {"doi": "10.1007/s00521-020-05296-6"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, brr=0.15, max_age=10)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = pop.shape[0]
        # 0=susceptible, 1=infected, 2=immune — start all susceptible
        return {
            "immunity": np.zeros(n, dtype=int),
            "age":      np.zeros(n, dtype=int),
        }

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        brr      = float(self._params.get("brr", 0.15))
        max_age  = int(self._params.get("max_age", 10))

        imm = np.asarray(state.payload.get("immunity", np.zeros(n, int)), dtype=int)
        age = np.asarray(state.payload.get("age",      np.zeros(n, int)), dtype=int)

        infected_idx   = np.where(imm == 1)[0]
        susceptible_idx= np.where(imm == 0)[0]
        immune_idx     = np.where(imm == 2)[0]

        # Advance age of infected; recover if too old
        for i in infected_idx:
            age[i] += 1
            if age[i] > max_age:
                imm[i] = 2; age[i] = 0

        new_pos = pop[:, :-1].copy()
        evals   = 0
        mean_fit = float(np.mean(pop[:, -1]))

        for i in range(n):
            pos = pop[i, :-1].copy()
            for j in range(dim):
                r = np.random.random()
                if r < brr / 3.0:                           # infected contact
                    if len(infected_idx):
                        k = np.random.choice(infected_idx)
                        pos[j] += np.random.random() * (pop[i, j] - pop[k, j])
                elif r < 2.0 * brr / 3.0:                  # susceptible contact
                    if len(susceptible_idx):
                        k = np.random.choice(susceptible_idx)
                        pos[j] += np.random.random() * (pop[i, j] - pop[k, j])
                elif r < brr:                               # immune contact
                    if len(immune_idx):
                        k_idx = immune_idx[np.argmin(pop[immune_idx, -1]) if self.problem.objective=="min"
                                           else np.argmax(pop[immune_idx, -1])]
                        pos[j] += np.random.random() * (pop[i, j] - pop[k_idx, j])
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos); evals += n
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i, -1])):
                pop[i, :-1] = new_pos[i]; pop[i, -1] = new_fit[i]
                # Become infected if previously susceptible and above-average
                if imm[i] == 0 and not self._is_better(float(pop[i, -1]), mean_fit):
                    imm[i] = 1; age[i] = 1
            else:
                age[i] += 1

        return pop, evals, {"immunity": imm, "age": age}
