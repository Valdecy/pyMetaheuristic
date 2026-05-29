"""pyMetaheuristic src — Electric Eel Foraging Optimization Engine."""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight


class EEFOEngine(PortedPopulationEngine):
    """Electric Eel Foraging Optimization (EEFO).

    Native NumPy port of the MATLAB reference algorithm with interacting,
    resting, migrating, and hunting behaviors controlled by the energy factor.
    """

    algorithm_id = "eefo"
    algorithm_name = "Electric Eel Foraging Optimization"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.eswa.2023.122200",
        "authors": "W. Zhao, L. Wang, Z. Zhang, H. Fan, J. Zhang, S. Mirjalili, N. Khodadadi, Q. Cao",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("EEFO requires population_size >= 2.")

    def _space_bound(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        mask = (x > self._hi) | (x < self._lo)
        if np.any(mask):
            x[mask] = np.random.random(np.count_nonzero(mask)) * (self._hi[mask] - self._lo[mask]) + self._lo[mask]
        return x

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 500)))
        positions = pop[:, :-1].copy()
        fitness = pop[:, -1].copy()
        best_pos = np.asarray(state.best_position, dtype=float)
        best_fit = float(state.best_fitness)
        mean_pos = np.mean(positions, axis=0)
        E0 = 4.0 * np.sin(1.0 - t / max_iter)
        operator_labels = ["carryover"] * n

        for i in range(n):
            rnd = max(float(np.random.random()), 1.0e-12)
            E = float(E0 * np.log(1.0 / rnd))
            direct = np.zeros(dim, dtype=float)
            if dim == 1:
                direct[0] = 1.0
            else:
                rand_num = int(np.ceil(((max_iter - t) / max_iter) * np.random.random() * max(0, dim - 2) + 2.0))
                rand_num = min(dim, max(1, rand_num))
                direct[np.random.permutation(dim)[:rand_num]] = 1.0

            attempted_label = "carryover"
            if E > 1.0:
                attempted_label = "eefo.interaction_migration"
                candidates = [j for j in range(n) if j != i]
                j = int(np.random.choice(candidates)) if candidates else i
                if self._is_better(fitness[j], fitness[i]):
                    base, other = positions[j], positions[i]
                else:
                    base, other = positions[i], positions[j]
                if np.random.random() > 0.5:
                    new_pos = base + np.random.randn() * direct * (mean_pos - other)
                else:
                    xr = np.random.random(dim) * (self._hi - self._lo) + self._lo
                    new_pos = base + np.random.randn() * direct * (xr - other)
            else:
                branch = float(np.random.random())
                if branch < 1.0 / 3.0:
                    attempted_label = "eefo.resting_area_update"
                    alpha = 2.0 * (np.e - np.exp(t / max_iter)) * np.sin(2.0 * np.pi * np.random.random())
                    rn = int(np.random.randint(0, n))
                    rd = int(np.random.randint(0, dim))
                    span = self._hi[rd] - self._lo[rd]
                    if abs(float(span)) > 1.0e-12:
                        z = (positions[rn, rd] - self._lo[rd]) / span
                        Z = self._lo + z * (self._hi - self._lo)
                    else:
                        Z = np.full(dim, positions[rn, rd], dtype=float)
                    Ri = Z + alpha * np.abs(Z - best_pos)
                    new_pos = Ri + np.random.randn() * (Ri - round(float(np.random.random())) * positions[i])
                elif branch > 2.0 / 3.0:
                    attempted_label = "eefo.levy_hunting_update"
                    rn = int(np.random.randint(0, n))
                    rd = int(np.random.randint(0, dim))
                    span = self._hi[rd] - self._lo[rd]
                    if abs(float(span)) > 1.0e-12:
                        z = (positions[rn, rd] - self._lo[rd]) / span
                        Z = self._lo + z * (self._hi - self._lo)
                    else:
                        Z = np.full(dim, positions[rn, rd], dtype=float)
                    alpha = 2.0 * (np.e - np.exp(t / max_iter)) * np.sin(2.0 * np.pi * np.random.random())
                    Ri = Z + alpha * np.abs(Z - best_pos)
                    beta = 2.0 * (np.e - np.exp(t / max_iter)) * np.sin(2.0 * np.pi * np.random.random())
                    Hr = best_pos + beta * np.abs(mean_pos - best_pos)
                    L = 0.01 * np.abs(levy_flight(dim, beta=1.5, scale=1.0))
                    new_pos = -np.random.random() * Ri + np.random.random() * Hr - L * (Hr - positions[i])
                else:
                    attempted_label = "eefo.prey_capture_update"
                    beta = 2.0 * (np.e - np.exp(t / max_iter)) * np.sin(2.0 * np.pi * np.random.random())
                    Hprey = best_pos + beta * np.abs(mean_pos - best_pos)
                    r4 = float(np.random.random())
                    eta = float(np.exp(r4 * (1.0 - t) / max_iter) * np.cos(2.0 * np.pi * r4))
                    new_pos = Hprey + eta * (Hprey - round(float(np.random.random())) * positions[i])

            new_pos = self._space_bound(new_pos)
            new_fit = float(self.problem.evaluate(new_pos))
            if self._is_better(new_fit, fitness[i]):
                positions[i] = new_pos
                fitness[i] = new_fit
                operator_labels[i] = attempted_label
                if self._is_better(new_fit, best_fit):
                    best_fit = new_fit
                    best_pos = new_pos.copy()

        pop[:, :-1] = positions
        pop[:, -1] = fitness
        return pop, n, {"prey_position": best_pos.copy(), "prey_fitness": float(best_fit), "operator_labels": operator_labels}
