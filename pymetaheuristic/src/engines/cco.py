"""pyMetaheuristic src — Cuckoo Catfish Optimizer Engine"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, levy_flight


class CCOEngine(PortedPopulationEngine):
    """Cuckoo Catfish Optimizer.

    Swarm optimizer integrating compressed-space movement, surround search,
    transition behavior, chaotic predation, and death/parasitism style rebirth.
    """

    algorithm_id = "cco"
    algorithm_name = "Cuckoo Catfish Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s10462-025-11291-x",
        "title": "Cuckoo catfish optimizer: a new meta-heuristic optimization algorithm",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30, death_probability=0.05, a=1.34, b=0.3)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        jin = self._aggregation_values(pop).mean()
        return {"fail_counts": np.zeros(pop.shape[0], dtype=int), "initial_aggregation": float(jin)}

    def _aggregation_values(self, pop: np.ndarray) -> np.ndarray:
        best = pop[self._best_index(pop[:, -1]), :-1]
        worst = pop[self._worst_index(pop[:, -1]), :-1]
        denom = np.abs(worst - best) + 1.0e-12
        return np.mean(np.abs((pop[:, :-1] - best[None, :]) / denom[None, :]), axis=1)

    def _greedy_single(self, pop: np.ndarray, i: int, x: np.ndarray, fail_counts: np.ndarray) -> None:
        x = np.clip(x, self._lo, self._hi)
        fx = float(self.problem.evaluate(x))
        if self._is_better(fx, float(pop[i, -1])):
            pop[i, :-1] = x
            pop[i, -1] = fx
            fail_counts[i] = 0
        else:
            fail_counts[i] += 1

    def _random_rows(self, pop: np.ndarray, i: int, k: int) -> np.ndarray:
        ids = self._rand_indices(pop.shape[0], i, k)
        return pop[ids, :-1]

    def _compressed_space(self, pop: np.ndarray, i: int, C: float) -> np.ndarray:
        x = pop[i, :-1]
        best = pop[self._best_index(pop[:, -1]), :-1]
        rows = self._random_rows(pop, i, 4)
        rd = abs(np.random.normal())
        choice = np.random.randint(3)
        if choice == 0:
            z1 = np.random.randint(0, 2)
            r3 = np.random.rand()
            return x + z1 * rd * ((best + rows[0]) / 2.0 - rows[1]) + (r3 / 2.0) * (rows[2] - rows[3])
        if choice == 1:
            z2 = np.random.randint(0, 2, size=self.problem.dimension)
            return z2 * (rows[0] + rd * (rows[0] - rows[1])) + (1 - z2) * x
        return rows[0] + rd * (best - x + rows[1] - rows[2])

    def _surround_search(self, pop: np.ndarray, i: int, C: float, T_shrink: float, Ji: float, Jin: float) -> np.ndarray:
        n, dim = pop.shape[0], self.problem.dimension
        x = pop[i, :-1]
        r_idx = int(np.random.choice(self._rand_indices(n, i, 1)))
        xr = pop[r_idx, :-1]
        xr_fit = pop[r_idx, -1]
        if self._is_better(float(xr_fit), float(pop[i, -1])):
            xe = x
            step = xr - (T_shrink + np.random.rand()) * x
        else:
            xe = xr
            step = x - (T_shrink + np.random.rand()) * xr

        theta = (1.0 - 10.0 * (i + 1) / max(1, n)) * math.pi
        a = float(self._params.get("a", 1.34))
        b = float(self._params.get("b", 0.3))
        c = a * math.exp(b * theta / 2.0) * math.cos(theta)
        s = a * math.exp(b * theta / 2.0) * math.sin(theta)
        best = pop[self._best_index(pop[:, -1]), :-1]
        mean = np.mean(pop[:, :-1], axis=0)
        rows = self._random_rows(pop, i, 3)
        r1, r2 = np.random.rand(), np.random.rand()
        if Ji < Jin:
            V = 2.0 * (r1 * (mean - x) + r2 * (best - x))
        else:
            V = 2.0 * (r1 * (rows[1] - rows[2]) + r2 * (rows[0] - x))
        F = np.random.choice([-1.0, 1.0])
        R1 = np.random.rand(dim)
        spiral = s if (i % 2 == 0) else c
        return xe + F * R1 * step / 2.0 + T_shrink**2 * spiral * (1.0 - R1) * np.abs(step) + V / (Ji + 1.0e-12)

    def _spherical_search(self, pop: np.ndarray, i: int, C: float) -> np.ndarray:
        x = pop[i, :-1]
        order = self._order(pop[:, -1])
        anchors = [pop[order[min(j, pop.shape[0] - 1)], :-1] for j in range(3)]
        anchors.append(np.mean(pop[:, :-1], axis=0))
        center = anchors[np.random.randint(0, len(anchors))]
        F = np.random.choice([-1.0, 1.0])
        w = C
        rt1 = np.random.uniform(0.0, 2.0 * math.pi, size=self.problem.dimension)
        rt2 = np.random.uniform(0.0, 2.0 * math.pi, size=self.problem.dimension)
        q = np.random.randint(1, 4)
        direction = center - x
        if q == 1:
            return center + 2.0 * w * F * np.cos(rt1) * np.sin(rt2) * direction
        if q == 2:
            return center + 2.0 * w * F * np.sin(rt1) * np.cos(rt2) * direction
        return center + 2.0 * w * F * np.cos(rt2) * direction

    def _transition(self, pop: np.ndarray, i: int, C: float, T_shrink: float) -> np.ndarray:
        x = pop[i, :-1]
        best = pop[self._best_index(pop[:, -1]), :-1]
        xr = pop[int(np.random.choice(self._rand_indices(pop.shape[0], i, 1))), :-1]
        E = T_shrink + np.random.rand()
        F = np.random.choice([-1.0, 1.0])
        De = C * F
        if i % 2 == 0:
            step2 = best - E * x
            return (C / max(1.0, float(state_step := 1))) * (np.random.rand() * best - np.random.rand() * x) + T_shrink**2 * levy_flight(self.problem.dimension, scale=1.0) * np.abs(step2)
        step2 = x - E * best
        R1 = np.random.rand(self.problem.dimension)
        R2 = np.random.rand(self.problem.dimension)
        R3 = np.random.rand(self.problem.dimension)
        return (best + xr) / 2.0 + De * 2.0 * R1 * step2 - R2**2 * (De * R3 - 1.0)

    def _chaotic_predation(self, pop: np.ndarray, i: int, C: float, T_shrink: float, Ji: float, Jin: float) -> np.ndarray:
        x = pop[i, :-1]
        best = pop[self._best_index(pop[:, -1]), :-1]
        F = np.random.choice([-1.0, 1.0])
        S = np.random.rand(self.problem.dimension)
        Lx = abs(np.random.normal()) * np.random.rand()
        Cy = 1.0 / (math.pi * (1.0 + C**2))
        Gs = np.random.normal(0.0, max(C, 1.0e-12), size=self.problem.dimension)
        if Ji > Jin:
            return best + F * S * (best - x)
        if Ji > Jin * Lx:
            return best * (1.0 + T_shrink**5 * Cy) * (T_shrink + np.random.rand()) + F * S * (best - x)
        return best * (1.0 + T_shrink**5 * Gs) + F * S * (best - x)

    def _death_or_parasitic(self, pop: np.ndarray, C: float) -> np.ndarray:
        best = pop[self._best_index(pop[:, -1]), :-1]
        if np.random.rand() > C:
            A = np.random.randint(0, 2)
            eggs = best * (float(levy_flight(1, scale=1.0)[0]) * A + abs(np.random.normal()) * (1 - A))
            lowc = float(np.min(eggs))
            upc = float(np.max(eggs))
            return np.random.rand(self.problem.dimension) * (upc - lowc) + lowc
        return self._new_positions(1)[0]

    def _step_impl(self, state, pop: np.ndarray):
        n = pop.shape[0]
        T_max = max(1, int(self.config.max_steps or max(100, state.step + 1)))
        it = min(T_max, state.step + 1)
        C = max(0.0, 1.0 - it / float(T_max))
        T_shrink = max(0.0, 1.0 - math.sin((math.pi / 2.0) * it / float(T_max))) ** (it / float(T_max))
        fail_counts = np.asarray(state.payload.get("fail_counts", np.zeros(n, dtype=int)), dtype=int)
        if fail_counts.shape[0] != n:
            fail_counts = np.resize(fail_counts, n)
        Jin = float(state.payload.get("initial_aggregation", self._aggregation_values(pop).mean()))
        Ji_all = self._aggregation_values(pop)
        die = float(self._params.get("death_probability", 0.05)) * (1.0 - C)
        evals = 0

        for i in range(n):
            Ji = float(Ji_all[i])
            exhausted = fail_counts[i] > max(1, int(0.8 * n))
            if exhausted or np.random.rand() < die:
                trial = self._death_or_parasitic(pop, C)
            elif C > 0.55:
                op = np.random.rand()
                if op < 0.40:
                    trial = self._compressed_space(pop, i, C)
                elif op < 0.75:
                    trial = self._surround_search(pop, i, C, T_shrink, Ji, Jin)
                else:
                    trial = self._spherical_search(pop, i, C)
            elif C > 0.25:
                if np.random.rand() < 0.5:
                    trial = self._transition(pop, i, C, T_shrink)
                else:
                    trial = self._surround_search(pop, i, C, T_shrink, Ji, Jin)
            else:
                trial = self._chaotic_predation(pop, i, C, T_shrink, Ji, Jin)

            self._greedy_single(pop, i, trial, fail_counts)
            evals += 1

        return pop, evals, {"fail_counts": fail_counts, "initial_aggregation": Jin}
