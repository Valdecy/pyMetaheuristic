"""pyMetaheuristic src — Greylag Goose Optimization Engine.

Native macro-step: dynamic exploration/exploitation groups -> position update
with GGO equations -> full-population evaluation -> elitist best preservation.
"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, EngineState
from ._ported_common import PortedPopulationEngine


class GGOEngine(PortedPopulationEngine):
    """Greylag Goose Optimization (GGO)."""

    algorithm_id = "ggo"
    algorithm_name = "Greylag Goose Optimization"
    family = "swarm"
    _REFERENCE = {"doi": "10.1016/j.eswa.2023.122147"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=10,
        b=1.0,
        c=2.0,
        stagnation_window=3,
    )

    _OPERATORS = (
        "ggo.initialization",
        "ggo.dynamic_group_update",
        "ggo.exploration_leader_move_eq1",
        "ggo.exploration_paddling_mutation_eq2",
        "ggo.exploration_spiral_move_eq4",
        "ggo.flock_local_search_eq7",
        "ggo.exploitation_sentry_guidance_eq5_6",
        "ggo.elitist_selection",
        "ggo.boundary_repair",
        "ggo.role_shuffle",
        "ggo.stagnation_group_boost",
        "ggo.candidate_injection",
    )

    @classmethod
    def _zero_counts(cls) -> dict[str, int]:
        return {name: 0 for name in cls._OPERATORS}

    @classmethod
    def _zero_contrib(cls) -> dict[str, float]:
        return {name: 0.0 for name in cls._OPERATORS}

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n = int(pop.shape[0])
        n1 = max(1, n // 2)
        n2 = max(1, n - n1)
        counts = self._zero_counts()
        contrib = self._zero_contrib()
        counts["ggo.initialization"] = n
        return {
            "n1": n1,
            "n2": n2,
            "previous_best_fitness": float(pop[self._best_index(pop[:, -1]), -1]),
            "stagnation_counter": 0,
            "operator_counts": counts.copy(),
            "operator_contributions": contrib.copy(),
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def _progress(self, state: EngineState) -> float:
        max_steps = self.config.max_steps or self.config.max_evaluations or int(self._params.get("iterations", 100))
        return min(1.0, max(0.0, (state.step + 1) / max(1, int(max_steps))))

    def _current_a(self, state: EngineState) -> float:
        # The paper decreases a linearly from 2 to 0 during iterations.
        return float(self._params.get("c", 2.0)) * (1.0 - self._progress(state))

    def _current_z(self, state: EngineState) -> float:
        # Eq. (3): z = 1 - (t / tmax)^2.
        p = self._progress(state)
        return float(1.0 - p * p)

    def _repair(self, raw: np.ndarray, counts: dict[str, int]) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        repaired = np.clip(raw, self._lo, self._hi)
        changed = np.any(np.abs(repaired - raw) > 1.0e-12, axis=1)
        counts["ggo.boundary_repair"] += int(np.count_nonzero(changed))
        return repaired

    def _improvement(self, old: float, new: float) -> float:
        return max(0.0, float(old) - float(new)) if self.problem.objective == "min" else max(0.0, float(new) - float(old))

    def _choose_three(self, n: int, exclude: int | None = None) -> np.ndarray:
        if exclude is None:
            pool = np.arange(n)
        else:
            pool = np.array([i for i in range(n) if i != exclude], dtype=int)
        replace = pool.size < 3
        return np.random.choice(pool, size=3, replace=replace)

    def _flock_local_search(self, x: np.ndarray, leader: np.ndarray, z: float) -> np.ndarray:
        # The paper denotes D in Eq. (7) but does not define it explicitly; using
        # distance to the leader is the natural counterpart to the nearby-flock text.
        w = np.random.uniform(0.0, 2.0, size=x.shape)
        D = np.abs(leader - x)
        return x + D * (1.0 + z) * w * (x - leader)

    def _step_impl(self, state: EngineState, pop: np.ndarray) -> tuple[np.ndarray, int, dict]:
        n, d = pop.shape[0], self.problem.dimension
        old_pop = pop.copy()
        old_best_pos = np.asarray(state.best_position, dtype=float)
        old_best_fit = float(state.best_fitness)
        leader = old_best_pos.copy()

        a = self._current_a(state)
        z = self._current_z(state)
        b = float(self._params.get("b", 1.0))
        t_number = int(state.step + 1)

        n1 = int(state.payload.get("n1", max(1, n // 2)))
        n1 = int(np.clip(n1, 1, max(1, n - 1))) if n > 1 else 1
        n2 = n - n1

        counts = self._zero_counts()
        contrib = self._zero_contrib()
        raw_new = pop[:, :-1].copy()
        source_label = ["ggo.dynamic_group_update"] * n
        counts["ggo.dynamic_group_update"] += 1

        # Exploration group: Eq. (1), Eq. (2), Eq. (4), or Eq. (7).
        for i in range(n1):
            x = pop[i, :-1]
            if t_number % 2 == 0:
                r3 = float(np.random.random())
                if r3 < 0.5:
                    r1 = np.random.random(d)
                    r2 = np.random.random(d)
                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2
                    if np.all(np.abs(A) < 1.0):
                        raw_new[i] = leader - A * np.abs(C * leader - x)
                        source_label[i] = "ggo.exploration_leader_move_eq1"
                    else:
                        p1, p2, p3 = self._choose_three(n, exclude=i)
                        w1 = np.random.uniform(0.0, 2.0, d)
                        w2 = np.random.uniform(0.0, 2.0, d)
                        w3 = np.random.uniform(0.0, 2.0, d)
                        raw_new[i] = (
                            w1 * pop[p1, :-1]
                            + z * w2 * (pop[p2, :-1] - pop[p3, :-1])
                            + (1.0 - z) * w3 * (x - pop[p1, :-1])
                        )
                        source_label[i] = "ggo.exploration_paddling_mutation_eq2"
                else:
                    l = np.random.uniform(-1.0, 1.0, d)
                    w1 = np.random.uniform(0.0, 2.0, d)
                    w4 = np.random.uniform(0.0, 2.0, d)
                    r4 = np.random.random(d)
                    r5 = np.random.random(d)
                    raw_new[i] = (
                        w4 * np.abs(leader - x) * np.exp(b * l) * np.cos(2.0 * np.pi * l)
                        + (2.0 * w1 * (r4 + r5)) * leader
                    )
                    source_label[i] = "ggo.exploration_spiral_move_eq4"
            else:
                raw_new[i] = self._flock_local_search(x, leader, z)
                source_label[i] = "ggo.flock_local_search_eq7"

        # Exploitation group: top-three sentry guidance Eq. (5)-(6), or Eq. (7).
        order = self._order(pop[:, -1])
        sentry_idx = order[: min(3, n)]
        while len(sentry_idx) < 3:
            sentry_idx = np.append(sentry_idx, sentry_idx[-1])
        s1, s2, s3 = pop[sentry_idx[0], :-1], pop[sentry_idx[1], :-1], pop[sentry_idx[2], :-1]
        for i in range(n1, n):
            x = pop[i, :-1]
            if t_number % 2 == 0:
                A1 = 2.0 * a * np.random.random(d) - a
                A2 = 2.0 * a * np.random.random(d) - a
                A3 = 2.0 * a * np.random.random(d) - a
                C1 = 2.0 * np.random.random(d)
                C2 = 2.0 * np.random.random(d)
                C3 = 2.0 * np.random.random(d)
                X1 = s1 - A1 * np.abs(C1 * s1 - x)
                X2 = s2 - A2 * np.abs(C2 * s2 - x)
                X3 = s3 - A3 * np.abs(C3 * s3 - x)
                raw_new[i] = (X1 + X2 + X3) / 3.0
                source_label[i] = "ggo.exploitation_sentry_guidance_eq5_6"
            else:
                raw_new[i] = self._flock_local_search(x, leader, z)
                source_label[i] = "ggo.flock_local_search_eq7"

        for label in source_label:
            counts[label] += 1

        new_pos = self._repair(raw_new, counts)
        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack((new_pos, new_fit[:, None]))

        for i, label in enumerate(source_label):
            contrib[label] += self._improvement(old_pop[i, -1], new_fit[i])

        # Elitist strategy: preserve the best-so-far if it was not surpassed.
        best_idx = self._best_index(new_pop[:, -1])
        if not self._is_better(new_pop[best_idx, -1], old_best_fit):
            worst_idx = self._worst_index(new_pop[:, -1])
            new_pop[worst_idx, :-1] = old_best_pos
            new_pop[worst_idx, -1] = old_best_fit
            counts["ggo.elitist_selection"] += 1
        else:
            counts["ggo.elitist_selection"] += 1
            contrib["ggo.elitist_selection"] += self._improvement(old_best_fit, new_pop[best_idx, -1])

        # Random role exchange between the exploration and exploitation groups.
        perm = np.random.permutation(n)
        new_pop = new_pop[perm]
        counts["ggo.role_shuffle"] += 1

        current_best_idx = self._best_index(new_pop[:, -1])
        current_best_fit = float(new_pop[current_best_idx, -1])
        previous_best = float(state.payload.get("previous_best_fitness", old_best_fit))
        improved = self._is_better(current_best_fit, previous_best)
        stagnation = 0 if improved else int(state.payload.get("stagnation_counter", 0)) + 1

        stagnation_window = max(1, int(self._params.get("stagnation_window", 3)))
        if stagnation >= stagnation_window:
            next_n1 = min(n - 1, n1 + 1) if n > 1 else 1
            counts["ggo.stagnation_group_boost"] += 1
        else:
            next_n1 = max(1, n1 - 1) if n > 1 else 1
        next_n2 = n - next_n1

        totals = dict(state.payload.get("operator_counts", self._zero_counts()))
        total_contrib = dict(state.payload.get("operator_contributions", self._zero_contrib()))
        for label in self._OPERATORS:
            totals[label] = int(totals.get(label, 0)) + int(counts.get(label, 0))
            total_contrib[label] = float(total_contrib.get(label, 0.0)) + float(contrib.get(label, 0.0))

        return new_pop, n, {
            "n1": int(next_n1),
            "n2": int(next_n2),
            "previous_best_fitness": min(previous_best, current_best_fit) if self.problem.objective == "min" else max(previous_best, current_best_fit),
            "stagnation_counter": int(stagnation),
            "a": float(a),
            "z": float(z),
            "operator_counts": totals,
            "operator_contributions": total_contrib,
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def _post_injection_repair(self, state: EngineState, replaced_indices: list[int], candidates) -> None:
        super()._post_injection_repair(state, replaced_indices, candidates)
        counts = dict(state.payload.get("operator_counts", self._zero_counts()))
        contrib = dict(state.payload.get("operator_contributions", self._zero_contrib()))
        last_counts = self._zero_counts()
        last_contrib = self._zero_contrib()
        k = int(len(replaced_indices))
        counts["ggo.candidate_injection"] = int(counts.get("ggo.candidate_injection", 0)) + k
        last_counts["ggo.candidate_injection"] = k
        state.payload["operator_counts"] = counts
        state.payload["operator_contributions"] = contrib
        state.payload["last_operator_counts"] = last_counts
        state.payload["last_operator_contributions"] = last_contrib

    def observe(self, state: EngineState) -> dict:
        obs = super().observe(state)
        obs.update(
            {
                "n1_exploration": int(state.payload.get("n1", 0)),
                "n2_exploitation": int(state.payload.get("n2", 0)),
                "a": float(state.payload.get("a", self._current_a(state))),
                "z": float(state.payload.get("z", self._current_z(state))),
                "stagnation_counter": int(state.payload.get("stagnation_counter", 0)),
                "operator_counts": dict(state.payload.get("last_operator_counts", state.payload.get("operator_counts", {}))),
                "operator_contributions": dict(state.payload.get("last_operator_contributions", state.payload.get("operator_contributions", {}))),
                "operator_counts_total": dict(state.payload.get("operator_counts", {})),
                "operator_contributions_total": dict(state.payload.get("operator_contributions", {})),
                "evomapx_delta_f": "direct_improvement",
                "evomapx_fidelity": "native",
                "native_evomapx_operator_labels": True,
            }
        )
        return obs
