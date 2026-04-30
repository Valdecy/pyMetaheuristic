"""pyMetaheuristic src — Polar Fox Optimization (PFA) Engine.

"""
from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineState,
    OptimizationResult,
)


class PFAPolarFoxEngine(BaseEngine):
    """Polar Fox Optimization — group-based hunting, leadership, and mutation."""

    algorithm_id = "pfa_polar_fox"
    algorithm_name = "Polar Fox Optimization"
    family = "swarm"
    _REFERENCE = {"doi": "10.1007/s00521-024-10346-4"}
    capabilities = CapabilityProfile(has_population=True)
    _DEFAULTS = {
        "population_size": 40,
        "mutation_factor": 0.2,
        "max_leader_motivation": 3,
        "group_weights": (1000.0, 1000.0, 1000.0, 1000.0),
        "group_initial": (
            (2.0, 2.0, 0.9, 0.9, 0.1),
            (10.0, 2.0, 0.2, 0.9, 0.3),
            (2.0, 10.0, 0.9, 0.2, 0.3),
            (2.0, 12.0, 0.9, 0.9, 0.01),
        ),
        "group_motivated": (
            (2.0, 2.0, 0.99, 0.99, 0.1),
            (10.0, 2.0, 0.2, 0.99, 0.3),
            (2.0, 10.0, 0.99, 0.2, 0.3),
            (2.0, 12.0, 0.9, 0.9, 0.001),
        ),
        "group_recovery": (
            (0.0, 0.0, 0.001, 0.001, 0.0),
            (0.0, 0.0, 0.0, 0.001, 0.0),
            (0.0, 0.0, 0.001, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0001),
        ),
        "max_phase_trials": 50,
    }

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])
        self._mutation_factor = float(p["mutation_factor"])
        self._max_leader_motivation = int(p["max_leader_motivation"])
        self._group_weights0 = self._as_matrix(p["group_weights"], expected=(4,))
        self._group_initial = self._as_matrix(p["group_initial"], expected=(4, 5))
        self._group_motivated = self._as_matrix(p["group_motivated"], expected=(4, 5))
        self._group_recovery = self._as_matrix(p["group_recovery"], expected=(4, 5))
        self._max_phase_trials = max(1, int(p["max_phase_trials"]))
        if self._n < 4:
            raise ValueError("population_size must be at least 4 for Polar Fox Optimization.")
        if not (0.0 < self._mutation_factor <= 1.0):
            raise ValueError("mutation_factor must be in (0, 1].")
        if self._max_leader_motivation < 1:
            raise ValueError("max_leader_motivation must be >= 1.")
        if np.any(self._group_weights0 <= 0.0):
            raise ValueError("group_weights must be strictly positive.")
        if config.seed is not None:
            np.random.seed(config.seed)

    @staticmethod
    def _as_matrix(value, expected):
        arr = np.asarray(value, dtype=float)
        if arr.shape != expected:
            raise ValueError(f"Expected shape {expected}, received {arr.shape}.")
        return arr.copy()

    def _objective_scores(self, fitness: np.ndarray) -> np.ndarray:
        raw = np.asarray(fitness, dtype=float)
        work = raw if self.problem.objective == "min" else -raw
        shift = work.min()
        return work - shift + 1e-12

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _init_pop(self, n=None):
        if n is None:
            n = self._n
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        return np.hstack((pos, fit[:, np.newaxis]))

    def _allocate_groups(self, n: int, weights: np.ndarray) -> np.ndarray:
        probs = weights / np.sum(weights)
        counts = np.maximum(1, np.floor(probs * n).astype(int))
        while counts.sum() > n:
            idx = int(np.argmax(counts))
            if counts[idx] > 1:
                counts[idx] -= 1
            else:
                break
        while counts.sum() < n:
            idx = int(np.argmax(probs - counts / max(1, n)))
            counts[idx] += 1
        labels = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])
        if labels.size > n:
            labels = labels[:n]
        elif labels.size < n:
            extra = np.random.choice(4, size=n - labels.size, p=probs)
            labels = np.concatenate([labels, extra.astype(int)])
        np.random.shuffle(labels)
        return labels

    def _move_toward_initial(self, current: np.ndarray) -> np.ndarray:
        moved = current.copy()
        for g in range(current.shape[0]):
            for j in range(current.shape[1]):
                target = self._group_initial[g, j]
                step = self._group_recovery[g, j]
                value = moved[g, j]
                if value > target:
                    moved[g, j] = max(target, value - step)
                elif value < target:
                    moved[g, j] = min(target, value + step)
        return moved

    def initialize(self):
        pop = self._init_pop()
        order = self._order(pop[:, -1])
        elite = pop[order[0], :].copy()
        group_assignments = self._allocate_groups(self._n, self._group_weights0)
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload={
                "population": pop,
                "elite": elite,
                "group_weights": self._group_weights0.copy(),
                "group_behaviors": self._group_initial.copy(),
                "leader_motivation_count": 0,
                "group_assignments": group_assignments,
            },
        )

    def _experience_phase(self, position: np.ndarray, fitness: float, params: np.ndarray):
        pf, _, a_coef, _, m_coef = params
        trials = 0
        current_pos = position.copy()
        current_fit = float(fitness)
        while trials < self._max_phase_trials:
            power_factor = pf * (a_coef ** trials)
            if power_factor < m_coef * pf:
                break
            jump_power = np.random.random(self.problem.dimension) * power_factor
            jump_direction = np.cos(np.deg2rad(np.random.random(self.problem.dimension) * 180.0))
            candidate = np.clip(current_pos + jump_power * jump_direction, self.problem.min_values, self.problem.max_values)
            cand_fit = float(self.problem.evaluate(candidate))
            trials += 1
            if self.problem.is_better(cand_fit, current_fit):
                return candidate, cand_fit, trials, True
        return current_pos, current_fit, trials, False

    def _leader_phase(self, position: np.ndarray, fitness: float, leader: np.ndarray, params: np.ndarray):
        _, lf, _, b_coef, m_coef = params
        trials = 0
        current_pos = position.copy()
        current_fit = float(fitness)
        leader_pos = np.asarray(leader, dtype=float)
        while trials < self._max_phase_trials:
            leader_factor = lf * (b_coef ** trials)
            if leader_factor < m_coef * lf:
                break
            step = np.random.uniform(-1.0, 1.0, self.problem.dimension) * (current_pos - leader_pos) * leader_factor
            candidate = np.clip(current_pos + step, self.problem.min_values, self.problem.max_values)
            cand_fit = float(self.problem.evaluate(candidate))
            trials += 1
            if self.problem.is_better(cand_fit, current_fit):
                return candidate, cand_fit, trials, True
        return current_pos, current_fit, trials, False

    def step(self, state):
        pop = np.asarray(state.payload["population"], dtype=float).copy()
        weights = np.asarray(state.payload["group_weights"], dtype=float).copy()
        behaviors = np.asarray(state.payload["group_behaviors"], dtype=float).copy()
        evals = 0
        order = self._order(pop[:, -1])
        pop = pop[order]
        assignments = self._allocate_groups(self._n, weights)
        assignments[0] = assignments[0]  # elite may belong to any group; kept for count accounting
        leader = pop[0, :-1].copy()
        prev_best = float(pop[0, -1])
        success_by_group = np.zeros(4, dtype=bool)

        for i in range(1, self._n):
            group_idx = int(assignments[i])
            params = behaviors[group_idx]
            pos = pop[i, :-1].copy()
            fit = float(pop[i, -1])

            pos, fit, used, improved = self._experience_phase(pos, fit, params)
            evals += used
            success_by_group[group_idx] |= improved

            pos, fit, used, improved = self._leader_phase(pos, fit, leader, params)
            evals += used
            success_by_group[group_idx] |= improved

            pop[i, :-1] = pos
            pop[i, -1] = fit

        counts = np.array([(assignments == g).sum() for g in range(4)], dtype=float)
        counts = np.maximum(counts, 1.0)
        t = state.step + 1
        for g in range(4):
            if success_by_group[g]:
                weights[g] += (t ** 2) / counts[g]

        pop = pop[self._order(pop[:, -1])]
        improved_global = self.problem.is_better(float(pop[0, -1]), prev_best)
        leader_motivation_count = int(state.payload.get("leader_motivation_count", 0))
        leader_motivation_count = 0 if improved_global else leader_motivation_count + 1
        T = max(1, self.config.max_steps or 1000)
        critical = (leader_motivation_count > self._max_leader_motivation) or (t > 0.8 * T)

        nm = self._n - 1 if critical else max(1, int(round(self._mutation_factor * self._n)))
        replace_idx = np.arange(self._n - nm, self._n)
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        new_pos = np.random.uniform(lo, hi, (replace_idx.size, self.problem.dimension))
        new_fit = self._evaluate_population(new_pos)
        evals += int(replace_idx.size)
        pop[replace_idx, :-1] = new_pos
        pop[replace_idx, -1] = new_fit

        if critical:
            behaviors = self._group_motivated.copy()
            random_pos = np.random.uniform(lo, hi, (self._n - 1, self.problem.dimension))
            random_fit = self._evaluate_population(random_pos)
            evals += self._n - 1
            pop[1:, :-1] = random_pos
            pop[1:, -1] = random_fit
        else:
            behaviors = self._move_toward_initial(behaviors)

        counts = np.array([(assignments == g).sum() for g in range(4)], dtype=float)
        for g in range(4):
            if counts[g] < 0.1 * self._n:
                behaviors[g] = self._group_motivated[g]

        pop = pop[self._order(pop[:, -1])]
        elite = pop[0, :].copy()
        state.step += 1
        state.evaluations += evals
        state.payload = {
            "population": pop,
            "elite": elite,
            "group_weights": weights,
            "group_behaviors": behaviors,
            "leader_motivation_count": leader_motivation_count,
            "group_assignments": assignments,
        }
        if self.problem.is_better(float(elite[-1]), float(state.best_fitness)):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        pos = pop[:, :-1]
        fitness = pop[:, -1]
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        denom = np.linalg.norm(hi - lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "diversity": diversity,
        }

    def get_best_candidate(self, state):
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def finalize(self, state):
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "elapsed_time": state.elapsed_time,
                "reference": self._REFERENCE,
                "assumptions": [
                    "Leader motivation is implemented as a random reinitialization of non-elite foxes when the paper's critical condition is triggered.",
                    "The fatigue phase moves each group parameter toward its initial table value component-wise using the published recovery rates.",
                    "Weights are updated once per successful group per iteration using Eq. (5).",
                ],
            },
        )

    def get_population(self, state):
        pop = state.payload["population"]
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current",
            )
            for i in range(pop.shape[0])
        ]
