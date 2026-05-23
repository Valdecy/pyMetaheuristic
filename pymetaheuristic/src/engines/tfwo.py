"""pyMetaheuristic src — Turbulent Flow of Water-based Optimization Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile, CandidateRecord, EngineState, OptimizationResult
from ._ported_common import PortedPopulationEngine


class TFWOEngine(PortedPopulationEngine):
    """Turbulent Flow of Water-based Optimization using whirlpools and object effects."""

    algorithm_id = "tfwo"
    algorithm_name = "Turbulent Flow of Water-based Optimization"
    family = "physics"
    _REFERENCE = {
        "doi": "10.1016/j.engappai.2020.103666",
        "authors": "Mojtaba Ghasemi, Iraj Faraji Davoudkhani, Ebrahim Akbari, Abolfazl Rahimnejad, Sahand Ghavidel, Li Li",
        "year": 2020,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=93, n_whirlpools=3, objects_per_whirlpool=None)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self._n_whirlpools = int(self._params.get("n_whirlpools", 3))
        if self._n_whirlpools < 1:
            raise ValueError("n_whirlpools must be >= 1.")
        objects_per_whirlpool = self._params.get("objects_per_whirlpool", None)
        if objects_per_whirlpool is None:
            objects_per_whirlpool = max(1, (self._n - self._n_whirlpools) // self._n_whirlpools)
        self._objects_per_whirlpool = int(objects_per_whirlpool)
        if self._objects_per_whirlpool < 1:
            raise ValueError("objects_per_whirlpool must be >= 1.")
        self._n = self._n_whirlpools * (self._objects_per_whirlpool + 1)

    def _flatten_population(self, state: EngineState) -> np.ndarray:
        whirl_pos = state.payload["whirlpool_positions"]
        whirl_cost = state.payload["whirlpool_costs"]
        obj_pos = state.payload["object_positions"].reshape(-1, self.problem.dimension)
        obj_cost = state.payload["object_costs"].reshape(-1)
        whirl_pop = np.hstack([whirl_pos, whirl_cost[:, None]])
        obj_pop = np.hstack([obj_pos, obj_cost[:, None]])
        return np.vstack([whirl_pop, obj_pop])

    def initialize(self) -> EngineState:
        positions = self._new_positions(self._n)
        fitness = self._evaluate_population(positions)
        order = self._order(fitness)
        positions = positions[order]
        fitness = fitness[order]

        n_wh = self._n_whirlpools
        n_obj_w = self._objects_per_whirlpool
        whirl_pos = positions[:n_wh].copy()
        whirl_cost = fitness[:n_wh].copy()
        whirl_delta = np.zeros(n_wh, dtype=float)

        object_positions = positions[n_wh:].copy()
        object_costs = fitness[n_wh:].copy()
        object_delta = np.zeros(n_wh * n_obj_w, dtype=float)
        perm = np.random.permutation(n_wh * n_obj_w)
        object_positions = object_positions[perm].reshape(n_wh, n_obj_w, self.problem.dimension)
        object_costs = object_costs[perm].reshape(n_wh, n_obj_w)
        object_delta = object_delta[perm].reshape(n_wh, n_obj_w)

        best_idx = self._best_index(whirl_cost)
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=whirl_pos[best_idx].tolist(),
            best_fitness=float(whirl_cost[best_idx]),
            initialized=True,
            payload={
                "whirlpool_positions": whirl_pos,
                "whirlpool_costs": whirl_cost,
                "whirlpool_delta": whirl_delta,
                "object_positions": object_positions,
                "object_costs": object_costs,
                "object_delta": object_delta,
            },
        )

    def _effect_of_objects(self, state: EngineState) -> int:
        whirl_pos = state.payload["whirlpool_positions"]
        whirl_cost = state.payload["whirlpool_costs"]
        object_positions = state.payload["object_positions"]
        object_costs = state.payload["object_costs"]
        object_delta = state.payload["object_delta"]
        n_wh, n_obj_w, dim = object_positions.shape
        evals = 0

        for i in range(n_wh):
            for j in range(n_obj_w):
                obj = object_positions[i, j]
                if n_wh == 1:
                    d = np.random.random(dim) * (whirl_pos[i] - obj)
                    d2 = np.zeros(dim, dtype=float)
                else:
                    other = [idx for idx in range(n_wh) if idx != i]
                    scores = np.array([
                        abs(float(whirl_cost[idx]))
                        * np.sqrt(abs(float(np.sum(whirl_pos[idx]) - np.sum(obj))) + 1.0e-12)
                        for idx in other
                    ])
                    d_min = other[int(np.argmin(scores))]
                    d_max = other[int(np.argmax(scores))]
                    d = np.random.random(dim) * (whirl_pos[d_min] - obj)
                    d2 = np.random.random(dim) * (whirl_pos[d_max] - obj)

                object_delta[i, j] += np.random.random() * np.random.random() * np.pi
                angle = object_delta[i, j]
                c = np.cos(angle)
                s = -np.sin(angle)
                displacement = (c * d + s * d2) * (1.0 + abs(c * s))
                trial = np.clip(whirl_pos[i] - displacement, self._lo, self._hi)
                trial_fitness = float(self.problem.evaluate(trial))
                evals += 1
                if self._is_better(trial_fitness, float(object_costs[i, j])) or trial_fitness == float(object_costs[i, j]):
                    object_positions[i, j] = trial
                    object_costs[i, j] = trial_fitness

                # Pseudo-code 3: random one-coordinate relocation. The source
                # applies the relocation directly when the probability triggers.
                flight_event = (abs((np.cos(angle) ** 2) * (np.sin(angle) ** 2))) ** 2
                if np.random.random() < flight_event:
                    k = np.random.randint(dim)
                    relocated = object_positions[i, j].copy()
                    relocated[k] = np.random.uniform(self._lo[k], self._hi[k])
                    object_positions[i, j] = relocated
                    object_costs[i, j] = float(self.problem.evaluate(relocated))
                    evals += 1

        return evals

    def _effect_of_whirlpools(self, state: EngineState) -> int:
        whirl_pos = state.payload["whirlpool_positions"]
        whirl_cost = state.payload["whirlpool_costs"]
        whirl_delta = state.payload["whirlpool_delta"]
        n_wh, dim = whirl_pos.shape
        evals = 0
        previous_best_idx = self._best_index(whirl_cost)
        previous_best_pos = whirl_pos[previous_best_idx].copy()
        previous_best_cost = float(whirl_cost[previous_best_idx])

        for i in range(n_wh):
            scores = np.empty(n_wh, dtype=float)
            for t in range(n_wh):
                if t == i:
                    scores[t] = np.inf if self.problem.objective == "min" else -np.inf
                else:
                    scores[t] = float(whirl_cost[t]) * abs(float(np.sum(whirl_pos[t]) - np.sum(whirl_pos[i])))
            neighbor_idx = self._best_index(scores)
            if neighbor_idx == i:
                neighbor_idx = previous_best_idx

            whirl_delta[i] += np.random.random() * np.random.random() * np.pi
            direction = whirl_pos[neighbor_idx] - whirl_pos[i]
            factor = abs(np.cos(whirl_delta[i]) + np.sin(whirl_delta[i]))
            trial = whirl_pos[neighbor_idx] - factor * np.random.random(dim) * direction
            trial = np.clip(trial, self._lo, self._hi)
            trial_fitness = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(trial_fitness, float(whirl_cost[i])) or trial_fitness == float(whirl_cost[i]):
                whirl_pos[i] = trial
                whirl_cost[i] = trial_fitness

        # Preserve the pre-step best whirlpool if all whirlpool moves degrade it.
        current_best_idx = self._best_index(whirl_cost)
        if self._is_better(previous_best_cost, float(whirl_cost[current_best_idx])):
            worst_idx = self._worst_index(whirl_cost)
            whirl_pos[worst_idx] = previous_best_pos
            whirl_cost[worst_idx] = previous_best_cost

        return evals

    def _object_whirlpool_exchange(self, state: EngineState) -> None:
        whirl_pos = state.payload["whirlpool_positions"]
        whirl_cost = state.payload["whirlpool_costs"]
        object_positions = state.payload["object_positions"]
        object_costs = state.payload["object_costs"]

        for i in range(self._n_whirlpools):
            best_obj_idx = self._best_index(object_costs[i])
            best_obj_cost = float(object_costs[i, best_obj_idx])
            if self._is_better(best_obj_cost, float(whirl_cost[i])) or best_obj_cost == float(whirl_cost[i]):
                old_pos = whirl_pos[i].copy()
                old_cost = float(whirl_cost[i])
                whirl_pos[i] = object_positions[i, best_obj_idx].copy()
                whirl_cost[i] = best_obj_cost
                object_positions[i, best_obj_idx] = old_pos
                object_costs[i, best_obj_idx] = old_cost

    def step(self, state: EngineState) -> EngineState:
        evals = self._effect_of_objects(state)
        evals += self._effect_of_whirlpools(state)
        self._object_whirlpool_exchange(state)

        state.step += 1
        state.evaluations += int(evals)
        pop = self._flatten_population(state)
        self._maybe_update_best(state, pop)
        return state

    def observe(self, state: EngineState) -> dict:
        pop = self._flatten_population(state)
        positions = pop[:, :-1]
        denom = float(np.linalg.norm(self._hi - self._lo)) or 1.0
        centroid = np.mean(positions, axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / denom)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(pop[:, -1])),
            "std_fitness": float(np.std(pop[:, -1])),
            "diversity": diversity,
            "n_whirlpools": self._n_whirlpools,
            "objects_per_whirlpool": self._objects_per_whirlpool,
        }

    def finalize(self, state: EngineState) -> OptimizationResult:
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
                "population_size": int(self._n),
                "n_whirlpools": int(self._n_whirlpools),
                "objects_per_whirlpool": int(self._objects_per_whirlpool),
                "elapsed_time": state.elapsed_time,
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = self._flatten_population(state)
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
