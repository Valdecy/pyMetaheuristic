"""pyMetaheuristic src — Parent-Centric Crossover (G3-PCX style) Engine"""
from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


class PCXEngine(BaseEngine):
    algorithm_id = "pcx"
    algorithm_name = "Parent-Centric Crossover (G3-PCX style)"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True)
    _DEFAULTS = dict(
        population_size=100,
        kids=2,
        family=2,
        parents=3,
        sigma_zeta=0.1,
        sigma_eta=0.1,
        max_parent_resampling=10,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p["population_size"])
        self._kids = int(p["kids"])
        self._family = int(p["family"])
        self._parents = int(p["parents"])
        self._sigma_zeta = float(p["sigma_zeta"])
        self._sigma_eta = float(p["sigma_eta"])
        self._max_parent_resampling = int(p["max_parent_resampling"])
        if self._n < 3:
            raise ValueError("pcx requires population_size >= 3")
        if self._parents < 3:
            raise ValueError("pcx requires parents >= 3 (elite anchor + at least two additional parents)")
        if self._parents > self._n:
            raise ValueError("pcx requires parents <= population_size")
        if self._kids < 1:
            raise ValueError("pcx requires kids >= 1")
        if self._family < 1:
            raise ValueError("pcx requires family >= 1")
        if self._family > self._n:
            raise ValueError("pcx requires family <= population_size")
        if config.seed is not None:
            np.random.seed(config.seed)

    def _init_pop(self) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        return np.hstack((pos, fit[:, np.newaxis]))

    def _rank_indices(self, fitness_values) -> np.ndarray:
        return self._fitness_order(fitness_values, descending=(self.problem.objective == "max"))

    def _best_index(self, pop: np.ndarray) -> int:
        return int(self._rank_indices(pop[:, -1])[0])

    def _sample_parent_indices(self, best_idx: int) -> np.ndarray:
        arr = np.arange(self._n, dtype=int)
        best_pos = int(np.where(arr == best_idx)[0][0])
        arr[0], arr[best_pos] = arr[best_pos], arr[0]
        for i in range(1, self._parents):
            j = np.random.randint(i, self._n)
            arr[i], arr[j] = arr[j], arr[i]
        return arr[: self._parents]

    def _sample_family_indices(self) -> np.ndarray:
        arr = np.arange(self._n, dtype=int)
        for i in range(self._family):
            j = np.random.randint(i, self._n)
            arr[i], arr[j] = arr[j], arr[i]
        return arr[: self._family]

    def _generate_child(self, pop: np.ndarray, parent_indices: np.ndarray) -> np.ndarray | None:
        parent_pos = pop[parent_indices, :-1]
        base = parent_pos[0]
        centroid = np.mean(parent_pos, axis=0)
        direction = centroid - base
        norm_direction = float(np.linalg.norm(direction))
        if norm_direction <= 1e-40:
            return None

        d_vals = []
        for j in range(1, parent_pos.shape[0]):
            diff = parent_pos[j] - base
            norm_diff = float(np.linalg.norm(diff))
            if norm_diff <= 1e-40:
                return None
            cosine = float(np.dot(diff, direction) / (norm_diff * norm_direction))
            cosine = float(np.clip(cosine, -1.0, 1.0))
            orth_distance = norm_diff * np.sqrt(max(0.0, 1.0 - cosine**2))
            d_vals.append(orth_distance)

        d_not = float(np.mean(d_vals)) if d_vals else 0.0
        eta = np.random.normal(0.0, d_not * self._sigma_eta, size=self.problem.dimension)
        projection = float(np.dot(eta, direction))
        eta = eta - (projection / (norm_direction**2)) * direction
        zeta = float(np.random.normal(0.0, self._sigma_zeta))
        child = base + eta + zeta * direction
        child = np.clip(child, self.problem.min_values, self.problem.max_values)
        fit = self.problem.evaluate(child)
        return np.concatenate((child, np.array([fit], dtype=float)))

    def initialize(self) -> EngineState:
        pop = self._init_pop()
        bi = self._best_index(pop)
        elite = pop[bi, :].copy()
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload=dict(
                population=pop,
                elite=elite,
                current=elite[:-1].tolist(),
                current_fit=float(elite[-1]),
                degeneracy_events=0,
                last_parent_indices=[],
                last_family_indices=[],
            ),
        )

    def step(self, state: EngineState) -> EngineState:
        pop = state.payload["population"]
        elite = state.payload.get("elite", pop[self._best_index(pop), :].copy())
        best_idx = self._best_index(pop)

        children = None
        parent_indices = None
        attempts = max(1, self._max_parent_resampling)
        for _ in range(attempts):
            candidate_parent_indices = self._sample_parent_indices(best_idx)
            batch = []
            for _kid in range(self._kids):
                child = self._generate_child(pop, candidate_parent_indices)
                if child is None:
                    batch = None
                    break
                batch.append(child)
            if batch is not None:
                children = batch
                parent_indices = candidate_parent_indices
                break

        evals = 0
        if children is None:
            lo = np.asarray(self.problem.min_values, dtype=float)
            hi = np.asarray(self.problem.max_values, dtype=float)
            rand_pos = np.random.uniform(lo, hi, (self._kids, self.problem.dimension))
            rand_fit = self._evaluate_population(rand_pos)
            children = [np.hstack((rand_pos[i], rand_fit[i])) for i in range(self._kids)]
            parent_indices = np.array([], dtype=int)
            state.payload["degeneracy_events"] = int(state.payload.get("degeneracy_events", 0)) + 1
            evals += self._kids
        else:
            evals += self._kids

        family_indices = self._sample_family_indices()
        family_rows = [pop[i, :].copy() for i in family_indices]
        pool = np.vstack(children + family_rows)
        selected = pool[self._rank_indices(pool[:, -1])[: self._family], :]

        for slot, row in zip(family_indices, selected):
            pop[slot, :] = row

        bi = self._best_index(pop)
        current = pop[bi, :].copy()
        if self.problem.is_better(float(current[-1]), float(elite[-1])):
            elite = current.copy()

        state.step += 1
        state.evaluations += evals
        state.payload = dict(
            population=pop,
            elite=elite,
            current=current[:-1].tolist(),
            current_fit=float(current[-1]),
            degeneracy_events=int(state.payload.get("degeneracy_events", 0)),
            last_parent_indices=parent_indices.tolist(),
            last_family_indices=family_indices.tolist(),
        )
        if self.problem.is_better(float(elite[-1]), state.best_fitness):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        pos = pop[:, :-1]
        fit = pop[:, -1]
        centroid = np.mean(pos, axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1))) if pos.shape[0] else 0.0
        current_fit = float(state.payload.get("current_fit", state.best_fitness))
        return dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            current_fitness=current_fit,
            mean_fitness=float(np.mean(fit)),
            std_fitness=float(np.std(fit)),
            diversity=diversity,
            kids=self._kids,
            family=self._family,
            parents=self._parents,
            sigma_zeta=self._sigma_zeta,
            sigma_eta=self._sigma_eta,
            degeneracy_events=int(state.payload.get("degeneracy_events", 0)),
        )

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=state.best_fitness,
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=state.best_fitness,
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata=dict(
                algorithm_name=self.algorithm_name,
                elapsed_time=state.elapsed_time,
                population_size=self._n,
                kids=self._kids,
                family=self._family,
                parents=self._parents,
                sigma_zeta=self._sigma_zeta,
                sigma_eta=self._sigma_eta,
                degeneracy_events=int(state.payload.get("degeneracy_events", 0)),
            ),
        )


    def inject_candidates(self, state: EngineState, candidates, policy="native") -> EngineState:
        if not candidates:
            return state
        pop = state.payload["population"]
        worst = self._rank_indices(pop[:, -1])[::-1]
        replaced = []
        for wi, cand in zip(worst, candidates):
            pos = np.clip(np.asarray(cand.position, dtype=float), self.problem.min_values, self.problem.max_values)
            fit = self.problem.evaluate(pos)
            pop[int(wi), :-1] = pos
            pop[int(wi), -1] = fit
            state.evaluations += 1
            replaced.append(int(wi))

        bi = self._best_index(pop)
        current = pop[bi, :].copy()
        elite = state.payload.get("elite", current.copy())
        if self.problem.is_better(float(current[-1]), float(elite[-1])):
            elite = current.copy()

        state.payload = dict(
            population=pop,
            elite=elite,
            current=current[:-1].tolist(),
            current_fit=float(current[-1]),
            degeneracy_events=int(state.payload.get("degeneracy_events", 0)),
            last_parent_indices=list(state.payload.get("last_parent_indices", [])),
            last_family_indices=list(state.payload.get("last_family_indices", [])),
            last_injected_indices=replaced,
        )
        if self.problem.is_better(float(elite[-1]), state.best_fitness):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
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
