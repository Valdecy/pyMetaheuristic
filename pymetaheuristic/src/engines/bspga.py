"""
pyMetaheuristic src — Binary Space Partition Tree Genetic Algorithm Engine
===========================================================================

This engine is implemented as a real-coded GA for continuous problems and uses
BSP/KD-tree neighbourhood learning as an intensification operator.  The previous
port rounded all variables to {0, 1}; that was inappropriate for continuous
benchmarks and made the engine effectively unable to search most domains.
"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

try:
    from scipy.spatial import KDTree as _KDTree
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_SCIPY = False


class BSPGAEngine(BaseEngine):
    algorithm_id   = "bspga"
    algorithm_name = "Binary Space Partition Tree Genetic Algorithm"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(
        population_size=50,
        crossover_rate=0.85,
        mutation_rate=None,
        mutation_scale=0.12,
        elite_fraction=0.10,
        tournament_size=3,
        bsp_learning_rate=0.35,
        bsp_neighbor_count=5,
    )
    _REFERENCE     = dict(doi="10.1016/j.ins.2019.10.016")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = max(6, int(p["population_size"]))
        self._cr = float(p.get("crossover_rate", 0.85))
        self._mutation_rate = p.get("mutation_rate", None)
        self._mutation_scale = max(0.0, float(p.get("mutation_scale", 0.12)))
        self._elite_fraction = min(max(float(p.get("elite_fraction", 0.10)), 0.0), 0.5)
        self._tournament_size = max(2, int(p.get("tournament_size", 3)))
        self._bsp_learning_rate = min(max(float(p.get("bsp_learning_rate", 0.35)), 0.0), 1.0)
        self._bsp_neighbor_count = max(1, int(p.get("bsp_neighbor_count", 5)))
        if not (0.0 <= self._cr <= 1.0):
            raise ValueError("bspga crossover_rate must be in [0, 1].")
        if config.seed is not None:
            np.random.seed(config.seed)
        self._lo = np.asarray(problem.min_values, dtype=float)
        self._hi = np.asarray(problem.max_values, dtype=float)
        self._span = np.where(self._hi > self._lo, self._hi - self._lo, 1.0)

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _better_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a < b if self.problem.objective == "min" else a > b

    def _evaluate_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([float(self.problem.evaluate(row.copy())) for row in positions], dtype=float)

    def _initial_positions(self) -> np.ndarray:
        return np.random.uniform(self._lo, self._hi, (self._n, self.problem.dimension))

    def _tournament_select(self, pop: np.ndarray) -> np.ndarray:
        ids = np.random.randint(0, pop.shape[0], size=self._tournament_size)
        local = pop[ids]
        winner = ids[self._order(local[:, -1])[0]]
        return pop[winner, :-1].copy()

    def _blend_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self._cr:
            return p1.copy(), p2.copy()
        alpha = np.random.rand(self.problem.dimension)
        c1 = alpha * p1 + (1.0 - alpha) * p2
        c2 = alpha * p2 + (1.0 - alpha) * p1
        # Mild BLX expansion helps preserve exploration without leaving bounds.
        diff = np.abs(p1 - p2)
        gamma = np.random.uniform(-0.25, 0.25, size=self.problem.dimension) * diff
        return c1 + gamma, c2 - gamma

    def _mutate(self, offspring: np.ndarray, state_step: int) -> np.ndarray:
        D = self.problem.dimension
        pm = 1.0 / max(1, D) if self._mutation_rate is None else float(self._mutation_rate)
        pm = min(max(pm, 0.0), 1.0)
        progress = 0.0
        if self.config.max_steps:
            progress = min(1.0, max(0.0, state_step / max(1, self.config.max_steps)))
        scale = self._mutation_scale * (1.0 - 0.75 * progress) * self._span
        mask = np.random.rand(*offspring.shape) < pm
        noise = np.random.normal(0.0, scale, size=offspring.shape)
        offspring = offspring + mask * noise
        return np.clip(offspring, self._lo, self._hi)

    def _bsp_learning(self, offspring: np.ndarray, pop: np.ndarray) -> np.ndarray:
        if offspring.size == 0 or self._bsp_learning_rate <= 0.0:
            return offspring
        order = self._order(pop[:, -1])
        elite_count = max(1, min(pop.shape[0], self._bsp_neighbor_count))
        elite_pos = pop[order[:elite_count], :-1]
        global_best = elite_pos[0]

        learned = offspring.copy()
        if _HAS_SCIPY and pop.shape[0] > 1:
            try:
                tree = _KDTree(pop[:, :-1])
                _, nn = tree.query(offspring, k=min(self._bsp_neighbor_count, pop.shape[0]))
                nn = np.asarray(nn)
                if nn.ndim == 1:
                    nn = nn[:, None]
                for i in range(offspring.shape[0]):
                    local_rows = pop[nn[i]]
                    local_best = local_rows[self._order(local_rows[:, -1])[0], :-1]
                    guide = 0.5 * (local_best + global_best)
                    if np.random.rand() < self._bsp_learning_rate:
                        w = np.random.rand(self.problem.dimension)
                        learned[i] = offspring[i] + w * (guide - offspring[i])
            except Exception:
                pass
        else:
            mask = np.random.rand(offspring.shape[0]) < self._bsp_learning_rate
            learned[mask] = offspring[mask] + np.random.rand(np.sum(mask), self.problem.dimension) * (global_best - offspring[mask])
        return np.clip(learned, self._lo, self._hi)

    def initialize(self) -> EngineState:
        pos = self._initial_positions()
        fit = self._evaluate_positions(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi = self._order(fit)[0]
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=pos[bi].tolist(),
            best_fitness=float(fit[bi]),
            initialized=True,
            payload=dict(population=pop),
        )

    def step(self, state: EngineState) -> EngineState:
        pop = np.asarray(state.payload["population"], dtype=float).copy()
        N = pop.shape[0]
        elite_n = max(1, int(round(self._elite_fraction * N)))
        order = self._order(pop[:, -1])
        elites = pop[order[:elite_n]].copy()

        offspring = []
        while len(offspring) < N - elite_n:
            p1 = self._tournament_select(pop)
            p2 = self._tournament_select(pop)
            c1, c2 = self._blend_crossover(p1, p2)
            offspring.append(c1)
            if len(offspring) < N - elite_n:
                offspring.append(c2)
        off = np.asarray(offspring, dtype=float)
        off = self._mutate(off, state.step + 1)
        off = self._bsp_learning(off, pop)
        off_fit = self._evaluate_positions(off)
        off_pop = np.hstack((off, off_fit[:, None]))

        combined = np.vstack((elites, pop, off_pop))
        pop = combined[self._order(combined[:, -1])[:N]].copy()

        bi = self._order(pop[:, -1])[0]
        bf = float(pop[bi, -1])
        state.payload = dict(population=pop)
        state.evaluations += off.shape[0]
        state.step += 1
        if self._is_better(bf, state.best_fitness):
            state.best_fitness = bf
            state.best_position = pop[bi, :-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        pos = pop[:, :-1]
        centroid = np.mean(pos, axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / (np.linalg.norm(self._span) + 1.0e-12))
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])), std_fitness=float(np.std(pop[:, -1])),
                    diversity=diversity)

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), float(state.best_fitness), self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=float(state.best_fitness),
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time,
                                                population_size=int(state.payload["population"].shape[0]),
                                                encoding="real"))

    def get_population(self, state):
        pop = state.payload["population"]
        return [CandidateRecord(position=pop[i, :-1].tolist(), fitness=float(pop[i, -1]),
                                source_algorithm=self.algorithm_id, source_step=state.step, role="current")
                for i in range(pop.shape[0])]
