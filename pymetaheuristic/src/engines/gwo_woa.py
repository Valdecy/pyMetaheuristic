from __future__ import annotations

import math
from typing import Any

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


class GWOWOAEngine(BaseEngine):

    algorithm_id = 'gwo_woa'
    algorithm_name = 'Hybrid Grey Wolf - Whale Optimization Algorithm (GWO-WOA)'
    family = 'swarm'
    _REFERENCE = {'variant': 'GWO_WOA', 'base_family': 'GWO', 'doi': '10.1177/10775463211003402'}
    _DEFAULTS = {"population_size": 100}
    _STRATEGY = 'leader'
    _VARIANT_CODE = 819512

    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = int(p.get("population_size", p.get("pop_size", p.get("pack_size", 50))))
        if self._n < 4:
            raise ValueError("population_size must be >= 4 for this algorithm.")
        self._elite_ratio = float(p.get("elite_ratio", 0.20))
        if not (0.0 < self._elite_ratio <= 0.5):
            raise ValueError("elite_ratio must be in (0, 0.5].")
        self._mutation = float(p.get("mutation_rate", 0.10))
        if self._mutation < 0.0:
            raise ValueError("mutation_rate must be non-negative.")
        self._local_trials = int(p.get("local_trials", max(4, min(12, self._n // 4))))
        if self._local_trials < 0:
            raise ValueError("local_trials must be non-negative.")
        self._memory = float(p.get("memory_rate", 0.35))
        if not (0.0 <= self._memory <= 1.0):
            raise ValueError("memory_rate must be in [0, 1].")
        self._variant_phase = (int(self._VARIANT_CODE) % 997) / 997.0
        if config.seed is not None:
            np.random.seed(int(config.seed))

    @staticmethod
    def _halton(size: int, dim: int, offset: int = 0) -> np.ndarray:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        while len(primes) < dim:
            candidate = primes[-1] + 2
            while any(candidate % p == 0 for p in primes if p * p <= candidate):
                candidate += 2
            primes.append(candidate)
        out = np.empty((size, dim), dtype=float)
        for j in range(dim):
            base = primes[j]
            for i in range(size):
                n = i + 1 + offset
                f = 1.0 / base
                value = 0.0
                while n > 0:
                    value += f * (n % base)
                    n //= base
                    f /= base
                out[i, j] = value
        return out

    def _bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        span = np.maximum(hi - lo, 1e-12)
        return lo, hi, span

    def _init_positions(self) -> np.ndarray:
        lo, hi, span = self._bounds()
        random_part = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        n_q = max(1, self._n // 2)
        q = self._halton(n_q, self.problem.dimension, offset=int(self._VARIANT_CODE) % 31)
        random_part[:n_q, :] = lo + q * span
        random_part[0, :] = (lo + hi) / 2.0
        return np.clip(random_part, lo, hi)

    def initialize(self) -> EngineState:
        positions = self._init_positions()
        fitness = self._evaluate_population(positions)
        population = np.hstack((positions, fitness[:, None]))
        order = self._order(population[:, -1])
        population = population[order]
        elite = population[0, :].copy()
        velocity = np.zeros_like(positions)
        personal_best = population.copy()
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload={
                "population": population,
                "elite": elite,
                "velocity": velocity,
                "personal_best": personal_best,
            },
        )

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _is_better_array(self, trial: np.ndarray, current: np.ndarray) -> np.ndarray:
        return trial < current if self.problem.objective == "min" else trial > current

    def _ranked(self, population: np.ndarray) -> np.ndarray:
        return population[self._order(population[:, -1])]

    def _leaders(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ranked = self._ranked(population)
        a = ranked[0, :-1]
        b = ranked[min(1, len(ranked) - 1), :-1]
        c = ranked[min(2, len(ranked) - 1), :-1]
        return a, b, c

    def _progress(self, state: EngineState) -> float:
        horizon = max(1, int(self.config.max_steps or 1))
        return min(1.0, max(0.0, state.step / horizon))

    def _differential_trials(self, positions: np.ndarray, scale: float, crossover: float) -> np.ndarray:
        n, dim = positions.shape
        trials = positions.copy()
        for i in range(n):
            ids = np.random.choice(n, size=3, replace=False)
            mutant = positions[ids[0]] + scale * (positions[ids[1]] - positions[ids[2]])
            mask = np.random.rand(dim) < crossover
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            trials[i] = np.where(mask, mutant, positions[i])
        return trials

    def _leader_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        alpha, beta, delta = self._leaders(population)
        n, dim = positions.shape
        a = 2.0 * (1.0 - progress)
        r1 = np.random.rand(n, dim)
        r2 = np.random.rand(n, dim)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2
        x1 = alpha - A1 * np.abs(C1 * alpha - positions)
        r1 = np.random.rand(n, dim)
        r2 = np.random.rand(n, dim)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2
        x2 = beta - A2 * np.abs(C2 * beta - positions)
        r1 = np.random.rand(n, dim)
        r2 = np.random.rand(n, dim)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2
        x3 = delta - A3 * np.abs(C3 * delta - positions)
        phase = math.sin(math.pi * (progress + self._variant_phase))
        return (x1 + x2 + x3) / 3.0 + 0.05 * phase * (beta - delta)

    def _pso_trials(self, positions: np.ndarray, population: np.ndarray, personal_best: np.ndarray, velocity: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray]:
        lo, hi, span = self._bounds()
        best = population[self._order(population[:, -1])[0], :-1]
        w = 0.9 - 0.5 * progress
        c1 = 1.3 + 0.4 * self._variant_phase
        c2 = 1.7 + 0.3 * (1.0 - self._variant_phase)
        r1 = np.random.rand(*positions.shape)
        r2 = np.random.rand(*positions.shape)
        v_new = w * velocity + c1 * r1 * (personal_best[:, :-1] - positions) + c2 * r2 * (best - positions)
        vmax = 0.35 * span
        v_new = np.clip(v_new, -vmax, vmax)
        return positions + v_new, v_new

    def _local_trials_around_best(self, best: np.ndarray, progress: float) -> np.ndarray:
        if self._local_trials <= 0:
            return np.empty((0, best.size), dtype=float)
        lo, hi, span = self._bounds()
        radius = (0.25 * (1.0 - progress) ** 2 + 0.002) * span
        noise = np.random.normal(0.0, 1.0, (self._local_trials, best.size)) * radius
        cauchy = np.random.standard_cauchy((self._local_trials, best.size)) * (0.01 * (1.0 - progress) + 0.0005) * span
        trials = best + noise + cauchy
        return np.clip(trials, lo, hi)

    def _memory_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        ranked = self._ranked(population)
        m = max(2, int(round(self._elite_ratio * len(ranked))))
        elites = ranked[:m, :-1]
        chosen = elites[np.random.randint(0, m, positions.shape[0])]
        lo, hi, span = self._bounds()
        bandwidth = (0.20 * (1.0 - progress) + 0.005) * span
        return chosen + np.random.normal(0.0, 1.0, positions.shape) * bandwidth

    def _human_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        ranked = self._ranked(population)
        n = positions.shape[0]
        teacher = ranked[0, :-1]
        mean = np.mean(positions, axis=0)
        peer = positions[np.random.permutation(n)]
        factor = 1.0 + np.random.randint(1, 3, size=(n, 1))
        learn = np.random.rand(n, 1) * (teacher - factor * mean)
        exchange = np.random.rand(n, 1) * (positions - peer)
        return positions + learn - 0.5 * exchange * (1.0 - progress)

    def _physics_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        ranked = self._ranked(population)
        pool = ranked[: min(5, len(ranked)), :-1]
        eq = pool[np.random.randint(0, len(pool), positions.shape[0])]
        lam = np.random.rand(*positions.shape)
        force = np.exp(-lam * (4.0 * progress + 0.5))
        return eq + (positions - eq) * force + np.random.normal(0.0, 0.05 * (1.0 - progress), positions.shape) * (np.max(positions, axis=0) - np.min(positions, axis=0) + 1e-12)

    def _trig_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        best = self._ranked(population)[0, :-1]
        r = 2.0 * (1.0 - progress)
        angle = 2.0 * math.pi * np.random.rand(*positions.shape)
        coeff = np.where(np.random.rand(*positions.shape) < 0.5, np.sin(angle), np.cos(angle))
        return positions + r * coeff * np.abs(np.random.rand(*positions.shape) * best - positions)

    def _nature_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        best = self._ranked(population)[0, :-1]
        lo, hi, span = self._bounds()
        food = best + np.random.normal(0.0, (0.18 * (1.0 - progress) + 0.004), positions.shape) * span
        drift = np.random.rand(*positions.shape) * (food - positions)
        opposition = lo + hi - positions
        mask = np.random.rand(positions.shape[0], 1) < 0.25
        return np.where(mask, opposition, positions + drift)

    def _ga_trials(self, positions: np.ndarray, population: np.ndarray, progress: float) -> np.ndarray:
        ranked = self._ranked(population)
        n, dim = positions.shape
        m = max(2, n // 3)
        parents_a = ranked[np.random.randint(0, m, n), :-1]
        parents_b = ranked[np.random.randint(0, n, n), :-1]
        blend = np.random.rand(n, dim)
        child = blend * parents_a + (1.0 - blend) * parents_b
        lo, hi, span = self._bounds()
        sigma = (0.12 * (1.0 - progress) + 0.002) * span
        mutation_mask = np.random.rand(n, dim) < max(self._mutation, 1.0 / max(1, dim))
        child = np.where(mutation_mask, child + np.random.normal(0.0, sigma, (n, dim)), child)
        return child

    def _select_strategy_trials(self, positions: np.ndarray, population: np.ndarray, personal_best: np.ndarray, velocity: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray]:
        strategy = str(self._STRATEGY).lower()
        if strategy == "pso":
            return self._pso_trials(positions, population, personal_best, velocity, progress)
        if strategy == "de":
            return self._differential_trials(positions, 0.45 + 0.35 * (1.0 - progress), 0.85), velocity
        if strategy == "ga" or strategy == "evolution":
            return self._ga_trials(positions, population, progress), velocity
        if strategy == "local":
            return self._memory_trials(positions, population, progress), velocity
        if strategy == "physics":
            return self._physics_trials(positions, population, progress), velocity
        if strategy == "trig":
            return self._trig_trials(positions, population, progress), velocity
        if strategy == "human":
            return self._human_trials(positions, population, progress), velocity
        if strategy == "nature":
            return self._nature_trials(positions, population, progress), velocity
        if strategy == "memory":
            return self._memory_trials(positions, population, progress), velocity
        if strategy == "swarm":
            return self._nature_trials(positions, population, progress), velocity
        return self._leader_trials(positions, population, progress), velocity

    def step(self, state: EngineState) -> EngineState:
        lo, hi, span = self._bounds()
        population = np.asarray(state.payload["population"], dtype=float)
        positions = population[:, :-1]
        velocity = np.asarray(state.payload.get("velocity", np.zeros_like(positions)), dtype=float)
        personal_best = np.asarray(state.payload.get("personal_best", population.copy()), dtype=float)
        progress = self._progress(state)

        primary, velocity = self._select_strategy_trials(positions, population, personal_best, velocity, progress)
        leader = self._leader_trials(positions, population, progress)
        diff = self._differential_trials(positions, 0.40 + 0.30 * (1.0 - progress), 0.80)
        memory = self._memory_trials(positions, population, progress)
        weights = np.array([0.45, 0.25, 0.20, 0.10], dtype=float)
        shift = int(self._VARIANT_CODE) % 4
        weights = np.roll(weights, shift)
        mixed = weights[0] * primary + weights[1] * leader + weights[2] * diff + weights[3] * memory
        explore_mask = np.random.rand(*mixed.shape) < (0.06 * (1.0 - progress) + 0.01)
        random_pos = np.random.uniform(lo, hi, mixed.shape)
        mixed = np.where(explore_mask, random_pos, mixed)
        mixed = np.clip(mixed, lo, hi)

        trial_fitness = self._evaluate_population(mixed)
        evals = mixed.shape[0]
        improved = self._is_better_array(trial_fitness, population[:, -1])
        population[improved, :-1] = mixed[improved]
        population[improved, -1] = trial_fitness[improved]

        p_improved = self._is_better_array(population[:, -1], personal_best[:, -1])
        personal_best[p_improved] = population[p_improved]

        best_row = self._ranked(population)[0, :].copy()
        local_positions = self._local_trials_around_best(best_row[:-1], progress)
        if local_positions.size:
            local_fitness = self._evaluate_population(local_positions)
            evals += local_positions.shape[0]
            local_pop = np.hstack((local_positions, local_fitness[:, None]))
            population = np.vstack((population, local_pop))

        population = self._ranked(population)[: self._n, :]
        elite = np.asarray(state.payload.get("elite", population[0, :].copy()), dtype=float)
        if self.problem.is_better(float(population[0, -1]), float(elite[-1])):
            elite = population[0, :].copy()
        elif not self.problem.is_better(float(elite[-1]), float(population[-1, -1])):
            population[-1, :] = elite.copy()
            population = self._ranked(population)

        state.step += 1
        state.evaluations += int(evals)
        state.payload = {
            "population": population,
            "elite": elite,
            "velocity": velocity[: self._n, :],
            "personal_best": self._ranked(np.vstack((personal_best, population)))[: self._n, :],
        }
        if state.best_fitness is None or self.problem.is_better(float(elite[-1]), float(state.best_fitness)):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        pop = np.asarray(state.payload["population"], dtype=float)
        pos = pop[:, :-1]
        fit = pop[:, -1]
        lo, hi, span = self._bounds()
        centroid = np.mean(pos, axis=0)
        denom = float(np.linalg.norm(span)) or 1.0
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "mean_fitness": float(np.mean(fit)),
            "std_fitness": float(np.std(fit)),
            "diversity": diversity,
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
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

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "family": self.family,
                "elapsed_time": float(state.elapsed_time),
            },
        )
