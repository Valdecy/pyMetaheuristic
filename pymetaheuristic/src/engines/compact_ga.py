"""pyMetaheuristic src — Compact Genetic Algorithm Engine"""
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


class CompactGAEngine(BaseEngine):

    algorithm_id = "compact_ga"
    algorithm_name = "Compact Genetic Algorithm"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/4235.797971",
        "title": "The Compact Genetic Algorithm",
        "authors": "Georges R. Harik, Fernando G. Lobo, David E. Goldberg",
        "year": 1999,
    }
    capabilities = CapabilityProfile(
        has_population=False,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_discrete=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        bits_per_variable=16,
        virtual_population_size=None,
        lower_probability=None,
        max_resample_attempts=64,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        params = {**self._DEFAULTS, **config.params}
        self._bits_per_variable = int(params["bits_per_variable"])
        self._chromosome_length = int(problem.dimension * self._bits_per_variable)
        requested_virtual_pop = params.get("virtual_population_size")
        if requested_virtual_pop is None:
            requested_virtual_pop = self._chromosome_length
        self._virtual_population_size = int(requested_virtual_pop)
        requested_lower_p = params.get("lower_probability")
        if requested_lower_p is None:
            requested_lower_p = 1.0 / (2.0 * self._virtual_population_size)
        self._lower_probability = float(requested_lower_p)
        self._max_resample_attempts = int(params.get("max_resample_attempts", 64))
        if config.seed is not None:
            np.random.seed(config.seed)
        self._validate_parameters()
        self._lo = np.asarray(problem.min_values, dtype=float)
        self._hi = np.asarray(problem.max_values, dtype=float)
        self._span = self._hi - self._lo
        self._bit_weights = (2 ** np.arange(self._bits_per_variable - 1, -1, -1, dtype=np.uint64)).astype(np.uint64)
        self._max_code = float((1 << self._bits_per_variable) - 1)
        self._eta = 1.0 / float(self._virtual_population_size)

    def _validate_parameters(self) -> None:
        if self.problem.dimension <= 0:
            raise ValueError("compact_ga requires a positive problem dimension.")
        if self._bits_per_variable < 1:
            raise ValueError("compact_ga bits_per_variable must be at least 1.")
        if self._bits_per_variable > 32:
            raise ValueError("compact_ga bits_per_variable must be <= 32 to keep decoding numerically safe.")
        if self._chromosome_length < 1:
            raise ValueError("compact_ga chromosome length must be positive.")
        if self._virtual_population_size < 2:
            raise ValueError("compact_ga virtual_population_size must be at least 2.")
        if not 0.0 < self._lower_probability < 0.5:
            raise ValueError("compact_ga lower_probability must be in (0, 0.5).")
        if self._max_resample_attempts < 1:
            raise ValueError("compact_ga max_resample_attempts must be at least 1.")

    def _decode_bits(self, bits: np.ndarray) -> np.ndarray:
        bit_matrix = np.asarray(bits, dtype=np.uint8).reshape(self.problem.dimension, self._bits_per_variable)
        codes = bit_matrix.astype(np.uint64) @ self._bit_weights
        ratio = codes.astype(float) / self._max_code if self._max_code > 0.0 else np.zeros(self.problem.dimension)
        return np.clip(self._lo + ratio * self._span, self._lo, self._hi)

    def _sample_bits(self, probabilities: np.ndarray) -> np.ndarray:
        return (np.random.rand(self._chromosome_length) < probabilities).astype(np.int8)

    def _sample_pair(self, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = self._sample_bits(probabilities)
        second = self._sample_bits(probabilities)
        attempts = 0
        while np.array_equal(first, second) and attempts < self._max_resample_attempts:
            second = self._sample_bits(probabilities)
            attempts += 1
        return first, second

    def _evaluate_bits(self, bits: np.ndarray) -> tuple[np.ndarray, float]:
        position = self._decode_bits(bits)
        return position, float(self.problem.evaluate(position.copy()))

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def initialize(self) -> EngineState:
        probabilities = np.full(self._chromosome_length, 0.5, dtype=float)
        # Package-level result objects require a valid incumbent.  We evaluate
        # the distribution mean once; the native cGA pairwise update starts in
        # the first call to step().
        mean_bits = (probabilities >= 0.5).astype(np.int8)
        best_position, best_fitness = self._evaluate_bits(mean_bits)
        best = np.hstack((best_position, best_fitness))
        return EngineState(
            step=0,
            evaluations=1,
            best_position=best_position.tolist(),
            best_fitness=float(best_fitness),
            initialized=True,
            payload={
                "probabilities": probabilities,
                "best_bits": mean_bits.copy(),
                "best": best,
                "last_samples": np.empty((0, self.problem.dimension + 1), dtype=float),
                "best_evaluation": 1,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        probabilities = np.asarray(state.payload["probabilities"], dtype=float).copy()
        x1, x2 = self._sample_pair(probabilities)
        p1, f1 = self._evaluate_bits(x1)
        p2, f2 = self._evaluate_bits(x2)

        if self._is_better(f2, f1):
            winner_bits, winner_pos, winner_fit = x2, p2, f2
            loser_bits = x1
        else:
            winner_bits, winner_pos, winner_fit = x1, p1, f1
            loser_bits = x2

        if f1 != f2:
            probabilities = probabilities + self._eta * (winner_bits.astype(float) - loser_bits.astype(float))
            probabilities = np.clip(probabilities, self._lower_probability, 1.0 - self._lower_probability)

        best = np.asarray(state.payload["best"], dtype=float).copy()
        best_bits = np.asarray(state.payload["best_bits"], dtype=np.int8).copy()
        best_evaluation = int(state.payload.get("best_evaluation", state.evaluations))
        new_evaluations = state.evaluations + 2
        if self._is_better(winner_fit, float(best[-1])):
            best = np.hstack((winner_pos, winner_fit))
            best_bits = winner_bits.copy()
            best_evaluation = new_evaluations
            state.best_position = winner_pos.tolist()
            state.best_fitness = float(winner_fit)
        elif state.best_fitness is None or self._is_better(float(best[-1]), state.best_fitness):
            state.best_position = best[:-1].tolist()
            state.best_fitness = float(best[-1])

        state.step += 1
        state.evaluations = new_evaluations
        state.payload = {
            "probabilities": probabilities,
            "best_bits": best_bits,
            "best": best,
            "last_samples": np.vstack((np.hstack((p1, f1)), np.hstack((p2, f2)))),
            "best_evaluation": best_evaluation,
        }

        # Match the minimal pycma stop convention: stop after a long period
        # without improving the best sampled solution.
        stagnation_limit = 2000 * self._chromosome_length
        if state.evaluations > best_evaluation + stagnation_limit:
            state.terminated = True
            state.termination_reason = "stagnation"
        return state

    def observe(self, state: EngineState) -> dict:
        probabilities = np.asarray(state.payload["probabilities"], dtype=float)
        entropy = -(
            probabilities * np.log2(probabilities)
            + (1.0 - probabilities) * np.log2(1.0 - probabilities)
        )
        samples = np.asarray(state.payload.get("last_samples", np.empty((0, self.problem.dimension + 1))), dtype=float)
        obs = {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "probability_entropy": float(np.mean(entropy)),
            "mean_probability": float(np.mean(probabilities)),
        }
        if samples.size:
            obs["current_best_fitness"] = float(samples[:, -1].min() if self.problem.objective == "min" else samples[:, -1].max())
        return obs

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
            metadata={"best_bits": np.asarray(state.payload["best_bits"], dtype=int).tolist()},
        )

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
                "bits_per_variable": self._bits_per_variable,
                "chromosome_length": self._chromosome_length,
                "virtual_population_size": self._virtual_population_size,
                "elapsed_time": state.elapsed_time,
            },
        )
