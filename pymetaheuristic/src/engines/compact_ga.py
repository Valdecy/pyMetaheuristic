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
    family = "distribution"
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
        initial_samples=None,
        local_search=True,
        local_search_scale=0.12,
        local_search_decay=0.85,
        random_immigrant_rate=0.05,
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
        requested_initial_samples = params.get("initial_samples")
        if requested_initial_samples is None:
            requested_initial_samples = max(4, 2 * self._chromosome_length)
        self._initial_samples = max(1, int(requested_initial_samples))
        self._local_search = bool(params.get("local_search", True))
        self._local_search_scale = max(0.0, float(params.get("local_search_scale", 0.12)))
        self._local_search_decay = min(max(float(params.get("local_search_decay", 0.85)), 0.0), 1.0)
        self._random_immigrant_rate = min(max(float(params.get("random_immigrant_rate", 0.05)), 0.0), 1.0)
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

    def _encode_position(self, position: np.ndarray) -> np.ndarray:
        position = np.clip(np.asarray(position, dtype=float), self._lo, self._hi)
        ratio = np.zeros(self.problem.dimension, dtype=float)
        mask = np.abs(self._span) > 1.0e-30
        ratio[mask] = (position[mask] - self._lo[mask]) / self._span[mask]
        codes = np.rint(np.clip(ratio, 0.0, 1.0) * self._max_code).astype(np.uint64)
        bit_rows = ((codes[:, None] & self._bit_weights[None, :]) > 0).astype(np.int8)
        return bit_rows.reshape(-1)

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
        # Seed the incumbent with a small unbiased random design.  Starting only
        # from the all-0.5 mean chromosome places the first solution at the
        # domain centre, which is a poor default for deceptive continuous
        # functions.  The probability vector itself remains neutral.
        samples = []
        best_position = None
        best_fitness = None
        best_bits = None
        for _ in range(self._initial_samples):
            bits = self._sample_bits(probabilities)
            pos, fit = self._evaluate_bits(bits)
            samples.append(np.hstack((pos, fit)))
            if best_fitness is None or self._is_better(fit, best_fitness):
                best_position = pos.copy()
                best_fitness = float(fit)
                best_bits = bits.copy()
        best = np.hstack((best_position, best_fitness))
        return EngineState(
            step=0,
            evaluations=self._initial_samples,
            best_position=best_position.tolist(),
            best_fitness=float(best_fitness),
            initialized=True,
            payload={
                "probabilities": probabilities,
                "best_bits": best_bits.copy(),
                "best": best,
                "last_samples": np.asarray(samples, dtype=float),
                "best_evaluation": self._initial_samples,
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
        last_samples = [np.hstack((p1, f1)), np.hstack((p2, f2))]
        if self._is_better(winner_fit, float(best[-1])):
            best = np.hstack((winner_pos, winner_fit))
            best_bits = winner_bits.copy()
            best_evaluation = new_evaluations
            state.best_position = winner_pos.tolist()
            state.best_fitness = float(winner_fit)
        elif state.best_fitness is None or self._is_better(float(best[-1]), state.best_fitness):
            state.best_position = best[:-1].tolist()
            state.best_fitness = float(best[-1])

        # Hybrid continuous refinement.  cGA remains the distribution model, but
        # one bounded local candidate around the incumbent prevents premature
        # probability-vector convergence on narrow continuous optima.
        if self._local_search and state.best_position is not None:
            entropy = -np.mean(
                probabilities * np.log2(probabilities)
                + (1.0 - probabilities) * np.log2(1.0 - probabilities)
            )
            radius = self._local_search_scale * (self._local_search_decay ** state.step)
            radius = max(1.0e-4, radius * max(float(entropy), 0.05))
            if np.random.rand() < self._random_immigrant_rate:
                local_pos = np.random.uniform(self._lo, self._hi, self.problem.dimension)
            else:
                local_pos = np.asarray(state.best_position, dtype=float) + np.random.normal(0.0, radius * self._span, self.problem.dimension)
                local_pos = np.clip(local_pos, self._lo, self._hi)
            local_fit = float(self.problem.evaluate(local_pos.copy()))
            new_evaluations += 1
            last_samples.append(np.hstack((local_pos, local_fit)))
            if self._is_better(local_fit, float(best[-1])):
                local_bits = self._encode_position(local_pos)
                best = np.hstack((local_pos, local_fit))
                best_bits = local_bits.copy()
                best_evaluation = new_evaluations
                state.best_position = local_pos.tolist()
                state.best_fitness = float(local_fit)
                # Nudge the probability vector toward the refined solution, but
                # keep bounds away from exact 0/1 to preserve future exploration.
                probabilities = probabilities + self._eta * (local_bits.astype(float) - probabilities)
                probabilities = np.clip(probabilities, self._lower_probability, 1.0 - self._lower_probability)

        state.step += 1
        state.evaluations = new_evaluations
        state.payload = {
            "probabilities": probabilities,
            "best_bits": best_bits,
            "best": best,
            "last_samples": np.asarray(last_samples, dtype=float),
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
                "initial_samples": self._initial_samples,
                "local_search": self._local_search,
                "elapsed_time": state.elapsed_time,
            },
        )
