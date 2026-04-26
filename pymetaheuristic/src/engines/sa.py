"""
pyMetaheuristic src — Simulated Annealing Engine
================================================
Reference implementation of a *single-solution / trajectory* engine.

Native macro-step:  one temperature epoch
    for each Markov iteration:
        perturb  →  evaluate  →  accept / reject

payload keys
------------
    current     : list[float]  current solution (without fitness)
    current_fit : float
    temperature : float
    accepted    : int          accepted moves this step
    rejected    : int
"""

from __future__ import annotations

import math
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

# Resolve the sibling import whether used as package or standalone
try:
    from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                           EngineConfig, EngineState, OptimizationResult, ProblemSpec)
except ImportError:
    from protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                          EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SAEngine(BaseEngine):
    """Simulated Annealing — native engine."""

    algorithm_id   = "sa"
    algorithm_name = "Simulated Annealing"
    family         = "trajectory"
    _REFERENCE     = {"doi": "10.1126/science.220.4598.671"}
    capabilities   = CapabilityProfile(
        has_population               = False,
        has_archive                  = False,
        supports_candidate_injection = True,   # restart around migrant
        supports_restart             = True,
        supports_checkpoint          = True,
        supports_constraints         = False,
        supports_diversity_metrics   = False,
    )

    _DEFAULTS = dict(
        initial_temperature   = 1.0,
        final_temperature     = 1e-4,
        alpha                 = 0.9,
        temperature_iterations= 100,
        mu                    = 0.0,
        sigma                 = 1.0,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._T0        = float(p["initial_temperature"])
        self._Tf        = float(p["final_temperature"])
        self._alpha     = float(p["alpha"])
        self._T_iters   = int(p["temperature_iterations"])
        self._mu        = float(p["mu"])
        self._sigma     = float(p["sigma"])
        if config.seed is not None:
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        prob = self.problem
        lo   = np.array(prob.min_values, dtype=float)
        hi   = np.array(prob.max_values, dtype=float)

        current     = np.random.uniform(lo, hi)
        current_fit = float(prob.evaluate(current))

        state = EngineState(
            step          = 0,
            evaluations   = 1,
            best_position = current.tolist(),
            best_fitness  = current_fit,
            initialized   = True,
            payload       = dict(
                current     = current.tolist(),
                current_fit = current_fit,
                temperature = self._T0,
                accepted    = 0,
                rejected    = 0,
            ),
        )
        return state

    def step(self, state: EngineState) -> EngineState:
        """One temperature epoch: run Markov chain, then cool."""
        pl   = state.payload
        prob = self.problem
        lo   = np.array(prob.min_values, dtype=float)
        hi   = np.array(prob.max_values, dtype=float)

        current     = np.array(pl["current"])
        current_fit = pl["current_fit"]
        temperature = pl["temperature"]
        accepted = rejected = 0

        for _ in range(self._T_iters):
            eps = np.random.normal(self._mu, self._sigma, prob.dimension)
            candidate = np.clip(current + eps, lo, hi)
            cand_fit  = float(prob.evaluate(candidate))
            delta     = cand_fit - current_fit

            if delta < 0 or np.random.rand() < math.exp(-delta / max(temperature, 1e-300)):
                current     = candidate
                current_fit = cand_fit
                accepted   += 1
            else:
                rejected += 1

            # update best
            if prob.is_better(cand_fit, state.best_fitness):
                state.best_fitness  = cand_fit
                state.best_position = candidate.tolist()

        # cool
        temperature *= self._alpha

        # detect natural termination (temperature exhausted)
        if temperature <= self._Tf:
            state.terminated         = True
            state.termination_reason = "temperature_exhausted"

        state.step        += 1
        state.evaluations += self._T_iters
        state.payload = dict(
            current     = current.tolist(),
            current_fit = current_fit,
            temperature = temperature,
            accepted    = accepted,
            rejected    = rejected,
        )
        return state

    def observe(self, state: EngineState) -> dict:
        pl = state.payload
        total = pl["accepted"] + pl["rejected"]
        return {
            "step":          state.step,
            "evaluations":   state.evaluations,
            "best_fitness":  state.best_fitness,
            "current_fitness": pl["current_fit"],
            "temperature":   pl["temperature"],
            "accepted_ratio": pl["accepted"] / total if total else 0.0,
            "phase":         "cooling",
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position         = list(state.best_position),
            fitness          = state.best_fitness,
            source_algorithm = self.algorithm_id,
            source_step      = state.step,
            role             = "best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id       = self.algorithm_id,
            best_position      = list(state.best_position),
            best_fitness       = state.best_fitness,
            steps              = state.step,
            evaluations        = state.evaluations,
            termination_reason = state.termination_reason,
            capabilities       = self.capabilities,
            metadata           = {
                "algorithm_name":   self.algorithm_name,
                "final_temperature": state.payload["temperature"],
                "elapsed_time":     state.elapsed_time,
            },
        )

    # ------------------------------------------------------------------
    # Candidate exchange
    # ------------------------------------------------------------------

    def export_candidates(
        self,
        state: EngineState,
        k: int = 1,
        mode: str = "best",
    ) -> list[CandidateRecord]:
        """Single-solution: always exports best (and optionally current)."""
        out = [self.get_best_candidate(state)]
        if k >= 2 and mode in ("current", "diverse"):
            pl = state.payload
            out.append(CandidateRecord(
                position         = list(pl["current"]),
                fitness          = pl["current_fit"],
                source_algorithm = self.algorithm_id,
                source_step      = state.step,
                role             = "current",
            ))
        return out[:k]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        """
        Accept migrant if it is better than current;
        otherwise restart SA around migrant position.
        """
        if not candidates:
            return state
        best_migrant = min(candidates, key=lambda c: c.fitness)
        prob = self.problem
        pos  = np.clip(best_migrant.position, prob.min_values, prob.max_values)
        fit  = float(prob.evaluate(pos))
        state.evaluations += 1

        if prob.is_better(fit, state.payload["current_fit"]):
            state.payload["current"]      = list(pos)
            state.payload["current_fit"]  = fit

        if prob.is_better(fit, state.best_fitness):
            state.best_fitness  = fit
            state.best_position = list(pos)

        return state
