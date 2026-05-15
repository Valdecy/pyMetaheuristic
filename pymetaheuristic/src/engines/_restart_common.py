"""Shared restart/metaheuristic helpers for continuous optimizers.

This module contains reusable building blocks for restart-oriented
single-trajectory optimizers and compact restart CMA-ES variants.

Design notes
------------
* Evaluation budgets are treated as hard limits whenever ``max_evaluations``
  is supplied through ``EngineConfig``.
* Randomness is engine-local via ``numpy.random.Generator``.  This avoids
  global RNG interference among islands/cooperative runs.
* Candidate injection is implemented for these restart engines so they can
  participate as receivers in cooperative/orchestrated island systems.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


def _as_array(values) -> np.ndarray:
    """Convert a sequence-like object to a float NumPy array."""
    return np.asarray(values, dtype=float)


class RestartLocalSearchEngine(BaseEngine):
    """Base class for restart-oriented single-trajectory continuous methods."""

    _DEFAULTS: dict[str, Any] = dict(
        local_search_steps=12,
        neighborhood_size=None,
        step_size=0.12,
        contraction=0.55,
        expansion=1.05,
        min_step_size=1.0e-8,
        restart_stagnation_steps=20,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        self._rng = np.random.default_rng(config.seed)
        self._validate_common_parameters()

    @property
    def _lo(self) -> np.ndarray:
        return _as_array(self.problem.min_values)

    @property
    def _hi(self) -> np.ndarray:
        return _as_array(self.problem.max_values)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _remaining_evaluations(self, state: EngineState | None = None, used: int = 0) -> int | None:
        """Return remaining evaluations under the configured hard budget.

        ``None`` means no explicit evaluation budget is active.
        """
        if self.config.max_evaluations is None:
            return None
        done = int(getattr(state, "evaluations", 0) if state is not None else 0)
        return max(0, int(self.config.max_evaluations) - done - int(used))

    def _budget_allows(self, state: EngineState | None = None, used: int = 0) -> bool:
        remaining = self._remaining_evaluations(state, used=used)
        return remaining is None or remaining > 0

    def _validate_common_parameters(self) -> None:
        if int(self._params.get("local_search_steps", 12)) < 0:
            raise ValueError(f"{self.algorithm_id} local_search_steps must be >= 0.")
        nbh = self._params.get("neighborhood_size", None)
        if nbh is not None and int(nbh) < 1:
            raise ValueError(f"{self.algorithm_id} neighborhood_size must be >= 1.")
        for key in ("step_size", "contraction", "expansion", "min_step_size"):
            if float(self._params.get(key)) <= 0.0:
                raise ValueError(f"{self.algorithm_id} {key} must be positive.")
        if int(self._params.get("restart_stagnation_steps", 20)) < 1:
            raise ValueError(f"{self.algorithm_id} restart_stagnation_steps must be >= 1.")

    def _random_position(self) -> np.ndarray:
        return self._rng.uniform(self._lo, self._hi, self.problem.dimension)

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=float), self._lo, self._hi)

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _energy(self, fitness: float) -> float:
        return float(fitness) if self.problem.objective == "min" else -float(fitness)

    def _best_index(self, fit: np.ndarray) -> int:
        return int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))

    def _substantial_improvement(self, new_fit: float, old_fit: float, tol: float = 1.0e-12) -> bool:
        if not self._is_better(new_fit, old_fit):
            return False
        scale = 1.0 + abs(float(old_fit))
        if self.problem.objective == "min":
            return (float(old_fit) - float(new_fit)) > tol * scale
        return (float(new_fit) - float(old_fit)) > tol * scale

    def _local_search(
        self,
        start: np.ndarray,
        start_fit: float | None = None,
        step_size: float | None = None,
        max_steps: int | None = None,
        neighborhood_size: int | None = None,
        max_evaluations: int | None = None,
    ) -> tuple[np.ndarray, float, int, float]:
        """Run a bounded stochastic local search from ``start``.

        Parameters
        ----------
        max_evaluations:
            Optional hard cap for this local-search call.  If supplied, this
            method never evaluates more than that number of objective calls.
        """
        if max_evaluations is not None:
            max_evaluations = max(0, int(max_evaluations))

        current = self._clip(start)
        evals = 0
        delta = float(self._params.get("step_size", 0.12) if step_size is None else step_size)

        if start_fit is None:
            if max_evaluations is not None and max_evaluations <= 0:
                return current, self.problem.worst_fitness(), 0, float(delta)
            current_fit = float(self.problem.evaluate(current))
            evals += 1
        else:
            current_fit = float(start_fit)

        max_steps = int(self._params.get("local_search_steps", 12) if max_steps is None else max_steps)
        if neighborhood_size is None:
            nbh = self._params.get("neighborhood_size", None)
            neighborhood_size = int(nbh) if nbh is not None else max(4, 2 * self.problem.dimension)
        neighborhood_size = max(1, int(neighborhood_size))

        contraction = float(self._params.get("contraction", 0.55))
        expansion = float(self._params.get("expansion", 1.05))
        min_step = float(self._params.get("min_step_size", 1.0e-8))

        for _ in range(max_steps):
            if delta <= min_step:
                break
            if max_evaluations is not None:
                remaining = max_evaluations - evals
                if remaining <= 0:
                    break
                n_trials = min(neighborhood_size, remaining)
            else:
                n_trials = neighborhood_size
            if n_trials <= 0:
                break

            trials = self._clip(
                current
                + self._rng.normal(0.0, delta, size=(n_trials, self.problem.dimension)) * self._span
            )
            fit = np.asarray([float(self.problem.evaluate(row)) for row in trials], dtype=float)
            evals += int(n_trials)
            idx = self._best_index(fit)
            if self._is_better(float(fit[idx]), current_fit):
                current = trials[idx].copy()
                current_fit = float(fit[idx])
                delta *= expansion
            else:
                delta *= contraction
        return current, float(current_fit), int(evals), float(delta)

    def _new_local_optimum(self, max_evaluations: int | None = None) -> tuple[np.ndarray, float, int, float]:
        return self._local_search(self._random_position(), max_evaluations=max_evaluations)

    def initialize(self) -> EngineState:
        budget = self._remaining_evaluations(None)
        pos, fit, evals, delta = self._new_local_optimum(max_evaluations=budget)
        return EngineState(
            step=0,
            evaluations=evals,
            best_position=pos.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload={
                "current": pos,
                "current_fit": float(fit),
                "delta": float(delta),
                "stagnation": 0,
                "restarts": 0,
                "last_accepted": True,
            },
        )

    def observe(self, state: EngineState) -> dict:
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "current_fitness": float(state.payload.get("current_fit", state.best_fitness)),
            "restarts": int(state.payload.get("restarts", 0)),
            "stagnation": int(state.payload.get("stagnation", 0)),
            "delta": float(state.payload.get("delta", self._params.get("step_size", 0.12))),
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def export_candidates(self, state: EngineState, k: int = 1, mode: str = "best") -> list[CandidateRecord]:
        return [self.get_best_candidate(state)]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        """Inject migrant candidates by locally improving the best incoming seed.

        This gives trajectory/restart methods a native receiver behaviour in
        island models.  The incoming candidate is clipped, evaluated/improved
        within the remaining evaluation budget, and accepted if it improves the
        current trajectory or the global best-so-far.
        """
        if not candidates:
            return state
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state

        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
        seed = self._clip(np.asarray(ranked[0].position, dtype=float))
        pos, fit, evals, delta = self._local_search(seed, max_evaluations=remaining)
        if evals <= 0:
            return state

        current_fit = float(state.payload.get("current_fit", state.best_fitness))
        accepted = self._is_better(fit, current_fit)
        if accepted:
            state.payload["current"] = pos
            state.payload["current_fit"] = float(fit)
            state.payload["delta"] = float(delta)
            state.payload["stagnation"] = 0
            state.payload["last_accepted"] = True
        if self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)
            state.payload["stagnation"] = 0
        state.evaluations += int(evals)
        state.payload.setdefault("injections", 0)
        state.payload["injections"] = int(state.payload["injections"]) + 1
        return state

    def restart(
        self,
        state: EngineState,
        seeds: list[CandidateRecord] | None = None,
        preserve_best: bool = True,
    ) -> EngineState:
        """Restart the current trajectory, optionally around incoming seeds."""
        if seeds:
            return self.inject_candidates(state, seeds, policy="native")
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        pos, fit, evals, delta = self._new_local_optimum(max_evaluations=remaining)
        if evals <= 0:
            return state
        state.evaluations += int(evals)
        state.payload.update(
            current=pos,
            current_fit=float(fit),
            delta=float(delta),
            restarts=int(state.payload.get("restarts", 0)) + 1,
            stagnation=0,
            last_accepted=True,
        )
        if (not preserve_best) or self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)
        return state

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
                "restarts": int(state.payload.get("restarts", 0)),
                "injections": int(state.payload.get("injections", 0)),
                "elapsed_time": state.elapsed_time,
            },
        )

    def _restart_current(
        self,
        state: EngineState,
        max_evaluations: int | None = None,
    ) -> tuple[np.ndarray, float, int, float]:
        if max_evaluations is None:
            max_evaluations = self._remaining_evaluations(state)
        pos, fit, evals, delta = self._new_local_optimum(max_evaluations=max_evaluations)
        if evals <= 0:
            current = np.asarray(state.payload.get("current", state.best_position), dtype=float)
            current_fit = float(state.payload.get("current_fit", state.best_fitness))
            return current, current_fit, 0, float(state.payload.get("delta", self._params.get("step_size", 0.12)))
        state.payload["restarts"] = int(state.payload.get("restarts", 0)) + 1
        state.payload["stagnation"] = 0
        if self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)
        return pos, fit, evals, delta


class RestartCMAESBase(BaseEngine):
    """CMA-ES with native independent restarts.

    This is a compact bounded CMA-ES core.  Bound handling is deliberately
    simple and clipping-based, which is robust for framework integration but
    should be documented as a simplified bounded CMA-ES treatment.
    """

    _DEFAULTS: dict[str, Any] = dict(
        population_size=20,
        sigma0=None,
        stagnation_steps=25,
        restart_after_steps=60,
        improvement_tol=1.0e-12,
        population_multiplier=2.0,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        self._rng = np.random.default_rng(config.seed)
        self._base_lambda = max(4, int(self._params.get("population_size", 20)))
        self._stagnation_steps = max(1, int(self._params.get("stagnation_steps", 25)))
        self._restart_after_steps = max(1, int(self._params.get("restart_after_steps", 60)))
        self._improvement_tol = max(0.0, float(self._params.get("improvement_tol", 1.0e-12)))
        self._population_multiplier = max(1.01, float(self._params.get("population_multiplier", 2.0)))

    @property
    def _lo(self) -> np.ndarray:
        return _as_array(self.problem.min_values)

    @property
    def _hi(self) -> np.ndarray:
        return _as_array(self.problem.max_values)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _remaining_evaluations(self, state: EngineState | None = None, used: int = 0) -> int | None:
        if self.config.max_evaluations is None:
            return None
        done = int(getattr(state, "evaluations", 0) if state is not None else 0)
        return max(0, int(self.config.max_evaluations) - done - int(used))

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=float), self._lo, self._hi)

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _best_index(self, fit: np.ndarray) -> int:
        return int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))

    def _order(self, fit: np.ndarray) -> np.ndarray:
        idx = np.argsort(fit)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _substantial_improvement(self, new_fit: float, old_fit: float) -> bool:
        if not self._is_better(new_fit, old_fit):
            return False
        scale = 1.0 + abs(float(old_fit))
        if self.problem.objective == "min":
            return (float(old_fit) - float(new_fit)) > self._improvement_tol * scale
        return (float(new_fit) - float(old_fit)) > self._improvement_tol * scale

    def _sample_initial_sigma(self) -> float:
        sigma0 = self._params.get("sigma0", None)
        if sigma0 is not None:
            return max(1.0e-12, float(sigma0))
        return max(1.0e-12, 0.25 * float(np.mean(self._span)))

    def _constants(self, dim: int, lam: int) -> dict[str, Any]:
        mu = max(1, lam // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights / np.sum(weights)
        mueff = float(1.0 / np.sum(weights ** 2))
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        return dict(mu=mu, weights=weights, mueff=mueff, cc=cc, cs=cs, c1=c1, cmu=cmu, damps=damps, chi_n=chi_n)

    def _next_lambda(self, restart_index: int, payload: dict[str, Any] | None = None) -> int:
        return int(round(self._base_lambda * (self._population_multiplier ** max(0, restart_index))))

    def _start_run_payload(
        self,
        restart_index: int,
        payload: dict[str, Any] | None = None,
        max_evaluations: int | None = None,
        seed_position: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], int, np.ndarray, float]:
        dim = self.problem.dimension
        lam = max(4, int(self._next_lambda(restart_index, payload)))
        const = self._constants(dim, lam)
        mean = self._clip(seed_position) if seed_position is not None else self._rng.uniform(self._lo, self._hi, dim)
        sigma = self._sample_initial_sigma()
        if max_evaluations is not None and int(max_evaluations) <= 0:
            fit = self.problem.worst_fitness()
            evals = 0
        else:
            fit = float(self.problem.evaluate(mean))
            evals = 1
        run_payload = {
            "mean": mean,
            "sigma": float(sigma),
            "C": np.eye(dim),
            "pc": np.zeros(dim),
            "ps": np.zeros(dim),
            "population": np.hstack((mean.reshape(1, -1), np.array([[fit]], dtype=float))),
            "lambda": lam,
            "mu": const["mu"],
            "weights": const["weights"],
            "mueff": const["mueff"],
            "cc": const["cc"],
            "cs": const["cs"],
            "c1": const["c1"],
            "cmu": const["cmu"],
            "damps": const["damps"],
            "chi_n": const["chi_n"],
            "run_step": 0,
            "run_evaluations": int(evals),
            "stagnation": 0,
            "restarts": int(restart_index),
            "injections": int((payload or {}).get("injections", 0)),
        }
        return run_payload, int(evals), mean, float(fit)

    def initialize(self) -> EngineState:
        budget = self._remaining_evaluations(None)
        payload, evals, mean, fit = self._start_run_payload(0, None, max_evaluations=budget)
        return EngineState(
            step=0,
            evaluations=evals,
            best_position=mean.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload=payload,
        )

    def _restart(self, state: EngineState, seed_position: np.ndarray | None = None) -> None:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return
        restart_index = int(state.payload.get("restarts", 0)) + 1
        payload, evals, pos, fit = self._start_run_payload(
            restart_index,
            state.payload,
            max_evaluations=remaining,
            seed_position=seed_position,
        )
        if evals <= 0:
            return
        state.payload = payload
        state.evaluations += int(evals)
        if self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)

    def step(self, state: EngineState) -> EngineState:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state

        p = state.payload
        dim = self.problem.dimension
        lam = int(p["lambda"])
        lam_eval = lam if remaining is None else min(lam, int(remaining))
        if lam_eval <= 0:
            return state

        mean = np.asarray(p["mean"], dtype=float)
        sigma = float(p["sigma"])
        C = np.asarray(p["C"], dtype=float)
        pc = np.asarray(p["pc"], dtype=float)
        ps = np.asarray(p["ps"], dtype=float)
        weights = np.asarray(p["weights"], dtype=float)
        mu = int(p["mu"])
        mueff = float(p["mueff"])
        cc, cs, c1, cmu = float(p["cc"]), float(p["cs"]), float(p["c1"]), float(p["cmu"])
        damps, chi_n = float(p["damps"]), float(p["chi_n"])

        eigvals, B = np.linalg.eigh(0.5 * (C + C.T))
        eigvals = np.maximum(eigvals, 1.0e-30)
        sqrt_eig = np.sqrt(eigvals)
        inv_sqrt_eig = 1.0 / sqrt_eig
        z = self._rng.standard_normal((lam_eval, dim))
        y = z @ (B * sqrt_eig).T
        x = self._clip(mean + sigma * y)
        fit = np.asarray([float(self.problem.evaluate(row)) for row in x], dtype=float)

        pop = np.hstack((x, fit[:, None]))
        best_idx = self._best_index(fit)
        generation_best_fit = float(fit[best_idx])
        generation_best_pos = x[best_idx].copy()
        improved = self._substantial_improvement(generation_best_fit, state.best_fitness)
        if self._is_better(generation_best_fit, state.best_fitness):
            state.best_fitness = generation_best_fit
            state.best_position = generation_best_pos.tolist()

        # If the remaining budget is too small for a full selection update,
        # use the evaluated candidates only to refresh the incumbent.  This
        # keeps the evaluation budget strict without applying a distorted
        # covariance update from an undersized partial generation.
        if lam_eval < mu:
            p.update(
                population=pop,
                run_step=int(p["run_step"]) + 1,
                run_evaluations=int(p["run_evaluations"]) + lam_eval,
                stagnation=0 if improved else int(p.get("stagnation", 0)) + 1,
            )
            state.payload = p
            state.step += 1
            state.evaluations += int(lam_eval)
            return state

        order = self._order(fit)
        x_sel = x[order[:mu]]
        y_sel = (x_sel - mean) / max(sigma, 1.0e-30)
        old_mean = mean.copy()
        y_w = np.sum(weights[:, None] * y_sel, axis=0)
        mean = self._clip(old_mean + sigma * y_w)

        inv_sqrt_C_yw = (B * inv_sqrt_eig) @ (B.T @ y_w)
        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_C_yw
        norm_ps = float(np.linalg.norm(ps))
        hsig = (
            norm_ps
            / math.sqrt(max(1.0e-30, 1.0 - (1.0 - cs) ** (2.0 * (int(p["run_step"]) + 1))))
            / chi_n
            < (1.4 + 2.0 / (dim + 1.0))
        )
        pc = (1.0 - cc) * pc + (1.0 if hsig else 0.0) * math.sqrt(cc * (2.0 - cc) * mueff) * y_w
        rank_mu = np.zeros_like(C)
        for wi, yi in zip(weights, y_sel):
            rank_mu += wi * np.outer(yi, yi)
        C = (1.0 - c1 - cmu + (0.0 if hsig else c1 * cc * (2.0 - cc))) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
        C = 0.5 * (C + C.T)
        sigma = sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1.0))
        sigma = float(np.clip(sigma, 1.0e-12, 10.0 * float(np.max(self._span))))

        p.update(
            mean=mean,
            sigma=sigma,
            C=C,
            pc=pc,
            ps=ps,
            population=pop,
            run_step=int(p["run_step"]) + 1,
            run_evaluations=int(p["run_evaluations"]) + lam_eval,
            stagnation=0 if improved else int(p.get("stagnation", 0)) + 1,
        )
        state.payload = p
        state.step += 1
        state.evaluations += int(lam_eval)

        should_restart = (
            int(state.payload.get("stagnation", 0)) >= self._stagnation_steps
            or int(state.payload.get("run_step", 0)) >= self._restart_after_steps
        )
        if should_restart and self._remaining_evaluations(state) not in {0}:
            self._restart(state)
        return state

    def observe(self, state: EngineState) -> dict:
        pop = np.asarray(state.payload.get("population"), dtype=float)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "population_size": int(state.payload.get("lambda", pop.shape[0] if pop.ndim == 2 else 0)),
            "evaluated_population_size": int(pop.shape[0] if pop.ndim == 2 else 0),
            "sigma": float(state.payload.get("sigma", 0.0)),
            "restarts": int(state.payload.get("restarts", 0)),
            "injections": int(state.payload.get("injections", 0)),
            "run_step": int(state.payload.get("run_step", 0)),
            "stagnation": int(state.payload.get("stagnation", 0)),
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
        )

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        """Inject candidates by replacing weak population members and shifting the mean."""
        if not candidates:
            return state
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state

        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
        budget = len(ranked) if remaining is None else min(len(ranked), int(remaining))
        if budget <= 0:
            return state

        incoming = []
        for cand in ranked[:budget]:
            pos = self._clip(np.asarray(cand.position, dtype=float))
            fit = float(self.problem.evaluate(pos))
            incoming.append((pos, fit))
        if not incoming:
            return state

        state.evaluations += len(incoming)
        pop = np.asarray(state.payload.get("population"), dtype=float)
        if pop.ndim != 2 or pop.shape[1] < self.problem.dimension + 1:
            pop = np.empty((0, self.problem.dimension + 1), dtype=float)
        incoming_pop = np.asarray([np.r_[pos, fit] for pos, fit in incoming], dtype=float)
        combined = np.vstack((pop, incoming_pop)) if pop.size else incoming_pop
        order = self._order(combined[:, -1])
        keep = max(1, min(int(state.payload.get("lambda", combined.shape[0])), combined.shape[0]))
        pop = combined[order[:keep]].copy()
        state.payload["population"] = pop

        best_row = pop[0].copy()
        if self._is_better(float(best_row[-1]), state.best_fitness):
            state.best_position = best_row[:-1].tolist()
            state.best_fitness = float(best_row[-1])
            state.payload["stagnation"] = 0

        old_mean = np.asarray(state.payload.get("mean", best_row[:-1]), dtype=float)
        state.payload["mean"] = self._clip(0.5 * old_mean + 0.5 * best_row[:-1])
        state.payload["injections"] = int(state.payload.get("injections", 0)) + len(incoming)
        return state

    def restart(
        self,
        state: EngineState,
        seeds: list[CandidateRecord] | None = None,
        preserve_best: bool = True,
    ) -> EngineState:
        if seeds:
            ranked = sorted(seeds, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
            self._restart(state, seed_position=np.asarray(ranked[0].position, dtype=float))
        else:
            self._restart(state)
        if not preserve_best:
            pop = np.asarray(state.payload.get("population"), dtype=float)
            if pop.ndim == 2 and pop.shape[0] > 0:
                idx = self._best_index(pop[:, -1])
                state.best_position = pop[idx, :-1].tolist()
                state.best_fitness = float(pop[idx, -1])
        return state

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
                "restarts": int(state.payload.get("restarts", 0)),
                "injections": int(state.payload.get("injections", 0)),
                "population_size": int(state.payload.get("lambda", 0)),
                "elapsed_time": state.elapsed_time,
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload.get("population"), dtype=float)
        if pop.ndim != 2 or pop.shape[0] == 0:
            return []
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
