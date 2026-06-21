"""pyMetaheuristic src — IPOP-CMA-ES Engine.

Native implementation of Auger and Hansen's IPOP-CMA-ES restart strategy:
a weighted-recombination CMA-ES core with independent restarts and population
size doubled after every restart.
"""
from __future__ import annotations

import math
from dataclasses import replace
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


class IPOPCMAESEngine(BaseEngine):
    """Restart CMA-ES with increasing population size (IPOP-CMA-ES)."""

    algorithm_id = "ipop_cmaes"
    algorithm_name = "IPOP-CMA-ES"
    family = "evolutionary"
    _EVOMAPX_DIRECT_OPERATORS = (
        "ipop_cmaes.cmaes_sampling",
        "ipop_cmaes.elite_recombination",
        "ipop_cmaes.candidate_injection",
    )
    _EVOMAPX_DIAGNOSTIC_OPERATORS = (
        "ipop_cmaes.initialization",
        "ipop_cmaes.distribution_update",
        "ipop_cmaes.step_size_adaptation",
        "ipop_cmaes.population_restart",
        "ipop_cmaes.boundary_penalty",
    )
    _EVOMAPX_OPERATORS = _EVOMAPX_DIRECT_OPERATORS + _EVOMAPX_DIAGNOSTIC_OPERATORS
    _REFERENCE = {
        "doi": "10.1109/CEC.2005.1554902",
        "title": "A restart CMA evolution strategy with increasing population size",
        "authors": "Anne Auger, Nikolaus Hansen",
        "year": 2005,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS: dict[str, Any] = {
        # Paper-native IPOP rule: start from the CMA-ES default population
        # size, then multiply it by two at each independent restart.
        "population_size": None,
        "population_multiplier": 2.0,
        # Practical package default: preserve IPOP doubling, but cap restart
        # population growth at 4x the initial lambda unless the user disables
        # the cap with max_population_multiplier=None.
        "max_population_multiplier": 4.0,
        # Paper setting for [A, B]^n: sigma0 = (B - A) / 2.  For heterogeneous
        # boxes we use mean side length / 2 unless the user gives sigma0.
        "sigma0": None,
        # Default CMA-ES stopping criteria used before triggering an IPOP restart.
        "tolfun": 1.0e-12,
        "tolx_factor": 1.0e-12,
        "conditioncov_threshold": 1.0e14,
        # Bounded black-box wrapper.  The paper used the standard CMA-ES
        # boundary penalty.  This implementation keeps the same idea with a
        # quadratic exterior penalty evaluated at the clipped phenotype.
        "boundary_penalty": 1.0e6,
        # Numerical safety.
        "min_sigma": 1.0e-300,
        "max_sigma_multiplier": 10.0,
        # Web-app and simple API runs often set max_steps but not
        # max_evaluations.  IPOP doubles lambda after restarts, so a
        # step-only budget can otherwise grow into very large generations.
        # When enabled, a step-only run is interpreted as a fixed-budget run
        # with max_steps * initial_population_size evaluations.
        "step_budget_evaluation_cap": True,
    }

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **(config.params or {})}
        self._rng = np.random.default_rng(config.seed)
        self._lo = np.asarray(problem.min_values, dtype=float)
        self._hi = np.asarray(problem.max_values, dtype=float)
        self._span = np.where(self._hi - self._lo == 0.0, 1.0, self._hi - self._lo)
        self._dim = int(problem.dimension)
        user_lambda = self._params.get("population_size", None)
        self._base_lambda = (
            self._default_lambda(self._dim) if user_lambda is None else max(4, int(user_lambda))
        )
        self._population_multiplier = max(1.01, float(self._params.get("population_multiplier", 2.0)))
        self._tolfun = max(0.0, float(self._params.get("tolfun", 1.0e-12)))
        self._tolx_factor = max(0.0, float(self._params.get("tolx_factor", 1.0e-12)))
        self._conditioncov_threshold = max(1.0, float(self._params.get("conditioncov_threshold", 1.0e14)))
        self._boundary_penalty = max(0.0, float(self._params.get("boundary_penalty", 1.0e6)))
        self._min_sigma = max(1.0e-300, float(self._params.get("min_sigma", 1.0e-300)))
        self._max_sigma = max(self._min_sigma, float(self._params.get("max_sigma_multiplier", 10.0)) * float(np.max(self._span)))
        self._step_budget_evaluation_cap = bool(self._params.get("step_budget_evaluation_cap", True))
        self._effective_max_evaluations = self._resolve_effective_max_evaluations()

    def _resolve_effective_max_evaluations(self) -> int | None:
        """Return the hard evaluation cap used internally by IPOP-CMA-ES.

        The paper and most CMA-ES comparisons are fixed-budget in objective
        evaluations.  pyMetaheuristic's web app, however, commonly supplies
        only max_steps.  Since IPOP-CMA-ES changes lambda across restarts,
        max_steps alone is not a stable cost unit.  The inferred cap keeps
        step-only runs responsive without affecting explicit max_evaluations
        runs.
        """
        if self.config.max_evaluations is not None:
            return max(0, int(self.config.max_evaluations))
        if self._step_budget_evaluation_cap and self.config.max_steps is not None:
            return max(1, int(self.config.max_steps) * int(self._base_lambda))
        return None

    def _mark_inferred_budget_stop(self, state: EngineState) -> None:
        if self.config.max_evaluations is None and self._effective_max_evaluations is not None:
            state.termination_reason = "max_evaluations_inferred_from_steps"
            state.terminated = True

    def _zero_operator_float_map(self) -> dict[str, float]:
        return {label: 0.0 for label in self._EVOMAPX_OPERATORS}

    def _zero_operator_count_map(self) -> dict[str, int]:
        return {label: 0 for label in self._EVOMAPX_OPERATORS}

    @staticmethod
    def _default_lambda(dim: int) -> int:
        return max(4, 4 + int(math.floor(3.0 * math.log(max(2, dim)))))

    def _remaining_evaluations(self, state: EngineState | None = None, used: int = 0) -> int | None:
        if self._effective_max_evaluations is None:
            return None
        done = int(getattr(state, "evaluations", 0) if state is not None else 0)
        return max(0, int(self._effective_max_evaluations) - done - int(used))

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _order(self, values: np.ndarray) -> np.ndarray:
        idx = np.argsort(values)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return self.problem.clip_position(np.asarray(x, dtype=float))

    def _initial_sigma(self) -> float:
        sigma0 = self._params.get("sigma0", None)
        if sigma0 is not None:
            return max(self._min_sigma, float(sigma0))
        return max(self._min_sigma, 0.5 * float(np.mean(self._span)))

    def _constants(self, lam: int) -> dict[str, Any]:
        dim = self._dim
        mu = max(1, lam // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights / np.sum(weights)
        mueff = float(1.0 / np.sum(weights**2))
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        return {
            "mu": mu,
            "weights": weights,
            "mueff": mueff,
            "cc": cc,
            "cs": cs,
            "c1": c1,
            "cmu": cmu,
            "damps": damps,
            "chi_n": chi_n,
        }

    def _lambda_for_restart(self, restart_index: int) -> int:
        uncapped = max(
            4,
            int(round(self._base_lambda * (self._population_multiplier ** max(0, int(restart_index))))),
        )
        max_population_multiplier = self._params.get("max_population_multiplier", 4.0)
        if max_population_multiplier is None:
            return uncapped
        cap = max(4, int(round(self._base_lambda * float(max_population_multiplier))))
        return min(uncapped, cap)

    def _evaluate_candidate(self, raw: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Evaluate candidate with standard CMA-style exterior boundary penalty."""
        raw = np.asarray(raw, dtype=float)
        phenotype = self._clip(raw)
        raw_fitness = float(self.problem.evaluate(phenotype.tolist()))
        if np.all(np.isfinite(raw)):
            normalized_violation = (raw - phenotype) / self._span
            penalty = self._boundary_penalty * float(np.sum(normalized_violation**2)) * (1.0 + abs(raw_fitness))
        else:
            penalty = float("inf")
        if self.problem.objective == "min":
            fitness = raw_fitness + penalty
        else:
            fitness = raw_fitness - penalty
        return phenotype, float(fitness), float(penalty)

    def _start_run(self, restart_index: int, carried: dict[str, Any] | None = None, seed: np.ndarray | None = None) -> tuple[dict[str, Any], int]:
        dim = self._dim
        lam = self._lambda_for_restart(restart_index)
        const = self._constants(lam)
        mean = self._clip(seed) if seed is not None else self._rng.uniform(self._lo, self._hi, dim)
        sigma0 = self._initial_sigma()
        phenotype, fit, penalty = self._evaluate_candidate(mean)
        evals = 1
        old_counts = dict((carried or {}).get("operator_counts", {}))
        old_contrib = dict((carried or {}).get("operator_contributions", {}))
        operator_counts = self._zero_operator_count_map()
        operator_contributions = self._zero_operator_float_map()
        for label in self._EVOMAPX_OPERATORS:
            operator_counts[label] = int(old_counts.get(label, 0))
            operator_contributions[label] = float(old_contrib.get(label, 0.0))
        operator_counts["ipop_cmaes.initialization"] += 1
        operator_counts["ipop_cmaes.boundary_penalty"] += int(penalty > 0.0)
        last_operator_counts = self._zero_operator_count_map()
        last_operator_contributions = self._zero_operator_float_map()
        last_operator_counts["ipop_cmaes.initialization"] = 1
        last_operator_counts["ipop_cmaes.boundary_penalty"] = int(penalty > 0.0)
        payload = {
            "mean": phenotype.copy(),
            "sigma": float(sigma0),
            "sigma0": float(sigma0),
            "C": np.eye(dim),
            "pc": np.zeros(dim),
            "ps": np.zeros(dim),
            "lambda": int(lam),
            "mu": int(const["mu"]),
            "weights": const["weights"],
            "mueff": float(const["mueff"]),
            "cc": float(const["cc"]),
            "cs": float(const["cs"]),
            "c1": float(const["c1"]),
            "cmu": float(const["cmu"]),
            "damps": float(const["damps"]),
            "chi_n": float(const["chi_n"]),
            "run_step": 0,
            "run_evaluations": int(evals),
            "restarts": int(restart_index),
            "function_history": [float(fit)],
            "best_history": [float(fit)],
            "population": np.asarray([[*phenotype.tolist(), float(fit)]], dtype=float),
            "penalty_last": float(penalty),
            "restart_reason": "initialization" if restart_index == 0 else "independent_restart",
            "operator_counts": operator_counts,
            "operator_contributions": operator_contributions,
            "last_operator_counts": last_operator_counts,
            "last_operator_contributions": last_operator_contributions,
            "injections": int((carried or {}).get("injections", 0)),
        }
        return payload, evals

    def initialize(self) -> EngineState:
        payload, evals = self._start_run(0)
        mean = np.asarray(payload["mean"], dtype=float)
        fit = float(payload["population"][0, -1])
        return EngineState(
            step=0,
            evaluations=int(evals),
            best_position=mean.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload=payload,
        )

    def _generation_stop_reason(self, payload: dict[str, Any]) -> str | None:
        dim = self._dim
        lam = int(payload["lambda"])
        sigma = float(payload["sigma"])
        sigma0 = float(payload.get("sigma0", self._initial_sigma()))
        C = np.asarray(payload["C"], dtype=float)
        pc = np.asarray(payload["pc"], dtype=float)
        mean = np.asarray(payload["mean"], dtype=float)
        funhist = [float(v) for v in payload.get("function_history", [])]
        besthist = [float(v) for v in payload.get("best_history", [])]

        hist_len = 10 + int(math.ceil(30.0 * dim / max(1, lam)))
        if len(besthist) >= hist_len:
            recent_best = np.asarray(besthist[-hist_len:], dtype=float)
            if float(np.max(recent_best) - np.min(recent_best)) == 0.0:
                return "equalfunvalhist"
            recent_all = np.asarray(funhist[-hist_len * max(1, lam):] + besthist[-hist_len:], dtype=float)
            if recent_all.size and float(np.max(recent_all) - np.min(recent_all)) < self._tolfun:
                return "tolfun"

        tolx = self._tolx_factor * sigma0
        std = sigma * np.sqrt(np.maximum(np.diag(C), 0.0))
        if np.all(std < tolx) and np.all(np.abs(sigma * pc) < tolx):
            return "tolx"

        eigvals, B = np.linalg.eigh(0.5 * (C + C.T))
        eigvals = np.maximum(eigvals, 0.0)
        axis_index = int(payload.get("run_step", 0)) % dim
        axis_step = 0.1 * sigma * math.sqrt(float(eigvals[axis_index])) * B[:, axis_index]
        if np.array_equal(mean, mean + axis_step):
            return "noeffectaxis"

        coord_step = 0.2 * sigma * np.sqrt(np.maximum(np.diag(C), 0.0))
        if np.array_equal(mean, mean + coord_step):
            return "noeffectcoord"

        positive = eigvals[eigvals > 0.0]
        if positive.size == 0 or float(np.max(positive) / max(np.min(positive), 1.0e-300)) > self._conditioncov_threshold:
            return "conditioncov"
        return None

    def _restart(self, state: EngineState, reason: str, seed: np.ndarray | None = None) -> None:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return
        restart_index = int(state.payload.get("restarts", 0)) + 1
        old_counts = dict(state.payload.get("operator_counts", {}))
        old_contrib = dict(state.payload.get("operator_contributions", {}))
        carried = {"operator_counts": old_counts, "operator_contributions": old_contrib, "injections": int(state.payload.get("injections", 0))}
        payload, evals = self._start_run(restart_index, carried=carried, seed=seed)
        payload["operator_counts"]["ipop_cmaes.population_restart"] = old_counts.get("ipop_cmaes.population_restart", 0) + 1
        payload["operator_contributions"]["ipop_cmaes.population_restart"] = old_contrib.get("ipop_cmaes.population_restart", 0.0)
        payload["last_operator_counts"] = self._zero_operator_count_map()
        payload["last_operator_contributions"] = self._zero_operator_float_map()
        payload["last_operator_counts"]["ipop_cmaes.initialization"] = 1
        payload["last_operator_counts"]["ipop_cmaes.population_restart"] = 1
        payload["restart_reason"] = str(reason)
        state.payload = payload
        state.evaluations += int(evals)
        fit = float(payload["population"][0, -1])
        if self._is_better(fit, state.best_fitness):
            state.best_position = np.asarray(payload["mean"], dtype=float).tolist()
            state.best_fitness = fit

    def step(self, state: EngineState) -> EngineState:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            self._mark_inferred_budget_stop(state)
            return state

        p = state.payload
        dim = self._dim
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
        previous_best = float(state.best_fitness)

        eigvals, B = np.linalg.eigh(0.5 * (C + C.T))
        eigvals = np.maximum(eigvals, 1.0e-300)
        sqrt_eig = np.sqrt(eigvals)
        inv_sqrt_eig = 1.0 / sqrt_eig

        z = self._rng.standard_normal((lam_eval, dim))
        y = z @ (B * sqrt_eig).T
        raw_x = mean + sigma * y
        evaluated = [self._evaluate_candidate(row) for row in raw_x]
        x = np.asarray([row[0] for row in evaluated], dtype=float)
        fit = np.asarray([row[1] for row in evaluated], dtype=float)
        penalties = np.asarray([row[2] for row in evaluated], dtype=float)
        p["function_history"].extend(float(v) for v in fit.tolist())

        pop = np.hstack((x, fit[:, None]))
        order = self._order(fit)
        best_idx = int(order[0])
        gen_best_fit = float(fit[best_idx])
        gen_best_pos = x[best_idx].copy()
        if self._is_better(gen_best_fit, state.best_fitness):
            state.best_position = gen_best_pos.tolist()
            state.best_fitness = gen_best_fit

        # Respect hard budgets: if a partial generation is too small to update
        # the distribution, still use the evaluated points to refresh the best.
        if lam_eval < mu:
            last_counts = self._zero_operator_count_map()
            last_contrib = self._zero_operator_float_map()
            last_counts["ipop_cmaes.cmaes_sampling"] = 1
            last_counts["ipop_cmaes.boundary_penalty"] = int(np.count_nonzero(penalties > 0.0))
            if self.problem.objective == "min":
                partial_improvement = max(0.0, previous_best - float(state.best_fitness))
            else:
                partial_improvement = max(0.0, float(state.best_fitness) - previous_best)
            last_contrib["ipop_cmaes.cmaes_sampling"] = float(partial_improvement)
            counts = p.setdefault("operator_counts", self._zero_operator_count_map())
            contrib = p.setdefault("operator_contributions", self._zero_operator_float_map())
            counts["ipop_cmaes.cmaes_sampling"] = int(counts.get("ipop_cmaes.cmaes_sampling", 0)) + 1
            counts["ipop_cmaes.boundary_penalty"] = int(counts.get("ipop_cmaes.boundary_penalty", 0)) + int(np.count_nonzero(penalties > 0.0))
            contrib["ipop_cmaes.cmaes_sampling"] = float(contrib.get("ipop_cmaes.cmaes_sampling", 0.0)) + float(partial_improvement)
            p.update(
                population=pop,
                run_step=int(p["run_step"]) + 1,
                run_evaluations=int(p["run_evaluations"]) + int(lam_eval),
                penalty_last=float(np.mean(penalties)) if penalties.size else 0.0,
                last_operator_counts=last_counts,
                last_operator_contributions=last_contrib,
            )
            p["best_history"].append(float(state.best_fitness))
            state.payload = p
            state.step += 1
            state.evaluations += int(lam_eval)
            if self._remaining_evaluations(state) == 0:
                self._mark_inferred_budget_stop(state)
            return state

        x_sel = x[order[:mu]]
        y_sel = (x_sel - mean) / max(sigma, self._min_sigma)
        old_mean = mean.copy()
        y_w = np.sum(weights[:, None] * y_sel, axis=0)
        mean = self._clip(old_mean + sigma * y_w)

        inv_sqrt_C_yw = (B * inv_sqrt_eig) @ (B.T @ y_w)
        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_C_yw
        norm_ps = float(np.linalg.norm(ps))
        hsig_den = math.sqrt(max(1.0e-300, 1.0 - (1.0 - cs) ** (2.0 * (int(p["run_step"]) + 1)))) * chi_n
        hsig = (norm_ps / hsig_den) < (1.4 + 2.0 / (dim + 1.0))
        pc = (1.0 - cc) * pc + (1.0 if hsig else 0.0) * math.sqrt(cc * (2.0 - cc) * mueff) * y_w
        rank_mu = np.zeros_like(C)
        for wi, yi in zip(weights, y_sel):
            rank_mu += float(wi) * np.outer(yi, yi)
        C = (1.0 - c1 - cmu + (0.0 if hsig else c1 * cc * (2.0 - cc))) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
        C = 0.5 * (C + C.T)
        sigma = sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1.0))
        sigma = float(np.clip(sigma, self._min_sigma, self._max_sigma))

        if self.problem.objective == "min":
            improvement = max(0.0, previous_best - float(state.best_fitness))
        else:
            improvement = max(0.0, float(state.best_fitness) - previous_best)

        last_counts = self._zero_operator_count_map()
        last_contrib = self._zero_operator_float_map()
        last_counts["ipop_cmaes.cmaes_sampling"] = 1
        last_counts["ipop_cmaes.elite_recombination"] = 1
        last_counts["ipop_cmaes.distribution_update"] = 1
        last_counts["ipop_cmaes.step_size_adaptation"] = 1
        last_counts["ipop_cmaes.boundary_penalty"] = int(np.count_nonzero(penalties > 0.0))
        # The immediate best-so-far improvement is generated by the sampled
        # offspring and the elite/parent recombination-selection step.  The
        # covariance and step-size updates are diagnostic model updates, so they
        # receive activity counts but no direct objective-improvement credit.
        last_contrib["ipop_cmaes.cmaes_sampling"] = 0.5 * float(improvement)
        last_contrib["ipop_cmaes.elite_recombination"] = 0.5 * float(improvement)

        p.update(
            mean=mean,
            sigma=sigma,
            C=C,
            pc=pc,
            ps=ps,
            population=pop,
            run_step=int(p["run_step"]) + 1,
            run_evaluations=int(p["run_evaluations"]) + int(lam_eval),
            penalty_last=float(np.mean(penalties)) if penalties.size else 0.0,
            last_operator_counts=last_counts,
            last_operator_contributions=last_contrib,
        )
        p["best_history"].append(float(state.best_fitness))
        counts = p.setdefault("operator_counts", self._zero_operator_count_map())
        contrib = p.setdefault("operator_contributions", self._zero_operator_float_map())
        for key, value in last_counts.items():
            counts[key] = int(counts.get(key, 0)) + int(value)
        for key, value in last_contrib.items():
            contrib[key] = float(contrib.get(key, 0.0)) + float(value)

        state.payload = p
        state.step += 1
        state.evaluations += int(lam_eval)
        if self._remaining_evaluations(state) == 0:
            self._mark_inferred_budget_stop(state)

        reason = self._generation_stop_reason(state.payload)
        if reason is not None and self._remaining_evaluations(state) not in {0}:
            self._restart(state, reason=reason)
        return state

    def observe(self, state: EngineState) -> dict:
        p = state.payload
        pop = np.asarray(p.get("population", []), dtype=float)
        C = np.asarray(p.get("C", np.eye(self._dim)), dtype=float)
        eigvals = np.linalg.eigvalsh(0.5 * (C + C.T)) if C.size else np.ones(self._dim)
        positive = eigvals[eigvals > 0.0]
        cond = float(np.max(positive) / max(np.min(positive), 1.0e-300)) if positive.size else float("inf")
        obs = {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "population_size": int(p.get("lambda", 0)),
            "evaluated_population_size": int(pop.shape[0] if pop.ndim == 2 else 0),
            "mu": int(p.get("mu", 0)),
            "sigma": float(p.get("sigma", 0.0)),
            "sigma0": float(p.get("sigma0", 0.0)),
            "condition_number": cond,
            "restarts": int(p.get("restarts", 0)),
            "restart_reason": str(p.get("restart_reason", "")),
            "run_step": int(p.get("run_step", 0)),
            "run_evaluations": int(p.get("run_evaluations", 0)),
            "injections": int(p.get("injections", 0)),
            "operator_counts": dict(p.get("last_operator_counts", p.get("operator_counts", {}))),
            "operator_contributions": dict(p.get("last_operator_contributions", p.get("operator_contributions", {}))),
            "operator_counts_total": dict(p.get("operator_counts", {})),
            "operator_contributions_total": dict(p.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }
        return obs

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload.get("population", []), dtype=float)
        if pop.ndim != 2 or pop.shape[1] < self._dim + 1:
            return []
        return [
            CandidateRecord(
                position=row[: self._dim].tolist(),
                fitness=float(row[self._dim]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="current",
            )
            for row in pop
        ]

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=int(state.step),
            role="best",
        )

    def export_candidates(self, state: EngineState, k: int = 1, mode: str = "best") -> list[CandidateRecord]:
        pop = np.asarray(state.payload.get("population", []), dtype=float)
        if pop.ndim != 2 or pop.shape[1] < self._dim + 1:
            return [self.get_best_candidate(state)]
        order = self._order(pop[:, -1])[: max(1, min(int(k), pop.shape[0]))]
        return [
            CandidateRecord(
                position=pop[idx, : self._dim].tolist(),
                fitness=float(pop[idx, -1]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="elite" if rank else "best",
            )
            for rank, idx in enumerate(order)
        ]

    def inject_candidates(self, state: EngineState, candidates: list[CandidateRecord], policy: str = "native") -> EngineState:
        if not candidates:
            return state
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state
        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
        budget = len(ranked) if remaining is None else min(len(ranked), int(remaining))
        if budget <= 0:
            return state
        incoming_rows = []
        best_incoming_pos = None
        best_incoming_fit = None
        for cand in ranked[:budget]:
            pos = self._clip(np.asarray(cand.position, dtype=float))
            phenotype, fit, _ = self._evaluate_candidate(pos)
            incoming_rows.append(np.r_[phenotype, fit])
            if best_incoming_fit is None or self._is_better(fit, best_incoming_fit):
                best_incoming_fit = float(fit)
                best_incoming_pos = phenotype.copy()
        state.evaluations += int(len(incoming_rows))
        state.payload["injections"] = int(state.payload.get("injections", 0)) + int(len(incoming_rows))
        counts = state.payload.setdefault("operator_counts", self._zero_operator_count_map())
        counts["ipop_cmaes.candidate_injection"] = int(counts.get("ipop_cmaes.candidate_injection", 0)) + int(len(incoming_rows))
        contrib = state.payload.setdefault("operator_contributions", self._zero_operator_float_map())
        last_counts = self._zero_operator_count_map()
        last_contrib = self._zero_operator_float_map()
        last_counts["ipop_cmaes.candidate_injection"] = int(len(incoming_rows))
        if best_incoming_pos is not None and best_incoming_fit is not None:
            old_best = float(state.best_fitness)
            if self._is_better(best_incoming_fit, state.best_fitness):
                state.best_position = best_incoming_pos.tolist()
                state.best_fitness = float(best_incoming_fit)
                if self.problem.objective == "min":
                    delta = max(0.0, old_best - best_incoming_fit)
                else:
                    delta = max(0.0, best_incoming_fit - old_best)
                last_contrib["ipop_cmaes.candidate_injection"] = float(delta)
                contrib["ipop_cmaes.candidate_injection"] = float(contrib.get("ipop_cmaes.candidate_injection", 0.0)) + delta
            old_mean = np.asarray(state.payload.get("mean", best_incoming_pos), dtype=float)
            state.payload["mean"] = self._clip(0.5 * old_mean + 0.5 * best_incoming_pos)
        pop = np.asarray(state.payload.get("population", []), dtype=float)
        incoming_pop = np.asarray(incoming_rows, dtype=float)
        if pop.ndim == 2 and pop.shape[1] == self._dim + 1:
            combined = np.vstack((pop, incoming_pop))
        else:
            combined = incoming_pop
        keep = max(1, min(int(state.payload.get("lambda", combined.shape[0])), combined.shape[0]))
        state.payload["population"] = combined[self._order(combined[:, -1])[:keep]].copy()
        state.payload["last_operator_counts"] = last_counts
        state.payload["last_operator_contributions"] = last_contrib
        return state

    def restart(self, state: EngineState, seeds: list[CandidateRecord] | None = None, preserve_best: bool = True) -> EngineState:
        seed = None
        if seeds:
            ranked = sorted(seeds, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
            seed = np.asarray(ranked[0].position, dtype=float)
        self._restart(state, reason="manual_restart", seed=seed)
        return state

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=replace(self.capabilities),
            metadata={
                "algorithm_name": self.algorithm_name,
                "restarts": int(state.payload.get("restarts", 0)),
                "population_size": int(state.payload.get("lambda", 0)),
                "sigma": float(state.payload.get("sigma", 0.0)),
                "injections": int(state.payload.get("injections", 0)),
                "elapsed_time": state.elapsed_time,
            },
        )
