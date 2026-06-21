"""pyMetaheuristic src — BIPOP-CMA-ES Engine.

Native BI-population CMA-ES following Hansen's BBOB-2009 implementation:
a default-population CMA-ES first run followed by two budget-controlled,
interlaced restart regimes, one with increasing large populations and one with
varying small populations.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from .protocol import (
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)
from ._restart_common import RestartCMAESBase


class BIPOPCMAESEngine(RestartCMAESBase):
    """BI-population CMA-ES with paper-style interlaced restart regimes."""

    algorithm_id = "bipop_cmaes"
    algorithm_name = "BIPOP-CMA-ES"
    family = "evolutionary"
    _EVOMAPX_DIRECT_OPERATORS = (
        "bipop_cmaes.cmaes_sampling",
        "bipop_cmaes.elite_recombination",
        "bipop_cmaes.candidate_injection",
    )
    _EVOMAPX_DIAGNOSTIC_OPERATORS = (
        "bipop_cmaes.initialization",
        "bipop_cmaes.large_population_restart",
        "bipop_cmaes.small_population_restart",
        "bipop_cmaes.budget_regime_selection",
        "bipop_cmaes.distribution_update",
        "bipop_cmaes.step_size_adaptation",
        "bipop_cmaes.termination_check",
        "bipop_cmaes.boundary_repair",
    )
    _EVOMAPX_OPERATORS = _EVOMAPX_DIRECT_OPERATORS + _EVOMAPX_DIAGNOSTIC_OPERATORS
    _REFERENCE = {
        "doi": "10.1145/1570256.1570333",
        "title": "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed",
        "authors": "Nikolaus Hansen",
        "year": 2009,
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
        **RestartCMAESBase._DEFAULTS,
        # Paper default: lambda_def = 4 + floor(3 ln D).  ``None``/``auto``
        # activates this rule; an integer preserves the package override path.
        "population_size": None,
        "population_multiplier": 2.0,
        # Paper limit: at most nine increasing-population restarts, largest
        # lambda = 2^9 * lambda_def = 512 * lambda_def.
        "max_large_restarts": 9,
        "max_population_multiplier": 512.0,
        # Initial large-regime sigma is 1/5 of the domain width in the paper
        # (sigma0=2 on the BBOB [-5,5]^D box).  The generic bounded version
        # uses the mean side length.
        "large_sigma_span_fraction": 0.2,
        # Small-regime sigma0 = large_sigma0 * 10^(-2 U[0,1]).
        "small_sigma_log10_range": 2.0,
        # The BBOB implementation samples m0 uniformly from [-4,4]^D inside
        # a [-5,5]^D domain.  This is the central 80% of each bounded range.
        "initial_mean_box_fraction": 0.8,
        # Paper single-run termination criteria.
        "tolfun": 1.0e-12,
        "tolx": 1.0e-12,
        "tolupsigma": 1.0e20,
        "conditioncov_threshold": 1.0e14,
        "min_sigma": 1.0e-300,
        "max_sigma_multiplier": 10.0,
    }

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        params = {**self._DEFAULTS, **(config.params or {})}
        pop_param = params.get("population_size", None)
        if pop_param in (None, 0, "paper", "auto", "default"):
            params["population_size"] = self._default_lambda(int(problem.dimension))
        else:
            params["population_size"] = max(4, int(pop_param))
        super().__init__(problem, replace(config, params=params))
        # RestartCMAESBase has initialized the common CMA-ES state above.  The
        # assignments below normalize BIPOP-specific controls for faster access.
        self._params = params
        self._base_lambda = max(4, int(params["population_size"]))
        self._population_multiplier = max(1.01, float(params.get("population_multiplier", 2.0)))
        max_population_multiplier = params.get("max_population_multiplier", 512.0)
        self._max_population_multiplier = (
            None if max_population_multiplier is None else max(1.0, float(max_population_multiplier))
        )
        self._max_large_restarts = max(1, int(params.get("max_large_restarts", 9)))
        self._tolfun = max(0.0, float(params.get("tolfun", 1.0e-12)))
        self._tolx = max(0.0, float(params.get("tolx", 1.0e-12)))
        self._tolupsigma = max(1.0, float(params.get("tolupsigma", 1.0e20)))
        self._conditioncov_threshold = max(1.0, float(params.get("conditioncov_threshold", 1.0e14)))
        self._min_sigma = max(1.0e-300, float(params.get("min_sigma", 1.0e-300)))
        self._max_sigma = max(
            self._min_sigma,
            float(params.get("max_sigma_multiplier", 10.0)) * float(np.max(self._span)),
        )

    # ------------------------------------------------------------------
    # Utilities and BIPOP schedules
    # ------------------------------------------------------------------
    @staticmethod
    def _default_lambda(dim: int) -> int:
        return max(4, 4 + int(math.floor(3.0 * math.log(max(2, int(dim))))))

    def _zero_operator_float_map(self) -> dict[str, float]:
        return {label: 0.0 for label in self._EVOMAPX_OPERATORS}

    def _zero_operator_count_map(self) -> dict[str, int]:
        return {label: 0 for label in self._EVOMAPX_OPERATORS}

    def _large_sigma0(self) -> float:
        sigma0 = self._params.get("sigma0", None)
        if sigma0 is not None:
            return max(self._min_sigma, float(sigma0))
        fraction = max(0.0, float(self._params.get("large_sigma_span_fraction", 0.2)))
        return max(self._min_sigma, fraction * float(np.mean(self._span)))

    def _small_sigma0(self) -> float:
        user_sigma = self._params.get("small_sigma0", None)
        if user_sigma is not None:
            return max(self._min_sigma, float(user_sigma))
        exponent_range = max(0.0, float(self._params.get("small_sigma_log10_range", 2.0)))
        return max(self._min_sigma, self._large_sigma0() * (10.0 ** (-exponent_range * float(self._rng.random()))))

    def _initial_mean(self, seed_position: np.ndarray | None = None) -> np.ndarray:
        if seed_position is not None:
            return self._clip(seed_position)
        fraction = float(np.clip(self._params.get("initial_mean_box_fraction", 0.8), 0.0, 1.0))
        center = 0.5 * (self._lo + self._hi)
        half_width = 0.5 * fraction * self._span
        lo = center - half_width
        hi = center + half_width
        return self._rng.uniform(lo, hi, int(self.problem.dimension))

    def _budget_after_previous_run(self, payload: dict[str, Any] | None) -> tuple[float, float]:
        if not payload:
            return 0.0, 0.0
        large_budget = float(payload.get("bipop_large_budget", 0.0))
        small_budget = float(payload.get("bipop_small_budget", 0.0))
        previous_regime = str(payload.get("bipop_regime", "initial"))
        previous_evals = float(payload.get("run_evaluations", 0.0))
        # Hansen's paper disregards the first single default-population run when
        # loading the large-regime budget; only actual restart regimes accrue.
        if previous_regime == "large":
            large_budget += previous_evals
        elif previous_regime == "small":
            small_budget += previous_evals
        return large_budget, small_budget

    def _bipop_schedule(self, restart_index: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        restart_index = int(restart_index)
        large_budget, small_budget = self._budget_after_previous_run(payload)
        large_started = int((payload or {}).get("bipop_large_restarts_started", 0))
        small_started = int((payload or {}).get("bipop_small_restarts_started", 0))
        latest_large_lambda = int((payload or {}).get("bipop_latest_large_lambda", self._base_lambda))

        if restart_index <= 0:
            return {
                "regime": "initial",
                "lambda": int(self._base_lambda),
                "sigma0": float(self._large_sigma0()),
                "large_budget": large_budget,
                "small_budget": small_budget,
                "large_started": large_started,
                "small_started": small_started,
                "latest_large_lambda": latest_large_lambda,
                "selection_rule": "initial_default_population",
            }

        if large_started >= self._max_large_restarts:
            # A caller should normally stop before asking for another run, but
            # return the final large lambda defensively if a manual restart does.
            final_lambda = self._cap_lambda(int(round(self._base_lambda * (self._population_multiplier ** large_started))))
            return {
                "regime": "large_exhausted",
                "lambda": int(final_lambda),
                "sigma0": float(self._large_sigma0()),
                "large_budget": large_budget,
                "small_budget": small_budget,
                "large_started": large_started,
                "small_started": small_started,
                "latest_large_lambda": max(latest_large_lambda, final_lambda),
                "selection_rule": "max_large_restarts_reached",
            }

        # Paper rule: after the initial run, launch the first restart in the
        # increasing-large regime.  Thereafter, launch a small-population run if
        # and only if its accumulated budget is smaller than the large budget.
        choose_small = restart_index > 1 and small_budget < large_budget
        if choose_small:
            latest_large_lambda = max(int(latest_large_lambda), 2 * int(self._base_lambda))
            upper_ratio = max(1.0, 0.5 * float(latest_large_lambda) / float(self._base_lambda))
            u = float(self._rng.random())
            lam = int(math.floor(float(self._base_lambda) * (upper_ratio ** (u * u))))
            upper = max(self._base_lambda, latest_large_lambda // 2)
            lam = int(np.clip(lam, self._base_lambda, upper))
            return {
                "regime": "small",
                "lambda": max(4, lam),
                "sigma0": float(self._small_sigma0()),
                "large_budget": large_budget,
                "small_budget": small_budget,
                "large_started": large_started,
                "small_started": small_started + 1,
                "latest_large_lambda": latest_large_lambda,
                "selection_rule": "small_budget_below_large_budget",
            }

        large_number = large_started + 1
        large_uncapped = int(round(float(self._base_lambda) * (self._population_multiplier ** large_number)))
        large_lambda = self._cap_lambda(large_uncapped)
        return {
            "regime": "large",
            "lambda": int(large_lambda),
            "sigma0": float(self._large_sigma0()),
            "large_budget": large_budget,
            "small_budget": small_budget,
            "large_started": large_number,
            "small_started": small_started,
            "latest_large_lambda": int(large_lambda),
            "selection_rule": "large_budget_not_above_small_budget" if restart_index > 1 else "first_restart_large_regime",
        }

    def _next_lambda(self, restart_index: int, payload: dict | None = None) -> int:
        return int(self._bipop_schedule(restart_index, payload)["lambda"])

    def _start_run_payload(
        self,
        restart_index: int,
        payload: dict[str, Any] | None = None,
        max_evaluations: int | None = None,
        seed_position: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], int, np.ndarray, float]:
        dim = int(self.problem.dimension)
        schedule = self._bipop_schedule(restart_index, payload)
        lam = max(4, int(schedule["lambda"]))
        const = self._constants(dim, lam)
        mean = self._initial_mean(seed_position)
        sigma = max(self._min_sigma, float(schedule["sigma0"]))

        if max_evaluations is not None and int(max_evaluations) <= 0:
            fit = self.problem.worst_fitness()
            evals = 0
        else:
            fit = float(self.problem.evaluate(mean.tolist()))
            evals = 1

        old_counts = dict((payload or {}).get("operator_counts", {}))
        old_contrib = dict((payload or {}).get("operator_contributions", {}))
        operator_counts = self._zero_operator_count_map()
        operator_contributions = self._zero_operator_float_map()
        for label in self._EVOMAPX_OPERATORS:
            operator_counts[label] = int(old_counts.get(label, 0))
            operator_contributions[label] = float(old_contrib.get(label, 0.0))

        last_counts = self._zero_operator_count_map()
        last_contrib = self._zero_operator_float_map()
        operator_counts["bipop_cmaes.initialization"] += 1
        last_counts["bipop_cmaes.initialization"] = 1
        if int(restart_index) > 0:
            operator_counts["bipop_cmaes.budget_regime_selection"] += 1
            last_counts["bipop_cmaes.budget_regime_selection"] = 1
            if schedule["regime"] == "large":
                operator_counts["bipop_cmaes.large_population_restart"] += 1
                last_counts["bipop_cmaes.large_population_restart"] = 1
            elif schedule["regime"] == "small":
                operator_counts["bipop_cmaes.small_population_restart"] += 1
                last_counts["bipop_cmaes.small_population_restart"] = 1

        run_payload = {
            "mean": mean,
            "sigma": float(sigma),
            "sigma0": float(sigma),
            "C": np.eye(dim),
            "pc": np.zeros(dim),
            "ps": np.zeros(dim),
            "population": np.hstack((mean.reshape(1, -1), np.array([[fit]], dtype=float))),
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
            "injections": int((payload or {}).get("injections", 0)),
            "function_history": [float(fit)],
            "best_history": [float(fit)],
            "median_history": [float(fit)],
            "equalfun_history": [],
            "restart_reason": "initialization" if int(restart_index) == 0 else str((payload or {}).get("termination_reason_last", "restart")),
            "termination_reason_last": None,
            "operator_counts": operator_counts,
            "operator_contributions": operator_contributions,
            "last_operator_counts": last_counts,
            "last_operator_contributions": last_contrib,
            "bipop_regime": str(schedule["regime"]),
            "bipop_selection_rule": str(schedule["selection_rule"]),
            "bipop_large_budget": float(schedule["large_budget"]),
            "bipop_small_budget": float(schedule["small_budget"]),
            "bipop_large_restarts_started": int(schedule["large_started"]),
            "bipop_small_restarts_started": int(schedule["small_started"]),
            "bipop_latest_large_lambda": int(schedule["latest_large_lambda"]),
        }
        return run_payload, int(evals), mean, float(fit)

    # ------------------------------------------------------------------
    # Initialization, restart, and termination
    # ------------------------------------------------------------------
    def initialize(self) -> EngineState:
        budget = self._remaining_evaluations(None)
        payload, evals, mean, fit = self._start_run_payload(0, None, max_evaluations=budget)
        return EngineState(
            step=0,
            evaluations=int(evals),
            best_position=mean.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload=payload,
        )

    def _restart(self, state: EngineState, seed_position: np.ndarray | None = None, reason: str = "restart") -> None:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return
        if int(state.payload.get("bipop_large_restarts_started", 0)) >= self._max_large_restarts:
            state.termination_reason = "max_large_restarts"
            state.terminated = True
            return
        state.payload["termination_reason_last"] = str(reason)
        restart_index = int(state.payload.get("restarts", 0)) + 1
        payload, evals, pos, fit = self._start_run_payload(
            restart_index,
            state.payload,
            max_evaluations=remaining,
            seed_position=seed_position,
        )
        if evals <= 0:
            return
        payload["restart_reason"] = str(reason)
        state.payload = payload
        state.evaluations += int(evals)
        if self._is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)

    def _generation_stop_reason(self, payload: dict[str, Any], fit: np.ndarray | None = None, order: np.ndarray | None = None) -> str | None:
        dim = int(self.problem.dimension)
        lam = max(1, int(payload.get("lambda", self._base_lambda)))
        run_step = int(payload.get("run_step", 0))
        sigma = float(payload.get("sigma", self._large_sigma0()))
        sigma0 = max(self._min_sigma, float(payload.get("sigma0", self._large_sigma0())))
        C = np.asarray(payload.get("C", np.eye(dim)), dtype=float)
        pc = np.asarray(payload.get("pc", np.zeros(dim)), dtype=float)
        mean = np.asarray(payload.get("mean", np.zeros(dim)), dtype=float)

        max_iter = int(math.ceil(100.0 + 50.0 * ((dim + 3.0) ** 2.0) / math.sqrt(max(1.0, float(lam)))))
        if run_step >= max_iter:
            return "maxiter"

        besthist = [float(v) for v in payload.get("best_history", [])]
        medhist = [float(v) for v in payload.get("median_history", [])]
        hist_len = 10 + int(math.ceil(30.0 * dim / max(1, lam)))
        if len(besthist) >= hist_len:
            recent_best = np.asarray(besthist[-hist_len:], dtype=float)
            if recent_best.size and float(np.max(recent_best) - np.min(recent_best)) < self._tolfun:
                return "tolhistfun"

        equalfun_history = [bool(v) for v in payload.get("equalfun_history", [])]
        if len(equalfun_history) >= dim and sum(equalfun_history[-dim:]) > dim / 3.0:
            return "equalfunvals"

        eigvals, B = np.linalg.eigh(0.5 * (C + C.T))
        eigvals = np.maximum(eigvals, 0.0)
        sqrt_diag = np.sqrt(np.maximum(np.diag(C), 0.0))
        sigma_ratio = sigma / sigma0
        if np.all(sigma_ratio * np.abs(pc) < self._tolx) and np.all(sigma_ratio * sqrt_diag < self._tolx):
            return "tolx"

        max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
        if max_eig > 0.0 and sigma_ratio > self._tolupsigma * math.sqrt(max_eig):
            return "tolupsigma"

        stagnation_len = int(math.ceil(0.2 * run_step + 120.0 + 30.0 * dim / max(1, lam)))
        if len(besthist) >= max(40, stagnation_len) and len(medhist) >= max(40, stagnation_len):
            recent_best = besthist[-stagnation_len:]
            recent_med = medhist[-stagnation_len:]
            newest_best = float(np.median(recent_best[-20:]))
            oldest_best = float(np.median(recent_best[:20]))
            newest_med = float(np.median(recent_med[-20:]))
            oldest_med = float(np.median(recent_med[:20]))
            if self.problem.objective == "min":
                stagnant = newest_best >= oldest_best and newest_med >= oldest_med
            else:
                stagnant = newest_best <= oldest_best and newest_med <= oldest_med
            if stagnant:
                return "stagnation"

        positive = eigvals[eigvals > 0.0]
        if positive.size == 0 or float(np.max(positive) / max(np.min(positive), 1.0e-300)) > self._conditioncov_threshold:
            return "conditioncov"

        if eigvals.size:
            axis_index = run_step % dim
            axis_step = 0.1 * sigma * math.sqrt(float(eigvals[axis_index])) * B[:, axis_index]
            if np.array_equal(mean, mean + axis_step):
                return "noeffectaxis"
        coord_step = 0.2 * sigma * sqrt_diag
        if np.any(mean == mean + coord_step):
            return "noeffectcoord"
        return None

    # ------------------------------------------------------------------
    # Main CMA-ES generation
    # ------------------------------------------------------------------
    def step(self, state: EngineState) -> EngineState:
        remaining = self._remaining_evaluations(state)
        if remaining is not None and remaining <= 0:
            return state

        p = state.payload
        dim = int(self.problem.dimension)
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
        x = self._clip(raw_x)
        repair_count = int(np.count_nonzero(np.abs(raw_x - x) > 0.0))
        fit = np.asarray([float(self.problem.evaluate(row.tolist())) for row in x], dtype=float)

        pop = np.hstack((x, fit[:, None]))
        order = self._order(fit)
        best_idx = int(order[0])
        generation_best_fit = float(fit[best_idx])
        generation_best_pos = x[best_idx].copy()
        if self._is_better(generation_best_fit, state.best_fitness):
            state.best_fitness = generation_best_fit
            state.best_position = generation_best_pos.tolist()

        last_counts = self._zero_operator_count_map()
        last_contrib = self._zero_operator_float_map()
        last_counts["bipop_cmaes.cmaes_sampling"] = 1
        last_counts["bipop_cmaes.termination_check"] = 1
        last_counts["bipop_cmaes.boundary_repair"] = repair_count

        p.setdefault("function_history", []).extend(float(v) for v in fit.tolist())
        p.setdefault("median_history", []).append(float(np.median(fit)))
        k_zero = min(lam_eval - 1, int(math.ceil(0.1 + lam / 4.0)))
        equalfun = bool(lam_eval > 0 and float(fit[order[0]]) == float(fit[order[k_zero]]))
        p.setdefault("equalfun_history", []).append(equalfun)

        if self.problem.objective == "min":
            improvement = max(0.0, previous_best - float(state.best_fitness))
        else:
            improvement = max(0.0, float(state.best_fitness) - previous_best)

        # If the remaining budget is too small for a full selection update, keep
        # the incumbent update but avoid a covariance update from too few points.
        if lam_eval < mu:
            last_contrib["bipop_cmaes.cmaes_sampling"] = float(improvement)
            p.update(
                population=pop,
                run_step=int(p["run_step"]) + 1,
                run_evaluations=int(p["run_evaluations"]) + int(lam_eval),
                last_operator_counts=last_counts,
                last_operator_contributions=last_contrib,
            )
            p.setdefault("best_history", []).append(float(state.best_fitness))
            counts = p.setdefault("operator_counts", self._zero_operator_count_map())
            contrib = p.setdefault("operator_contributions", self._zero_operator_float_map())
            for key, value in last_counts.items():
                counts[key] = int(counts.get(key, 0)) + int(value)
            for key, value in last_contrib.items():
                contrib[key] = float(contrib.get(key, 0.0)) + float(value)
            state.payload = p
            state.step += 1
            state.evaluations += int(lam_eval)
            reason = self._generation_stop_reason(state.payload, fit=fit, order=order)
            if reason is not None and self._remaining_evaluations(state) not in {0}:
                self._restart(state, reason=reason)
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

        last_counts["bipop_cmaes.elite_recombination"] = 1
        last_counts["bipop_cmaes.distribution_update"] = 1
        last_counts["bipop_cmaes.step_size_adaptation"] = 1
        last_contrib["bipop_cmaes.cmaes_sampling"] = 0.5 * float(improvement)
        last_contrib["bipop_cmaes.elite_recombination"] = 0.5 * float(improvement)

        p.update(
            mean=mean,
            sigma=sigma,
            C=C,
            pc=pc,
            ps=ps,
            population=pop,
            run_step=int(p["run_step"]) + 1,
            run_evaluations=int(p["run_evaluations"]) + int(lam_eval),
            last_operator_counts=last_counts,
            last_operator_contributions=last_contrib,
        )
        p.setdefault("best_history", []).append(float(state.best_fitness))
        counts = p.setdefault("operator_counts", self._zero_operator_count_map())
        contrib = p.setdefault("operator_contributions", self._zero_operator_float_map())
        for key, value in last_counts.items():
            counts[key] = int(counts.get(key, 0)) + int(value)
        for key, value in last_contrib.items():
            contrib[key] = float(contrib.get(key, 0.0)) + float(value)

        state.payload = p
        state.step += 1
        state.evaluations += int(lam_eval)

        reason = self._generation_stop_reason(state.payload, fit=fit, order=order)
        if reason is not None and self._remaining_evaluations(state) not in {0}:
            self._restart(state, reason=reason)
        return state

    # ------------------------------------------------------------------
    # Telemetry and integration hooks
    # ------------------------------------------------------------------
    def observe(self, state: EngineState) -> dict:
        p = state.payload
        pop = np.asarray(p.get("population", []), dtype=float)
        C = np.asarray(p.get("C", np.eye(int(self.problem.dimension))), dtype=float)
        eigvals = np.linalg.eigvalsh(0.5 * (C + C.T)) if C.size else np.ones(int(self.problem.dimension))
        positive = eigvals[eigvals > 0.0]
        cond = float(np.max(positive) / max(np.min(positive), 1.0e-300)) if positive.size else float("inf")
        return {
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
            "bipop_regime": str(p.get("bipop_regime", "")),
            "bipop_selection_rule": str(p.get("bipop_selection_rule", "")),
            "bipop_large_budget": float(p.get("bipop_large_budget", 0.0)),
            "bipop_small_budget": float(p.get("bipop_small_budget", 0.0)),
            "bipop_large_restarts_started": int(p.get("bipop_large_restarts_started", 0)),
            "bipop_small_restarts_started": int(p.get("bipop_small_restarts_started", 0)),
            "bipop_latest_large_lambda": int(p.get("bipop_latest_large_lambda", self._base_lambda)),
            "operator_counts": dict(p.get("last_operator_counts", p.get("operator_counts", {}))),
            "operator_contributions": dict(p.get("last_operator_contributions", p.get("operator_contributions", {}))),
            "operator_counts_total": dict(p.get("operator_counts", {})),
            "operator_contributions_total": dict(p.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def inject_candidates(self, state: EngineState, candidates: list[CandidateRecord], policy: str = "native") -> EngineState:
        old_best = float(state.best_fitness)
        old_injections = int(state.payload.get("injections", 0))
        state = super().inject_candidates(state, candidates, policy=policy)
        injected = int(state.payload.get("injections", 0)) - old_injections
        if injected > 0:
            last_counts = self._zero_operator_count_map()
            last_contrib = self._zero_operator_float_map()
            last_counts["bipop_cmaes.candidate_injection"] = int(injected)
            if self.problem.objective == "min":
                delta = max(0.0, old_best - float(state.best_fitness))
            else:
                delta = max(0.0, float(state.best_fitness) - old_best)
            last_contrib["bipop_cmaes.candidate_injection"] = float(delta)
            counts = state.payload.setdefault("operator_counts", self._zero_operator_count_map())
            contrib = state.payload.setdefault("operator_contributions", self._zero_operator_float_map())
            counts["bipop_cmaes.candidate_injection"] = int(counts.get("bipop_cmaes.candidate_injection", 0)) + int(injected)
            contrib["bipop_cmaes.candidate_injection"] = float(contrib.get("bipop_cmaes.candidate_injection", 0.0)) + float(delta)
            state.payload["last_operator_counts"] = last_counts
            state.payload["last_operator_contributions"] = last_contrib
        return state

    def restart(
        self,
        state: EngineState,
        seeds: list[CandidateRecord] | None = None,
        preserve_best: bool = True,
    ) -> EngineState:
        seed = None
        if seeds:
            ranked = sorted(seeds, key=lambda c: c.fitness, reverse=(self.problem.objective == "max"))
            seed = np.asarray(ranked[0].position, dtype=float)
        self._restart(state, seed_position=seed, reason="manual_restart")
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
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=replace(self.capabilities),
            metadata={
                "algorithm_name": self.algorithm_name,
                "restarts": int(state.payload.get("restarts", 0)),
                "population_size": int(state.payload.get("lambda", 0)),
                "bipop_regime": str(state.payload.get("bipop_regime", "")),
                "bipop_large_restarts_started": int(state.payload.get("bipop_large_restarts_started", 0)),
                "bipop_small_restarts_started": int(state.payload.get("bipop_small_restarts_started", 0)),
                "bipop_large_budget": float(state.payload.get("bipop_large_budget", 0.0)),
                "bipop_small_budget": float(state.payload.get("bipop_small_budget", 0.0)),
                "sigma": float(state.payload.get("sigma", 0.0)),
                "injections": int(state.payload.get("injections", 0)),
                "elapsed_time": state.elapsed_time,
            },
        )
