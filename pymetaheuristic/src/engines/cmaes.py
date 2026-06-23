"""
pyMetaheuristic src — Covariance Matrix Adaptation Evolution Strategy Engine
=============================================================================

Paper-faithful CMA core after Hansen & Ostermeier (1996): a (1,lambda)-ES
that samples normally distributed offspring, selects the best offspring as the
next parent, and adapts both an evolution-path covariance matrix and a global
step size by cumulation.
"""
from __future__ import annotations

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


class CMAESEngine(BaseEngine):
    """Covariance Matrix Adaptation Evolution Strategy."""

    algorithm_id = "cmaes"
    algorithm_name = "Covariance Matrix Adaptation Evolution Strategy"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/ICEC.1996.542381",
        "title": "Adapting Arbitrary Normal Mutation Distributions in Evolution Strategies: The Covariance Matrix Adaptation",
        "authors": "Nikolaus Hansen and Andreas Ostermeier",
        "year": 1996,
    }
    _DEFAULTS = dict(
        population_size=10,   # lambda in the paper's (1,10)-ES simulations
        sigma=0.1,            # delta_start in the paper's simulations
        cumulation=None,      # c = 1/sqrt(n) when None
        beta=None,            # beta = 1/n when None
        ccov=None,            # ccov = 2/n^2 when None
    )
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )

    _EVOMAPX_DIRECT_OPERATORS = (
        "cmaes.offspring_sampling",
        "cmaes.parent_selection",
        "cmaes.candidate_injection",
    )
    _EVOMAPX_DIAGNOSTIC_OPERATORS = (
        "cmaes.evolution_path_update",
        "cmaes.covariance_update",
        "cmaes.step_size_update",
        "cmaes.boundary_repair",
        "cmaes.initialization",
    )
    _EVOMAPX_OPERATORS = _EVOMAPX_DIRECT_OPERATORS + _EVOMAPX_DIAGNOSTIC_OPERATORS

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **(config.params or {})}
        self._dim = int(problem.dimension)
        self._lambda = max(2, int(self._params.get("population_size", 10)))
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Paper constants and utilities
    # ------------------------------------------------------------------

    def _constants(self) -> dict[str, float]:
        n = max(1, self._dim)
        c_default = 1.0 / np.sqrt(float(n))
        beta_default = 1.0 / float(n)
        ccov_default = 2.0 / float(n * n)
        c = self._params.get("cumulation", self._params.get("c", None))
        beta = self._params.get("beta", None)
        ccov = self._params.get("ccov", None)
        c = c_default if c is None else float(c)
        beta = beta_default if beta is None else float(beta)
        ccov = ccov_default if ccov is None else float(ccov)
        c = float(np.clip(c, 1.0e-12, 1.0))
        beta = max(float(beta), 1.0e-12)
        ccov = float(np.clip(ccov, 1.0e-12, 1.0 - 1.0e-12))
        c_u = float(np.sqrt(c * (2.0 - c)))
        chi_n = float(np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)))
        return {"c": c, "beta": beta, "ccov": ccov, "c_u": c_u, "chi_n": chi_n}

    @staticmethod
    def _zero_int_map() -> dict[str, int]:
        return {label: 0 for label in CMAESEngine._EVOMAPX_OPERATORS}

    @staticmethod
    def _zero_float_map() -> dict[str, float]:
        return {label: 0.0 for label in CMAESEngine._EVOMAPX_OPERATORS}

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        return np.argsort(fitness) if self.problem.objective == "min" else np.argsort(-fitness)

    def _objective_improvement(self, old: float, new: float) -> float:
        if not np.isfinite(old) or not np.isfinite(new):
            return 0.0
        return max(0.0, old - new) if self.problem.objective == "min" else max(0.0, new - old)

    def _eigensystem(self, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1.0e-300)
        return vals, vecs

    def _evaluate_position(self, x: np.ndarray) -> tuple[np.ndarray, float, bool]:
        raw = np.asarray(x, dtype=float).copy()
        candidate = raw.copy()
        fitness = float(self.problem.evaluate(candidate))
        repaired = not np.allclose(candidate, raw, rtol=0.0, atol=1.0e-12)
        return np.asarray(candidate, dtype=float).copy(), fitness, bool(repaired)

    def _make_initial_mean(self) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        initial = self._params.get("initial_mean", None)
        if initial is None:
            mean = self._rng.uniform(lo, hi)
        else:
            mean = np.asarray(initial, dtype=float)
            if mean.shape != (self._dim,):
                raise ValueError(f"initial_mean must have shape {(self._dim,)}, received {mean.shape}")
        return self.problem.apply_variable_types(np.clip(mean, lo, hi)).astype(float)

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        const = self._constants()
        mean = self._make_initial_mean()
        sigma = max(float(self._params.get("sigma", 0.1)), 1.0e-300)
        mean, fit, repaired = self._evaluate_position(mean)

        counts = self._zero_int_map()
        contrib = self._zero_float_map()
        last_counts = self._zero_int_map()
        last_contrib = self._zero_float_map()
        counts["cmaes.initialization"] = 1
        last_counts["cmaes.initialization"] = 1
        if repaired:
            counts["cmaes.boundary_repair"] += 1
            last_counts["cmaes.boundary_repair"] += 1

        population = np.asarray([[*mean.tolist(), float(fit)]], dtype=float)
        payload: dict[str, Any] = {
            "mean": mean.copy(),
            "sigma": float(sigma),
            "sigma0": float(sigma),
            "C": np.eye(self._dim),
            "s_cov": np.zeros(self._dim),
            "s_sigma": np.zeros(self._dim),
            "lambda": int(self._lambda),
            "cumulation": float(const["c"]),
            "c_u": float(const["c_u"]),
            "beta": float(const["beta"]),
            "ccov": float(const["ccov"]),
            "chi_n": float(const["chi_n"]),
            "generation": 0,
            "population": population,
            "offspring_z": np.zeros((0, self._dim)),
            "offspring_y": np.zeros((0, self._dim)),
            "operator_counts": counts,
            "operator_contributions": contrib,
            "last_operator_counts": last_counts,
            "last_operator_contributions": last_contrib,
            "injections": 0,
            "boundary_repairs": int(repaired),
        }
        return EngineState(
            step=0,
            evaluations=1,
            best_position=mean.tolist(),
            best_fitness=float(fit),
            initialized=True,
            payload=payload,
        )

    def step(self, state: EngineState) -> EngineState:
        remaining = None
        if self.config.max_evaluations is not None:
            remaining = max(0, int(self.config.max_evaluations) - int(state.evaluations))
            if remaining <= 0:
                state.termination_reason = state.termination_reason or "max_evaluations"
                return state
        lam_eval = self._lambda if remaining is None else min(self._lambda, int(remaining))
        if lam_eval <= 0:
            return state

        p = state.payload
        mean = np.asarray(p["mean"], dtype=float)
        sigma = float(p["sigma"])
        C = np.asarray(p["C"], dtype=float)
        s_cov = np.asarray(p["s_cov"], dtype=float)
        s_sigma = np.asarray(p["s_sigma"], dtype=float)
        c = float(p["cumulation"])
        c_u = float(p["c_u"])
        beta = float(p["beta"])
        ccov = float(p["ccov"])
        chi_n = float(p["chi_n"])
        previous_best = float(state.best_fitness)

        eigvals, Q = self._eigensystem(C)
        B = Q * np.sqrt(eigvals)  # columns are eigenvectors scaled by sqrt eigenvalues
        z = self._rng.standard_normal((lam_eval, self._dim))
        y = z @ B.T
        raw_x = mean + sigma * y

        x = np.empty_like(raw_x)
        fitness = np.empty(lam_eval, dtype=float)
        repair_count = 0
        for i in range(lam_eval):
            x_i, f_i, repaired = self._evaluate_position(raw_x[i])
            x[i] = x_i
            fitness[i] = f_i
            repair_count += int(repaired)

        order = self._order(fitness)
        selected = int(order[0])
        selected_fitness = float(fitness[selected])
        selected_x = x[selected].copy()
        selected_y = y[selected].copy()
        selected_z = z[selected].copy()
        selected_normalized_step = Q @ selected_z

        # Paper equations (2) and (3): cumulate selected Bz and update C.
        s_cov = (1.0 - c) * s_cov + c_u * selected_y
        C = (1.0 - ccov) * C + ccov * np.outer(s_cov, s_cov)
        C = 0.5 * (C + C.T)
        eig_after, V_after = np.linalg.eigh(C)
        if np.any(eig_after <= 0.0):
            C = V_after @ np.diag(np.maximum(eig_after, 1.0e-300)) @ V_after.T
            C = 0.5 * (C + C.T)

        # Paper equations (4) and (5): cumulative global step-size adaptation.
        s_sigma = (1.0 - c) * s_sigma + c_u * selected_normalized_step
        sigma = float(sigma * np.exp(beta * (float(np.linalg.norm(s_sigma)) - chi_n)))
        sigma = float(np.clip(sigma, 1.0e-300, 1.0e300))

        mean = selected_x
        if self.problem.is_better(selected_fitness, state.best_fitness):
            state.best_fitness = selected_fitness
            state.best_position = selected_x.tolist()

        improvement = self._objective_improvement(previous_best, float(state.best_fitness))
        counts_total = dict(p.get("operator_counts", self._zero_int_map()))
        contrib_total = dict(p.get("operator_contributions", self._zero_float_map()))
        last_counts = self._zero_int_map()
        last_contrib = self._zero_float_map()
        for label, inc in (
            ("cmaes.offspring_sampling", 1),
            ("cmaes.parent_selection", 1),
            ("cmaes.evolution_path_update", 1),
            ("cmaes.covariance_update", 1),
            ("cmaes.step_size_update", 1),
            ("cmaes.boundary_repair", repair_count),
        ):
            counts_total[label] = int(counts_total.get(label, 0)) + int(inc)
            last_counts[label] = int(inc)
        last_contrib["cmaes.offspring_sampling"] = 0.5 * float(improvement)
        last_contrib["cmaes.parent_selection"] = 0.5 * float(improvement)
        for label, value in last_contrib.items():
            contrib_total[label] = float(contrib_total.get(label, 0.0)) + float(value)

        p.update(
            mean=mean.copy(),
            sigma=float(sigma),
            C=C.copy(),
            s_cov=s_cov.copy(),
            s_sigma=s_sigma.copy(),
            generation=int(p.get("generation", 0)) + 1,
            population=np.hstack((x, fitness[:, None])),
            offspring_z=z.copy(),
            offspring_y=y.copy(),
            selected_index=int(selected),
            selected_fitness=float(selected_fitness),
            selected_step=selected_y.copy(),
            operator_counts=counts_total,
            operator_contributions=contrib_total,
            last_operator_counts=last_counts,
            last_operator_contributions=last_contrib,
            boundary_repairs=int(p.get("boundary_repairs", 0)) + int(repair_count),
        )
        state.payload = p
        state.evaluations += int(lam_eval)
        state.step += 1
        return state

    def observe(self, state: EngineState) -> dict:
        p = state.payload
        C = np.asarray(p.get("C", np.eye(self._dim)), dtype=float)
        eigvals = np.linalg.eigvalsh(0.5 * (C + C.T))
        eigvals = np.maximum(eigvals, 1.0e-300)
        condition = float(np.max(eigvals) / np.min(eigvals)) if eigvals.size else 1.0
        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "population_size": int(p.get("lambda", self._lambda)),
            "sigma": float(p.get("sigma", 0.0)),
            "sigma0": float(p.get("sigma0", 0.0)),
            "cumulation": float(p.get("cumulation", 0.0)),
            "beta": float(p.get("beta", 0.0)),
            "ccov": float(p.get("ccov", 0.0)),
            "chi_n": float(p.get("chi_n", 0.0)),
            "condition_number": condition,
            "generation": int(p.get("generation", 0)),
            "boundary_repairs": int(p.get("boundary_repairs", 0)),
            "injections": int(p.get("injections", 0)),
            "operator_counts": dict(p.get("last_operator_counts", p.get("operator_counts", {}))),
            "operator_contributions": dict(p.get("last_operator_contributions", p.get("operator_contributions", {}))),
            "operator_counts_total": dict(p.get("operator_counts", {})),
            "operator_contributions_total": dict(p.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=int(state.step),
            role="best",
        )

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
                "reference": dict(self._REFERENCE),
                "elapsed_time": state.elapsed_time,
                "population_size": int(state.payload.get("lambda", self._lambda)),
                "sigma": float(state.payload.get("sigma", 0.0)),
                "cumulation": float(state.payload.get("cumulation", 0.0)),
                "beta": float(state.payload.get("beta", 0.0)),
                "ccov": float(state.payload.get("ccov", 0.0)),
                "evomapx_operator_labels": list(self._EVOMAPX_OPERATORS),
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload.get("population", []), dtype=float)
        if pop.ndim != 2 or pop.shape[1] < self._dim + 1:
            return [self.get_best_candidate(state)]
        return [
            CandidateRecord(
                position=pop[i, :-1].astype(float).tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="parent" if i == int(state.payload.get("selected_index", -1)) else "offspring",
            )
            for i in range(pop.shape[0])
        ]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        if not candidates:
            return state
        best_injected: tuple[np.ndarray, float] | None = None
        evaluated = 0
        for cand in candidates:
            pos = np.asarray(cand.position, dtype=float)
            pos = self.problem.apply_variable_types(pos)
            pos, fit, _ = self._evaluate_position(pos)
            evaluated += 1
            if best_injected is None or self.problem.is_better(fit, best_injected[1]):
                best_injected = (pos, fit)
        state.evaluations += int(evaluated)
        if best_injected is None:
            return state

        pos, fit = best_injected
        p = state.payload
        old_best = float(state.best_fitness)
        if self.problem.is_better(fit, state.best_fitness):
            state.best_position = pos.tolist()
            state.best_fitness = float(fit)
            p["mean"] = pos.copy()
            # External candidate injection is not in the 1996 CMA dynamics;
            # reset paths so injected states do not fake selected evolution paths.
            p["s_cov"] = np.zeros(self._dim)
            p["s_sigma"] = np.zeros(self._dim)
            p["population"] = np.asarray([[*pos.tolist(), float(fit)]], dtype=float)

        last_counts = self._zero_int_map()
        last_contrib = self._zero_float_map()
        counts_total = dict(p.get("operator_counts", self._zero_int_map()))
        contrib_total = dict(p.get("operator_contributions", self._zero_float_map()))
        last_counts["cmaes.candidate_injection"] = int(evaluated)
        last_contrib["cmaes.candidate_injection"] = self._objective_improvement(old_best, float(state.best_fitness))
        counts_total["cmaes.candidate_injection"] = int(counts_total.get("cmaes.candidate_injection", 0)) + int(evaluated)
        contrib_total["cmaes.candidate_injection"] = float(contrib_total.get("cmaes.candidate_injection", 0.0)) + float(last_contrib["cmaes.candidate_injection"])
        p["operator_counts"] = counts_total
        p["operator_contributions"] = contrib_total
        p["last_operator_counts"] = last_counts
        p["last_operator_contributions"] = last_contrib
        p["injections"] = int(p.get("injections", 0)) + int(evaluated)
        state.payload = p
        return state
