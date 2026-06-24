"""
SAMSO - Surrogate-Assisted Multiswarm Optimization.

Paper-faithful implementation notes
-----------------------------------
This engine implements the mechanics described in Li, Cai, Gao and Shen
(IEEE TCYB 2021): LHS archive initialization, an RBF archive model, dynamic
S/L swarm sizes, SPSO updates with original/eigen coordinate systems, TLBO
learner-phase updates, distance-gated exact infill, and the paper's
self-improvement prescreening rule.  The original paper uses MATLAB fmincon
to optimize the RBF model; this package implementation uses a bounded random
multi-start surrogate search to preserve the architecture without depending on
an external nonlinear optimizer.
"""
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


class _CubicRBF:
    """Cubic RBF interpolant with a linear polynomial tail."""

    def __init__(self, x: np.ndarray, y: np.ndarray, ridge: float = 1e-10) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        keep = np.unique(np.round(x, decimals=14), axis=0, return_index=True)[1]
        keep.sort()
        self.x = x[keep]
        self.y = y[keep]
        self.xmin = self.x.min(axis=0)
        self.xmax = self.x.max(axis=0)
        span = self.xmax - self.xmin
        span[span == 0.0] = 1.0
        self.span = span
        xn = 2.0 * (self.x - self.xmin) / self.span - 1.0
        n, d = xn.shape
        if n < d + 2:
            self.mode = "linear"
            A = np.c_[np.ones(n), xn]
            self.beta = np.linalg.pinv(A) @ self.y
            self.alpha = np.zeros(n)
            self.nodes = xn
            return
        self.mode = "rbf"
        r = np.linalg.norm(xn[:, None, :] - xn[None, :, :], axis=2)
        phi = r ** 3
        P = np.c_[np.ones(n), xn]
        A = np.block([[phi + ridge * np.eye(n), P], [P.T, np.zeros((d + 1, d + 1))]])
        b = np.r_[self.y, np.zeros(d + 1)]
        theta = np.linalg.pinv(A) @ b
        self.alpha = theta[:n]
        self.beta = theta[n:]
        self.nodes = xn

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xn = 2.0 * (x - self.xmin) / self.span - 1.0
        P = np.c_[np.ones(xn.shape[0]), xn]
        if self.mode == "linear":
            return P @ self.beta
        r = np.linalg.norm(xn[:, None, :] - self.nodes[None, :, :], axis=2)
        return (r ** 3) @ self.alpha + P @ self.beta


class SAMSOEngine(BaseEngine):
    """Surrogate-assisted multiswarm optimization for expensive problems."""

    algorithm_id = "samso"
    algorithm_name = "Surrogate-Assisted Multiswarm Optimization"
    family = "swarm"
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_framework_constraints=True,
    )
    _REFERENCE = {
        "doi": "10.1109/TCYB.2020.2967553",
        "title": "A Surrogate-Assisted Multiswarm Optimization Algorithm for High-Dimensional Computationally Expensive Problems",
        "authors": "F. Li, X. Cai, L. Gao, and W. Shen",
        "venue": "IEEE Transactions on Cybernetics, 2021",
    }
    _DEFAULTS = {
        "population_size": None,       # 40 for d <= 50, otherwise 80
        "initial_samples": None,       # N for d <= 50, otherwise 2*d
        "w": 0.792,
        "c1": 1.491,
        "c2": 1.491,
        "Wnc": 1.0,
        "Pr": 0.5,
        "eta": None,
        "rbf_search_samples": 256,
        "rbf_local_samples": 64,
        "velocity_factor": 0.5,
    }
    _EVOMAPX_OPERATORS = (
        "samso.lhs_initialization",
        "samso.rbf_model_fit",
        "samso.rbf_optimum_infill",
        "samso.s_swarm_pso_update",
        "samso.l_swarm_tlbo_learner_update",
        "samso.prescreen_exact_evaluation",
        "samso.archive_update",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **(config.params or {})}
        d = int(problem.dimension)
        n_default = 40 if d <= 50 else 80
        self._n = max(4, int(p["population_size"] or n_default))
        k_default = self._n if d <= 50 else 2 * d
        self._k = max(self._n, int(p["initial_samples"] or k_default))
        self._w = float(p["w"])
        self._c1 = float(p["c1"])
        self._c2 = float(p["c2"])
        self._wnc = float(p["Wnc"])
        self._pr = float(p["Pr"])
        self._rbf_search_samples = max(16, int(p["rbf_search_samples"]))
        self._rbf_local_samples = max(8, int(p["rbf_local_samples"]))
        self._velocity_factor = float(p["velocity_factor"])
        lo = np.asarray(problem.min_values, dtype=float)
        hi = np.asarray(problem.max_values, dtype=float)
        if p["eta"] is None:
            self._eta = min(math.sqrt(0.001) * d, 5.0e-4 * d * float(np.min(hi - lo)))
        else:
            self._eta = float(p["eta"])
        self._rng = np.random.default_rng(config.seed)
        self._last_counts = {op: 0 for op in self._EVOMAPX_OPERATORS}
        self._last_contrib = {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def _lhs(self, n: int) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        d = self.problem.dimension
        u = np.empty((n, d), dtype=float)
        for j in range(d):
            u[:, j] = (self._rng.permutation(n) + self._rng.random(n)) / n
        return lo + u * (hi - lo)

    def _best_index(self, fitness: np.ndarray) -> int:
        return int(np.argmin(fitness) if self.problem.objective == "min" else np.argmax(fitness))

    def _fitness_better_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a < b if self.problem.objective == "min" else a > b

    def _sort_indices(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _score_for_surrogate_min(self, values: np.ndarray) -> np.ndarray:
        return values if self.problem.objective == "min" else -values

    def _min_distance_to_db(self, x: np.ndarray, db_x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        db_x = np.asarray(db_x, dtype=float)
        if db_x.ndim == 2 and db_x.shape[1] == self.problem.dimension + 1:
            db_x = db_x[:, :-1]
        return np.min(np.linalg.norm(x[:, None, :] - db_x[None, :, :], axis=2), axis=1)

    def _build_rbf(self, db: np.ndarray) -> _CubicRBF:
        return _CubicRBF(db[:, :-1], db[:, -1])

    def _surrogate_argbest(self, rbf: _CubicRBF, db: np.ndarray, gbest: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        d = self.problem.dimension
        global_pool = self._lhs(self._rbf_search_samples)
        scale = (hi - lo) * 0.15
        local_pool = gbest + self._rng.normal(0.0, scale, size=(self._rbf_local_samples, d))
        elite = db[self._sort_indices(db[:, -1])[: min(8, db.shape[0])], :-1]
        pool = np.vstack([global_pool, local_pool, elite, gbest.reshape(1, -1)])
        pool = np.clip(pool, lo, hi)
        pred = rbf.predict(pool)
        return pool[int(np.argmin(self._score_for_surrogate_min(pred)))]

    def _eigen_basis(self, db: np.ndarray) -> np.ndarray:
        d = self.problem.dimension
        m = min(max(2 * d, d + 1), db.shape[0])
        if m < 2:
            return np.eye(d)
        elite = db[self._sort_indices(db[:, -1])[:m], :-1]
        centered = elite - elite.mean(axis=0)
        try:
            cov = np.cov(centered, rowvar=False)
            vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
            order = np.argsort(vals)[::-1]
            return vecs[:, order]
        except Exception:
            return np.eye(d)

    def _dynamic_sizes(self, evaluations: int) -> tuple[int, int]:
        max_fe = self.config.max_evaluations or max(evaluations + self._n, 1000)
        remaining = max(0.0, (float(max_fe) - float(evaluations)) / max(1.0, float(max_fe)))
        n_l = min(self._n, int(2 + math.floor(self._n * (remaining ** self._wnc))))
        n_s = max(0, self._n - n_l)
        return n_s, n_l

    def _blank_counts(self) -> dict[str, int]:
        return {op: 0 for op in self._EVOMAPX_OPERATORS}

    def _blank_contrib(self) -> dict[str, float]:
        return {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def initialize(self) -> EngineState:
        x0 = self._lhs(self._k)
        f0 = self._evaluate_population(x0)
        db = np.c_[x0, f0]
        order = self._sort_indices(f0)
        pop_x = x0[order[: self._n]].copy()
        pop_f = f0[order[: self._n]].copy()
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        vmax = self._velocity_factor * (hi - lo)
        vel = self._rng.uniform(-vmax, vmax, size=pop_x.shape)
        pbest_x = pop_x.copy()
        pbest_f = pop_f.copy()
        bi = self._best_index(pop_f)
        counts = self._blank_counts()
        counts["samso.lhs_initialization"] = int(self._k)
        contrib = self._blank_contrib()
        self._last_counts = counts
        self._last_contrib = contrib
        return EngineState(
            step=0,
            evaluations=int(self._k),
            best_position=pop_x[bi].tolist(),
            best_fitness=float(pop_f[bi]),
            initialized=True,
            payload={
                "population": np.c_[pop_x, pop_f],
                "velocity": vel,
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "archive": db,
                "operator_counts": counts,
                "operator_contributions": contrib,
                "last_operator_counts": counts,
                "last_operator_contributions": contrib,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        pop = np.asarray(state.payload["population"], dtype=float)
        pop_x = pop[:, :-1].copy()
        pop_f = pop[:, -1].copy()
        vel = np.asarray(state.payload["velocity"], dtype=float).copy()
        pbest_x = np.asarray(state.payload["pbest_x"], dtype=float).copy()
        pbest_f = np.asarray(state.payload["pbest_f"], dtype=float).copy()
        db = np.asarray(state.payload["archive"], dtype=float).copy()
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        vmax = self._velocity_factor * (hi - lo)
        d = self.problem.dimension
        counts = self._blank_counts()
        contrib = self._blank_contrib()
        old_best = float(state.best_fitness)

        rbf = self._build_rbf(db)
        counts["samso.rbf_model_fit"] = 1
        gbest_idx = self._best_index(pop_f)
        gbest = pop_x[gbest_idx].copy()
        xmin = self._surrogate_argbest(rbf, db, gbest)
        if self._min_distance_to_db(xmin, db)[0] > self._eta:
            fx = self.problem.evaluate(xmin.copy())
            db = np.vstack([db, np.r_[xmin, fx]])
            state.evaluations += 1
            counts["samso.rbf_optimum_infill"] = 1
            counts["samso.archive_update"] += 1
            if self.problem.is_better(fx, state.best_fitness):
                contrib["samso.rbf_optimum_infill"] += abs(old_best - float(fx))
                state.best_fitness = float(fx)
                state.best_position = xmin.tolist()
                gbest = xmin.copy()

        n_s, _ = self._dynamic_sizes(state.evaluations)
        basis = self._eigen_basis(db)
        new_x = pop_x.copy()
        new_vel = vel.copy()
        whole = pop_x.copy()
        for i in range(self._n):
            if i < n_s:
                r1 = self._rng.random(d)
                r2 = self._rng.random(d)
                if self._rng.random() < self._pr:
                    R1 = basis @ np.diag(r1) @ basis.T
                    R2 = basis @ np.diag(r2) @ basis.T
                    vi = self._w * vel[i] + self._c1 * (R1 @ (pbest_x[i] - pop_x[i])) + self._c2 * (R2 @ (gbest - pop_x[i]))
                else:
                    vi = self._w * vel[i] + self._c1 * r1 * (pbest_x[i] - pop_x[i]) + self._c2 * r2 * (gbest - pop_x[i])
                vi = np.clip(vi, -vmax, vmax)
                new_vel[i] = vi
                new_x[i] = np.clip(pop_x[i] + vi, lo, hi)
                counts["samso.s_swarm_pso_update"] += 1
            else:
                choices = [j for j in range(self._n) if j != i]
                j = int(self._rng.choice(choices)) if choices else i
                r = self._rng.random(d)
                if self.problem.is_better(pop_f[i], pop_f[j]):
                    trial = pop_x[i] + r * (pop_x[i] - whole[j])
                else:
                    trial = pop_x[i] + r * (whole[j] - pop_x[i])
                new_x[i] = np.clip(trial, lo, hi)
                counts["samso.l_swarm_tlbo_learner_update"] += 1

        pred = rbf.predict(new_x)
        predicted_better = self._fitness_better_mask(pred, pop_f)
        far = self._min_distance_to_db(new_x, db) > self._eta
        selected = np.where(predicted_better & far)[0].tolist()
        if not selected:
            selected = [int(np.argmin(self._score_for_surrogate_min(pred)))]
        counts["samso.prescreen_exact_evaluation"] = len(selected)
        for i in selected:
            cand = new_x[i].copy()
            fit = self.problem.evaluate(cand)
            state.evaluations += 1
            db = np.vstack([db, np.r_[cand, fit]])
            counts["samso.archive_update"] += 1
            if self.problem.is_better(fit, pop_f[i]):
                before = float(pop_f[i])
                pop_x[i] = cand
                pop_f[i] = float(fit)
                if self.problem.is_better(fit, pbest_f[i]):
                    pbest_x[i] = cand
                    pbest_f[i] = float(fit)
                if self.problem.is_better(fit, state.best_fitness):
                    state.best_fitness = float(fit)
                    state.best_position = cand.tolist()
                contrib["samso.prescreen_exact_evaluation"] += abs(before - float(fit))

        state.payload = {
            "population": np.c_[pop_x, pop_f],
            "velocity": new_vel,
            "pbest_x": pbest_x,
            "pbest_f": pbest_f,
            "archive": db,
            "operator_counts": {k: int(state.payload.get("operator_counts", {}).get(k, 0)) + int(v) for k, v in counts.items()},
            "operator_contributions": {k: float(state.payload.get("operator_contributions", {}).get(k, 0.0)) + float(v) for k, v in contrib.items()},
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
        }
        self._last_counts = counts
        self._last_contrib = contrib
        state.step += 1
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        pop = np.asarray(state.payload["population"])
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(pop[:, -1])),
            "archive_size": int(np.asarray(state.payload["archive"]).shape[0]),
            "operator_counts": dict(state.payload.get("last_operator_counts", self._last_counts)),
            "operator_contributions": dict(state.payload.get("last_operator_contributions", self._last_contrib)),
            "operator_contributions_total": dict(state.payload.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native",
            "native_evomapx_operator_labels": True,
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(list(state.best_position), float(state.best_fitness), self.algorithm_id, state.step, "best")

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
                "elapsed_time": state.elapsed_time,
                "reference": dict(self._REFERENCE),
                "defaults": dict(self._DEFAULTS),
                "operator_counts": dict(state.payload.get("operator_counts", {})),
                "operator_contributions": dict(state.payload.get("operator_contributions", {})),
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"])
        return [
            CandidateRecord(pop[i, :-1].tolist(), float(pop[i, -1]), self.algorithm_id, state.step, "current")
            for i in range(pop.shape[0])
        ]


__all__ = ["SAMSOEngine"]
