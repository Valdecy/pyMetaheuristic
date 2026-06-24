"""
L2SMEA - Linear Subspace Surrogate Modeling Evolutionary Algorithm.

This standalone engine follows the single-objective L2SMEA procedure from Si
et al.: 2D LHS archive initialization, Gaussian-mutation construction of NL
linear subspaces from NE elites, NT nearest training associations, surrogate
subproblem optimization on each one-dimensional subspace, bi-criterion infill
selection, expensive evaluation, archive update, and Gaussian mutation parameter
update.  The paper uses ordinary Kriging and NSGA-III internally; this engine
uses a numerically robust one-dimensional Gaussian RBF/Kriging-like interpolant
and a bounded multi-start subproblem search so it remains dependency-light inside
pyMetaheuristic.
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


class _OneDimSurrogate:
    """Gaussian radial interpolant on projected 1-D subspace coordinates."""

    def __init__(self, t: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> None:
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        order = np.argsort(t[:, 0])
        t = t[order]
        y = y[order]
        keep = np.unique(np.round(t[:, 0], 14), return_index=True)[1]
        keep.sort()
        self.t = t[keep]
        self.y = y[keep]
        n = len(self.y)
        if n <= 2:
            self.mode = "linear"
            self.beta = np.linalg.pinv(np.c_[np.ones(n), self.t[:, 0]]) @ self.y
            self.alpha = np.zeros(n)
            self.gamma = 1.0
            return
        span = float(np.ptp(self.t[:, 0])) or 1.0
        self.gamma = 1.0 / (span * span)
        r2 = (self.t - self.t.T) ** 2
        K = np.exp(-self.gamma * r2)
        P = np.c_[np.ones(n), self.t[:, 0]]
        A = np.block([[K + ridge * np.eye(n), P], [P.T, np.zeros((2, 2))]])
        b = np.r_[self.y, np.zeros(2)]
        theta = np.linalg.pinv(A) @ b
        self.alpha = theta[:n]
        self.beta = theta[n:]
        self.mode = "rbf"

    def predict(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        P = np.c_[np.ones(t.shape[0]), t[:, 0]]
        if self.mode == "linear":
            return P @ self.beta
        r2 = (t - self.t.T) ** 2
        return np.exp(-self.gamma * r2) @ self.alpha + P @ self.beta


class L2SMEAEngine(BaseEngine):
    """Linear subspace surrogate modeling EA for large-scale expensive search."""

    algorithm_id = "l2smea"
    algorithm_name = "Linear Subspace Surrogate Modeling Evolutionary Algorithm"
    family = "evolutionary"
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_framework_constraints=True,
    )
    _REFERENCE = {
        "doi": "10.1109/TEVC.2023.3319640",
        "title": "Linear subspace surrogate modeling for large-scale expensive single/multi-objective optimization",
        "authors": "L. Si, X. Zhang, Y. Tian, S. Yang, L. Zhang, and Y. Jin",
        "venue": "IEEE Transactions on Evolutionary Computation",
    }
    _DEFAULTS = {
        "initial_samples": None,     # paper: NI = 2D
        "population_size": 40,       # paper: NP = 40
        "NLinear": 8,                # paper: NL = 8
        "n_elite": None,            # paper: NE = 2*NL
        "n_training": 10,           # paper: NT = 10
        "inner_iterations": 30,      # paper NSGA-III Iter = 30
        "infill_per_subspace": 1,
        "sigma": 1.0,
        "candidate_pool_multiplier": 1,
    }
    _EVOMAPX_OPERATORS = (
        "l2smea.lhs_initialization",
        "l2smea.gaussian_subspace_construction",
        "l2smea.linear_subspace_surrogate_fit",
        "l2smea.multi_task_candidate_search",
        "l2smea.bi_criteria_infill_selection",
        "l2smea.expensive_evaluation_archive_update",
        "l2smea.gaussian_parameter_update",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **(config.params or {})}
        d = int(problem.dimension)
        self._ni = max(2, int(p["initial_samples"] or 2 * d))
        self._npop = max(4, int(p["population_size"]))
        self._nl = max(1, int(p["NLinear"]))
        self._ne = max(2, int(p["n_elite"] or 2 * self._nl))
        self._nt = max(2, int(p["n_training"]))
        self._iter = max(1, int(p["inner_iterations"]))
        self._infill_per_subspace = max(1, int(p["infill_per_subspace"]))
        self._sigma0 = float(p["sigma"])
        self._pool_multiplier = max(1, int(p["candidate_pool_multiplier"]))
        self._rng = np.random.default_rng(config.seed)
        self._last_counts = {op: 0 for op in self._EVOMAPX_OPERATORS}
        self._last_contrib = {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def _lhs_unit(self, n: int, d: int) -> np.ndarray:
        u = np.empty((n, d), dtype=float)
        for j in range(d):
            u[:, j] = (self._rng.permutation(n) + self._rng.random(n)) / n
        return u

    def _denormalize(self, z: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        return lo + np.clip(z, 0.0, 1.0) * (hi - lo)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        span = hi - lo
        span[span == 0.0] = 1.0
        return np.clip((x - lo) / span, 0.0, 1.0)

    def _sort_indices(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _score_min(self, y: np.ndarray) -> np.ndarray:
        return y if self.problem.objective == "min" else -y

    def _elite_stats(self, z: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        elite = z[self._sort_indices(f)[: min(self._ne, len(f))]]
        m = elite.shape[0]
        if m == 1:
            mu = elite[0]
            cov = np.eye(z.shape[1]) * 0.05
        else:
            weights = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
            weights = weights / weights.sum()
            mu = weights @ elite
            centered = elite - mu
            cov = (centered.T * weights) @ centered + 1e-8 * np.eye(z.shape[1])
        sigma = self._sigma0 * max(0.05, min(1.0, float(np.mean(np.std(elite, axis=0))) * 2.0 if m > 1 else 0.2))
        return mu, cov, sigma

    def _sample_gaussian_point(self, mu: np.ndarray, cov: np.ndarray, sigma: float) -> np.ndarray:
        try:
            v = self._rng.multivariate_normal(mu, cov, check_valid="ignore")
        except Exception:
            v = mu + self._rng.normal(0.0, 0.1, size=mu.shape[0])
        return np.clip(mu + sigma * (v - mu), 0.0, 1.0)

    @staticmethod
    def _project_to_line(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-14:
            direction = np.zeros_like(a)
            direction[0] = 1.0
            norm = 1.0
        unit = direction / norm
        t = (points - a) @ unit
        projection = a + np.outer(t, unit)
        dist = np.linalg.norm(points - projection, axis=1)
        return t, dist

    def _nondominated_indices(self, pred: np.ndarray, dist: np.ndarray) -> list[int]:
        # Minimize predicted fitness and maximize minimum distance to archive.
        p = self._score_min(pred)
        dcrit = -dist
        n = len(p)
        nd: list[int] = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if j == i:
                    continue
                if (p[j] <= p[i] and dcrit[j] <= dcrit[i]) and (p[j] < p[i] or dcrit[j] < dcrit[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        if not nd:
            nd = [int(np.argmin(p))]
        nd.sort(key=lambda i: (p[i], dcrit[i]))
        return nd

    def _blank_counts(self) -> dict[str, int]:
        return {op: 0 for op in self._EVOMAPX_OPERATORS}

    def _blank_contrib(self) -> dict[str, float]:
        return {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def initialize(self) -> EngineState:
        d = self.problem.dimension
        z0 = self._lhs_unit(self._ni, d)
        x0 = self._denormalize(z0)
        f0 = self._evaluate_population(x0)
        archive = np.c_[x0, f0]
        bi = int(self._sort_indices(f0)[0])
        counts = self._blank_counts()
        counts["l2smea.lhs_initialization"] = self._ni
        contrib = self._blank_contrib()
        self._last_counts = counts
        self._last_contrib = contrib
        return EngineState(
            step=0,
            evaluations=self._ni,
            best_position=x0[bi].tolist(),
            best_fitness=float(f0[bi]),
            initialized=True,
            payload={
                "archive": archive,
                "population": archive.copy(),
                "operator_counts": counts,
                "operator_contributions": contrib,
                "last_operator_counts": counts,
                "last_operator_contributions": contrib,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        archive = np.asarray(state.payload["archive"], dtype=float).copy()
        x_arc = archive[:, :-1]
        f_arc = archive[:, -1]
        z_arc = self._normalize(x_arc)
        mu, cov, sigma = self._elite_stats(z_arc, f_arc)
        counts = self._blank_counts()
        contrib = self._blank_contrib()
        new_rows: list[np.ndarray] = []
        old_best = float(state.best_fitness)
        d = self.problem.dimension
        candidates_per_subspace = max(self._npop * self._iter * self._pool_multiplier, self._npop)

        for _ in range(self._nl):
            a = self._sample_gaussian_point(mu, cov, sigma)
            b = self._sample_gaussian_point(mu, cov, sigma)
            if np.linalg.norm(a - b) <= 1e-10:
                b = np.clip(a + self._rng.normal(0.0, 0.05, size=d), 0.0, 1.0)
            counts["l2smea.gaussian_subspace_construction"] += 1
            t_all, dist_all = self._project_to_line(z_arc, a, b)
            train_idx = np.argsort(dist_all)[: min(self._nt, len(dist_all))]
            surrogate = _OneDimSurrogate(t_all[train_idx], f_arc[train_idx])
            counts["l2smea.linear_subspace_surrogate_fit"] += 1

            # The paper uses NSGA-III on the NL-dimensional multitask encoding.
            # Here each 1-D subproblem is searched by LHS plus endpoint/elite seeds.
            direction = b - a
            length = float(np.linalg.norm(direction)) or 1.0
            t_min, t_max = 0.0, length
            t_pool = np.r_[
                self._rng.random(candidates_per_subspace) * (t_max - t_min) + t_min,
                np.array([t_min, t_max, 0.5 * (t_min + t_max)]),
            ]
            z_pool = a + np.outer(t_pool / length, direction)
            z_pool = np.clip(z_pool, 0.0, 1.0)
            t_pool, _ = self._project_to_line(z_pool, a, b)
            pred = surrogate.predict(t_pool)
            min_dist = np.min(np.linalg.norm(z_pool[:, None, :] - z_arc[None, :, :], axis=2), axis=1)
            nd = self._nondominated_indices(pred, min_dist)[: self._infill_per_subspace]
            counts["l2smea.multi_task_candidate_search"] += int(len(t_pool))
            counts["l2smea.bi_criteria_infill_selection"] += len(nd)
            for idx in nd:
                x = self._denormalize(z_pool[idx])
                fit = self.problem.evaluate(x.copy())
                state.evaluations += 1
                row = np.r_[x, fit]
                new_rows.append(row)
                counts["l2smea.expensive_evaluation_archive_update"] += 1
                if self.problem.is_better(fit, state.best_fitness):
                    contrib["l2smea.expensive_evaluation_archive_update"] += abs(float(state.best_fitness) - float(fit))
                    state.best_fitness = float(fit)
                    state.best_position = x.tolist()

        if new_rows:
            archive = np.vstack([archive, np.vstack(new_rows)])
        counts["l2smea.gaussian_parameter_update"] = 1
        order = self._sort_indices(archive[:, -1])
        population = archive[order[: min(max(self._npop, self._ne), archive.shape[0])]].copy()
        prev_counts = state.payload.get("operator_counts", {})
        prev_contrib = state.payload.get("operator_contributions", {})
        state.payload = {
            "archive": archive,
            "population": population,
            "operator_counts": {k: int(prev_counts.get(k, 0)) + int(v) for k, v in counts.items()},
            "operator_contributions": {k: float(prev_contrib.get(k, 0.0)) + float(v) for k, v in contrib.items()},
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
        }
        self._last_counts = counts
        self._last_contrib = contrib
        if self.problem.is_better(state.best_fitness, old_best):
            pass
        state.step += 1
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        archive = np.asarray(state.payload["archive"])
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "archive_size": int(archive.shape[0]),
            "operator_counts": dict(state.payload.get("last_operator_counts", self._last_counts)),
            "operator_contributions": dict(state.payload.get("last_operator_contributions", self._last_contrib)),
            "operator_contributions_total": dict(state.payload.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native_approx_subproblem_search",
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


__all__ = ["L2SMEAEngine"]
