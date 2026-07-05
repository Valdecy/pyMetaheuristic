"""
Surrogate-Assisted Differential Evolution with Adaptive Multi-Subspace Search.

This engine implements the native mechanics described by Gu, Wang, and Jin
(IEEE TEVC 2023): Latin-hypercube archive initialization, cubic-RBF surrogate
models with a linear polynomial tail, adaptive original/PCA-mapped subspace
selection, DE/best/1/bin subspace search, and one exact archive update per
macro-generation.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _CubicRBF:
    """Cubic radial-basis model with the paper's linear polynomial tail."""

    centers: np.ndarray
    weights: np.ndarray
    tail: np.ndarray
    shift: np.ndarray
    scale: np.ndarray
    constant_fallback: float | None = None

    @staticmethod
    def _normalize(x: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - shift) / scale

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        ridge: float = 1.0e-10,
    ) -> "_CubicRBF":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] == 0:
            return cls(x, np.zeros(0), np.zeros(x.shape[1] + 1), np.zeros(x.shape[1]), np.ones(x.shape[1]), 0.0)
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        scale = np.maximum(hi - lo, 1.0e-12)
        z = cls._normalize(x, lo, scale)
        n, d = z.shape
        if n < d + 2 or np.allclose(y, y[0]):
            return cls(z, np.zeros(n), np.zeros(d + 1), lo, scale, float(np.mean(y)))
        dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        phi = dist ** 3
        p = np.hstack((z, np.ones((n, 1), dtype=float)))
        lhs = np.block(
            [
                [phi + ridge * np.eye(n), p],
                [p.T, np.zeros((d + 1, d + 1), dtype=float)],
            ]
        )
        rhs = np.concatenate((y, np.zeros(d + 1, dtype=float)))
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return cls(z, sol[:n], sol[n:], lo, scale, None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.constant_fallback is not None:
            return np.full(x.shape[0], float(self.constant_fallback), dtype=float)
        z = self._normalize(x, self.shift, self.scale)
        dist = np.linalg.norm(z[:, None, :] - self.centers[None, :, :], axis=2)
        phi = dist ** 3
        p = np.hstack((z, np.ones((z.shape[0], 1), dtype=float)))
        return phi @ self.weights + p @ self.tail


class SADEAMSSEngine(BaseEngine):
    algorithm_id = "sade_amss"
    algorithm_name = "Surrogate-Assisted Differential Evolution with Adaptive Multi-Subspace Search"
    family = "evolutionary"
    _REFERENCE = {
        "authors": "Gu, Wang, and Jin",
        "title": "Surrogate-Assisted Differential Evolution With Adaptive Multisubspace Search for Large-Scale Expensive Optimization",
        "journal": "IEEE Transactions on Evolutionary Computation",
        "year": 2023,
        "doi": "10.1109/TEVC.2022.3226837",
    }
    _DEFAULTS = {
        "population_size": 10,
        "initial_samples": 200,  # Ns
        "K": 20,
        "max_subspace_dim": 100,
        "subspace_iterations": 5,
        "F": 0.8,
        "CR": 1.0,
        "training_multiplier": 2,
        "tes": 50,
        "tr": 500,
        "alpha": 1.0,
        "eta": 0.9999,
        "beta": 2.0,
        "ratio": 3,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )

    _LABELS = {
        "lhs": "sade_amss.lhs_initialization",
        "strategy": "sade_amss.adaptive_strategy_switch",
        "original": "sade_amss.random_original_subspace_construction",
        "mapping": "sade_amss.pca_mapping_subspace_construction",
        "rbf": "sade_amss.cubic_rbf_fit_predict",
        "de": "sade_amss.de_best_1_binomial",
        "archive": "sade_amss.exact_evaluation_archive_update",
        "repair": "sade_amss.bound_repair",
    }

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **(config.params or {})}
        self._n = int(p.get("population_size", p.get("pop_size", 10)))
        if self._n < 3:
            raise ValueError("population_size must be at least 3 for DE/best/1.")
        self._Ns = int(p.get("initial_samples", p.get("Ns", 200)))
        if self._Ns < self._n:
            raise ValueError("initial_samples/Ns must be >= population_size.")
        self._K = max(1, int(p.get("K", 20)))
        self._maxd = max(1, int(p.get("max_subspace_dim", p.get("maxd", 100))))
        self._Gm = max(1, int(p.get("subspace_iterations", p.get("Gm", 5))))
        self._F = float(p.get("F", 0.8))
        self._CR = float(p.get("CR", 1.0))
        self._training_multiplier = max(1, int(p.get("training_multiplier", 2)))
        self._tes = max(1, int(p.get("tes", 50)))
        self._tr = max(1, int(p.get("tr", 500)))
        self._alpha = float(p.get("alpha", 1.0))
        self._eta = min(1.0, max(0.0, float(p.get("eta", 0.9999))))
        self._beta = float(p.get("beta", 2.0))
        self._ratio = max(1, int(p.get("ratio", 3)))
        self._rng = np.random.default_rng(None if config.seed is None else int(config.seed))

    # ------------------------------------------------------------------
    # initialization and exact archive handling
    # ------------------------------------------------------------------
    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        return lo, hi

    def _lhs(self, n: int, d: int) -> np.ndarray:
        sample = np.empty((n, d), dtype=float)
        for j in range(d):
            perm = self._rng.permutation(n)
            sample[:, j] = (perm + self._rng.random(n)) / n
        return sample

    def _evaluate_one(self, position: np.ndarray) -> tuple[np.ndarray, float]:
        repaired = np.asarray(self.problem.clip_position(position), dtype=float)
        fitness = float(self.problem.evaluate(repaired))
        return repaired, fitness

    def _empty_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._LABELS.values()}

    def _empty_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._LABELS.values()}

    def initialize(self) -> EngineState:
        lo, hi = self._bounds()
        positions = lo + self._lhs(self._Ns, self.problem.dimension) * (hi - lo)
        positions = np.asarray([self.problem.clip_position(row) for row in positions], dtype=float)
        fitness = self._evaluate_population(positions)
        order = self._order(fitness)
        archive_X = positions.copy()
        archive_y = fitness.copy()
        pop_X = archive_X[order[: self._n]].copy()
        pop_f = archive_y[order[: self._n]].copy()
        population = np.hstack((pop_X, pop_f[:, None]))
        best_idx = int(order[0])
        counts = self._empty_counts()
        counts[self._LABELS["lhs"]] = self._Ns
        contribs = self._empty_contribs()
        return EngineState(
            step=0,
            evaluations=self._Ns,
            best_position=archive_X[best_idx].tolist(),
            best_fitness=float(archive_y[best_idx]),
            initialized=True,
            payload={
                "population": population,
                "archive_X": archive_X,
                "archive_y": archive_y,
                "best_history": [float(archive_y[best_idx])],
                "strategy_k": 1,
                "tstop": 0,
                "last_space": "initialization",
                "operator_counts": counts,
                "operator_contributions": contribs,
                "native_evomapx_operator_labels": True,
            },
        )

    # ------------------------------------------------------------------
    # paper-specific mechanics
    # ------------------------------------------------------------------
    def _order(self, fitness: np.ndarray) -> np.ndarray:
        idx = np.argsort(fitness)
        return idx if self.problem.objective == "min" else idx[::-1]

    def _is_better_value(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _best_index(self, fitness: np.ndarray) -> int:
        return int(self._order(np.asarray(fitness, dtype=float))[0])

    def _signed_improvement(self, old: float, new: float) -> float:
        return float(old - new) if self.problem.objective == "min" else float(new - old)

    def _safe_log_metric(self, f: float, offset: float) -> float:
        value = float(f) + offset
        return float(np.log10(max(value, 1.0e-300)))

    def _maybe_switch_strategy(self, state: EngineState, counts: dict[str, int]) -> tuple[int, int]:
        history = [float(v) for v in state.payload.get("best_history", [])]
        k = int(state.payload.get("strategy_k", 1))
        tstop = int(state.payload.get("tstop", 0))
        t = len(history) - 1
        if t >= self._tr + 2 + tstop and k < 3 and len(history) >= self._tes + 3:
            min_h = min(history)
            offset = 1.0 - min_h if min_h <= 0.0 else 0.0
            logs = [self._safe_log_metric(v, offset) for v in history]
            second = []
            for i in range(2, min(len(logs), self._tes + 3)):
                second.append(abs(((logs[i] - logs[i - 1]) - (logs[i - 1] - logs[i - 2])) / 2.0))
            emean = float(np.mean(second)) if second else 0.0
            if emean > 1.0e-300:
                scaled = []
                for i in range(max(2, t - self._tr + 1), t + 1):
                    val = ((logs[i] - logs[i - 1]) - (logs[i - 1] - logs[i - 2])) / 2.0
                    scaled.append(abs(val / emean))
                dsd = float(np.sum(scaled))
                if dsd < self._beta:
                    k += 1
                    tstop = t + self._tr
                    counts[self._LABELS["strategy"]] += 1
                    state.payload["last_dsd"] = dsd
        return k, tstop

    def _space_for_generation(self, generation: int, strategy_k: int) -> str:
        if strategy_k <= 1:
            return "original" if (generation - 1) % 2 == 0 else "mapping"
        if strategy_k == 2:
            return "original" if (generation - 1) % (self._ratio + 1) < self._ratio else "mapping"
        return "original"

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        counts: dict[str, int],
    ) -> _CubicRBF:
        counts[self._LABELS["rbf"]] += 1
        return _CubicRBF.fit(X, y, lower, upper)

    def _choose_training_rows(self, archive_size: int, d: int) -> np.ndarray:
        size = min(archive_size, max(2, self._training_multiplier * int(d)))
        return self._rng.choice(archive_size, size=size, replace=False)

    def _de_best_1_bin(
        self,
        subpop: np.ndarray,
        xbest_sub: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        model: _CubicRBF,
        counts: dict[str, int],
        contribs: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        subpop = np.asarray(subpop, dtype=float).copy()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        n, d = subpop.shape
        pred = model.predict(subpop)
        if d == 0:
            bi = self._best_index(pred)
            return subpop, subpop[bi].copy(), float(pred[bi])
        for _ in range(self._Gm):
            for i in range(n):
                choices = [j for j in range(n) if j != i]
                r1, r2 = self._rng.choice(choices, size=2, replace=False)
                mutant = np.asarray(xbest_sub, dtype=float) + self._F * (subpop[r1] - subpop[r2])
                trial = subpop[i].copy()
                mask = self._rng.random(d) <= self._CR
                mask[int(self._rng.integers(0, d))] = True
                trial[mask] = mutant[mask]
                repaired = np.minimum(np.maximum(trial, lower), upper)
                if not np.allclose(repaired, trial):
                    counts[self._LABELS["repair"]] += 1
                trial = repaired
                f_trial = float(model.predict(trial)[0])
                counts[self._LABELS["de"]] += 1
                if self._is_better_value(f_trial, float(pred[i])):
                    contribs[self._LABELS["de"]] += self._signed_improvement(float(pred[i]), f_trial)
                    subpop[i] = trial
                    pred[i] = f_trial
        bi = self._best_index(pred)
        return subpop, subpop[bi].copy(), float(pred[bi])

    def _pca_context(self, pop_X: np.ndarray, archive_X: np.ndarray) -> dict[str, np.ndarray | int]:
        lo, hi = self._bounds()
        mean = np.mean(pop_X, axis=0)
        demeaned = pop_X - mean
        cov = np.cov(demeaned, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        y_pop = demeaned @ eigvecs
        y_archive = (archive_X - mean) @ eigvecs
        lu = np.vstack((hi, lo))
        y_lu = (lu - mean) @ eigvecs
        y_lower = np.minimum(y_lu[0], y_lu[1])
        y_upper = np.maximum(y_lu[0], y_lu[1])
        total = float(np.sum(eigvals))
        d1_alpha = int(np.sum(eigvals > self._alpha))
        if total > 0.0:
            d1_eta = int(np.searchsorted(np.cumsum(eigvals) / total, self._eta) + 1)
        else:
            d1_eta = 1
        d1 = max(1, min(pop_X.shape[1], max(d1_alpha, d1_eta)))
        return {
            "mean": mean,
            "vec": eigvecs,
            "y_pop": y_pop,
            "y_archive": y_archive,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "d1": d1,
        }

    def _mapping_columns(self, D: int, d: int, d1: int) -> np.ndarray:
        d1 = min(max(1, int(d1)), D)
        if d > d1:
            main = np.arange(d1, dtype=int)
            remaining = np.arange(d1, D, dtype=int)
            extra = self._rng.choice(remaining, size=d - d1, replace=False) if remaining.size and d > d1 else np.array([], dtype=int)
            cols = np.concatenate((main, extra))
        else:
            cols = np.arange(d, dtype=int)
        return self._rng.permutation(cols)

    def _original_subspace_step(
        self,
        pop_X: np.ndarray,
        archive_X: np.ndarray,
        archive_y: np.ndarray,
        best_exact: np.ndarray,
        counts: dict[str, int],
        contribs: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        lo, hi = self._bounds()
        D = pop_X.shape[1]
        d = int(self._rng.integers(1, min(D, self._maxd) + 1))
        cols = self._rng.permutation(self._rng.choice(D, size=d, replace=False))
        counts[self._LABELS["original"]] += 1
        rows = self._choose_training_rows(archive_X.shape[0], d)
        model = self._train_model(archive_X[rows][:, cols], archive_y[rows], lo[cols], hi[cols], counts)
        updated, best_sub, best_pred = self._de_best_1_bin(
            pop_X[:, cols], best_exact[cols], lo[cols], hi[cols], model, counts, contribs
        )
        pop_X = pop_X.copy()
        pop_X[:, cols] = updated
        candidate = best_exact.copy()
        candidate[cols] = best_sub
        candidate = np.asarray(self.problem.clip_position(candidate), dtype=float)
        return pop_X, {"candidate": candidate, "predicted": best_pred, "space": "original"}

    def _mapping_generation(
        self,
        pop_X: np.ndarray,
        archive_X: np.ndarray,
        archive_y: np.ndarray,
        best_exact: np.ndarray,
        counts: dict[str, int],
        contribs: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        lo, hi = self._bounds()
        D = pop_X.shape[1]
        ctx = self._pca_context(pop_X, archive_X)
        y_pop = np.asarray(ctx["y_pop"], dtype=float).copy()
        y_archive = np.asarray(ctx["y_archive"], dtype=float)
        y_lower = np.asarray(ctx["y_lower"], dtype=float)
        y_upper = np.asarray(ctx["y_upper"], dtype=float)
        vec = np.asarray(ctx["vec"], dtype=float)
        mean = np.asarray(ctx["mean"], dtype=float)
        y_best = (best_exact - mean) @ vec
        best_info: dict[str, Any] | None = None
        for _ in range(self._K):
            d = int(self._rng.integers(1, min(D, self._maxd) + 1))
            cols = self._mapping_columns(D, d, int(ctx["d1"]))
            counts[self._LABELS["mapping"]] += 1
            rows = self._choose_training_rows(archive_X.shape[0], len(cols))
            model = self._train_model(y_archive[rows][:, cols], archive_y[rows], y_lower[cols], y_upper[cols], counts)
            updated, best_sub, best_pred = self._de_best_1_bin(
                y_pop[:, cols], y_best[cols], y_lower[cols], y_upper[cols], model, counts, contribs
            )
            y_pop[:, cols] = updated
            y_candidate = y_best.copy()
            y_candidate[cols] = best_sub
            x_candidate = y_candidate @ np.linalg.inv(vec) + mean
            x_candidate = np.asarray(self.problem.clip_position(np.minimum(np.maximum(x_candidate, lo), hi)), dtype=float)
            if best_info is None or self._is_better_value(best_pred, float(best_info["predicted"])):
                best_info = {"candidate": x_candidate, "predicted": float(best_pred), "space": "mapping"}
        x_pop = y_pop @ np.linalg.inv(vec) + mean
        clipped = np.minimum(np.maximum(x_pop, lo), hi)
        if not np.allclose(clipped, x_pop):
            counts[self._LABELS["repair"]] += int(np.sum(np.any(np.abs(clipped - x_pop) > 1.0e-12, axis=1)))
        x_pop = np.asarray([self.problem.clip_position(row) for row in clipped], dtype=float)
        return x_pop, best_info or {"candidate": best_exact.copy(), "predicted": 0.0, "space": "mapping"}

    def step(self, state: EngineState) -> EngineState:
        population = np.asarray(state.payload["population"], dtype=float)
        pop_X = population[:, :-1].copy()
        pop_scores = population[:, -1].copy()
        archive_X = np.asarray(state.payload["archive_X"], dtype=float).copy()
        archive_y = np.asarray(state.payload["archive_y"], dtype=float).copy()
        best_exact = np.asarray(state.best_position, dtype=float).copy()
        old_best = float(state.best_fitness)
        counts = self._empty_counts()
        contribs = self._empty_contribs()
        strategy_k, tstop = self._maybe_switch_strategy(state, counts)
        generation = state.step + 1
        space = self._space_for_generation(generation, strategy_k)

        generation_candidates: list[dict[str, Any]] = []
        if space == "mapping":
            pop_X, info = self._mapping_generation(pop_X, archive_X, archive_y, best_exact, counts, contribs)
            generation_candidates.append(info)
        else:
            for _ in range(self._K):
                pop_X, info = self._original_subspace_step(pop_X, archive_X, archive_y, best_exact, counts, contribs)
                generation_candidates.append(info)

        best_info = generation_candidates[0]
        for info in generation_candidates[1:]:
            if self._is_better_value(float(info["predicted"]), float(best_info["predicted"])):
                best_info = info
        candidate = np.asarray(best_info["candidate"], dtype=float)
        candidate, candidate_fit = self._evaluate_one(candidate)
        state.evaluations += 1
        counts[self._LABELS["archive"]] += 1
        contribs[self._LABELS["archive"]] += max(0.0, self._signed_improvement(old_best, candidate_fit))
        archive_X = np.vstack((archive_X, candidate.reshape(1, -1)))
        archive_y = np.append(archive_y, candidate_fit)
        if self._is_better_value(candidate_fit, state.best_fitness):
            state.best_fitness = float(candidate_fit)
            state.best_position = candidate.tolist()
            best_exact = candidate.copy()
        else:
            best_exact = np.asarray(state.best_position, dtype=float)

        # Keep an exact anchor in the first row and the newest exact candidate in
        # the current population; other population scores remain surrogate-like.
        pop_scores = np.resize(pop_scores, pop_X.shape[0])
        worst = int(self._order(pop_scores)[-1]) if pop_scores.size else 0
        pop_X[worst] = candidate
        pop_scores[worst] = candidate_fit
        pop_X[0] = best_exact
        pop_scores[0] = float(state.best_fitness)
        population = np.hstack((pop_X, pop_scores.reshape(-1, 1)))

        history = list(state.payload.get("best_history", []))
        history.append(float(state.best_fitness))
        state.payload = {
            "population": population,
            "archive_X": archive_X,
            "archive_y": archive_y,
            "best_history": history,
            "strategy_k": strategy_k,
            "tstop": tstop,
            "last_space": space,
            "last_candidate_fitness": float(candidate_fit),
            "last_candidate_predicted": float(best_info["predicted"]),
            "last_candidate_source": str(best_info["space"]),
            "operator_counts": counts,
            "operator_contributions": contribs,
            "native_evomapx_operator_labels": True,
            "population_scores_are_mixed_exact_and_surrogate": True,
        }
        state.step += 1
        return state

    # ------------------------------------------------------------------
    # package interface and telemetry
    # ------------------------------------------------------------------
    def observe(self, state: EngineState) -> dict[str, Any]:
        pop = np.asarray(state.payload.get("population"), dtype=float)
        counts = dict(state.payload.get("operator_counts", {}))
        contribs = dict(state.payload.get("operator_contributions", {}))
        active_counts = {k: int(v) for k, v in counts.items() if int(v) != 0}
        active_contribs = {k: float(v) for k, v in contribs.items() if abs(float(v)) > 0.0 or k in active_counts}
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "archive_size": int(np.asarray(state.payload.get("archive_y", [])).size),
            "strategy": f"Str{int(state.payload.get('strategy_k', 1))}",
            "space": state.payload.get("last_space"),
            "mean_population_score": float(np.mean(pop[:, -1])) if pop.size else float("nan"),
            "operator_counts": active_counts,
            "operator_contributions": active_contribs,
            "evomapx_fidelity": "native",
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
            metadata={"archive_size": int(np.asarray(state.payload.get("archive_y", [])).size)},
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current" if i else "best_anchor",
                metadata={"score_type": "mixed_exact_surrogate"},
            )
            for i in range(pop.shape[0])
        ]

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
                "archive_size": int(np.asarray(state.payload.get("archive_y", [])).size),
                "strategy": f"Str{int(state.payload.get('strategy_k', 1))}",
                "elapsed_time": state.elapsed_time,
            },
        )
