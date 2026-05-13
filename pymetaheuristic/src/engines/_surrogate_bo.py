"""Shared sequential model-based optimization machinery for BO engines.

This module intentionally depends only on NumPy/SciPy-compatible primitives.
It provides lightweight surrogates for Gaussian-process, forest, extra-trees,
and gradient-boosted-tree Bayesian optimization engines without requiring
scikit-learn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import BaseEngine, CandidateRecord, CapabilityProfile, EngineConfig, EngineState, OptimizationResult, ProblemSpec


# ---------------------------------------------------------------------------
# Lightweight estimators
# ---------------------------------------------------------------------------


class PureGaussianProcessRegressor:
    """Small Gaussian-process regressor with a fixed Matern-5/2 kernel.

    Inputs are expected in the unit hypercube.  The target is expected to be
    standardized by the caller.  Hyperparameter optimization is deliberately not
    implemented; this keeps the engine dependency-light and deterministic.
    """

    def __init__(self, *, alpha: float = 1.0e-10, length_scale: float | str = "auto") -> None:
        self.alpha = max(float(alpha), 1.0e-12)
        self.length_scale = length_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureGaussianProcessRegressor":
        self.X_ = np.asarray(X, dtype=float).copy()
        self.y_ = np.asarray(y, dtype=float).copy()
        n = self.X_.shape[0]
        self.length_scale_ = self._resolve_length_scale(self.X_)
        K = self._kernel(self.X_, self.X_) + self.alpha * np.eye(n)
        # Robust Cholesky with escalating jitter for near-duplicate designs.
        jitter = self.alpha
        for _ in range(8):
            try:
                self.L_ = np.linalg.cholesky(K + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            self.L_ = np.linalg.cholesky(K + (jitter + 1.0e-6) * np.eye(n))
        tmp = np.linalg.solve(self.L_, self.y_)
        self.alpha_vec_ = np.linalg.solve(self.L_.T, tmp)
        return self

    def _resolve_length_scale(self, X: np.ndarray) -> float:
        value = self.length_scale
        if isinstance(value, str) and value.lower() == "auto":
            if X.shape[0] < 2:
                return 0.5
            diff = X[:, None, :] - X[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            positive = dist[dist > 1.0e-12]
            if positive.size == 0:
                return 0.5
            return float(np.clip(np.median(positive), 0.05, 2.0))
        return float(np.clip(float(value), 1.0e-6, 1.0e6))

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        diff = (A[:, None, :] - B[None, :, :]) / self.length_scale_
        r = np.sqrt(np.sum(diff * diff, axis=2))
        s5r = math.sqrt(5.0) * r
        return (1.0 + s5r + (5.0 / 3.0) * r * r) * np.exp(-s5r)

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.asarray(X, dtype=float)
        K_trans = self._kernel(X, self.X_)
        mean = K_trans @ self.alpha_vec_
        if not return_std:
            return mean
        v = np.linalg.solve(self.L_, K_trans.T)
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-12)
        return mean, np.sqrt(var)


@dataclass
class _TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: Any | None = None
    right: Any | None = None


class SimpleRegressionTree:
    """Compact CART-style regression tree used by the lightweight ensembles."""

    def __init__(
        self,
        *,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        max_features: int | str | None = "sqrt",
        splitter: str = "best",
        n_thresholds: int = 8,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_depth = max(1, int(max_depth))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.max_features = max_features
        self.splitter = splitter
        self.n_thresholds = max(1, int(n_thresholds))
        self.rng = rng or np.random.default_rng()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRegressionTree":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        self.n_features_in_ = self.X_.shape[1]
        self.root_ = self._build(self.X_, self.y_, depth=0)
        return self

    def _feature_subset(self, dim: int) -> np.ndarray:
        mf = self.max_features
        if mf is None or mf == "all":
            count = dim
        elif mf == "sqrt":
            count = max(1, int(round(math.sqrt(dim))))
        elif mf == "log2":
            count = max(1, int(round(math.log2(max(2, dim)))))
        elif isinstance(mf, float):
            count = max(1, min(dim, int(round(mf * dim))))
        else:
            count = max(1, min(dim, int(mf)))
        return self.rng.choice(dim, size=count, replace=False)

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        if unique.size < 2:
            return np.empty(0, dtype=float)
        if self.splitter == "extra":
            lo, hi = float(unique[0]), float(unique[-1])
            if hi <= lo:
                return np.empty(0, dtype=float)
            return self.rng.uniform(lo, hi, size=self.n_thresholds)
        mids = 0.5 * (unique[:-1] + unique[1:])
        if mids.size <= self.n_thresholds:
            return mids
        idx = self.rng.choice(mids.size, size=self.n_thresholds, replace=False)
        return mids[idx]

    def _sse(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        centered = y - float(np.mean(y))
        return float(np.dot(centered, centered))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        n, dim = X.shape
        if n < 2 * self.min_samples_leaf:
            return None, None
        base = self._sse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        for feature in self._feature_subset(dim):
            thresholds = self._candidate_thresholds(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                loss = self._sse(y[left_mask]) + self._sse(y[~left_mask])
                gain = base - loss
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node_value = float(np.mean(y)) if y.size else 0.0
        if depth >= self.max_depth or X.shape[0] < 2 * self.min_samples_leaf or np.var(y) <= 1.0e-30:
            return _TreeNode(value=node_value)
        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return _TreeNode(value=node_value)
        left_mask = X[:, feature] <= threshold
        return _TreeNode(
            value=node_value,
            feature=feature,
            threshold=threshold,
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], depth + 1),
        )

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> float:
        while node.feature is not None and node.threshold is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return float(node.value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.asarray([self._predict_one(row, self.root_) for row in X], dtype=float)


class SimpleTreeEnsembleRegressor:
    """Small random-forest / extra-trees regressor for SMBO."""

    def __init__(
        self,
        *,
        kind: str,
        n_estimators: int = 64,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        max_features: int | str | None = "sqrt",
        n_thresholds: int = 8,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.kind = kind
        self.n_estimators = max(1, int(n_estimators))
        self.max_depth = max(1, int(max_depth))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.max_features = max_features
        self.n_thresholds = max(1, int(n_thresholds))
        self.bootstrap = bool(bootstrap)
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleTreeEnsembleRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        self.estimators_: list[SimpleRegressionTree] = []
        splitter = "extra" if self.kind == "extra_trees" else "best"
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)
            tree = SimpleRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                splitter=splitter,
                n_thresholds=self.n_thresholds,
                rng=np.random.default_rng(int(self.rng.integers(0, np.iinfo(np.int32).max))),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.asarray([tree.predict(X) for tree in self.estimators_], dtype=float)
        return np.mean(preds, axis=0)


class SimpleGBRTRegressor:
    """Minimal gradient-boosted regression trees for continuous targets."""

    def __init__(
        self,
        *,
        n_estimators: int = 64,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = max(1, int(n_estimators))
        self.learning_rate = float(learning_rate)
        self.max_depth = max(1, int(max_depth))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.rng = np.random.default_rng(random_state)
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGBRTRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = float(np.mean(y)) if y.size else 0.0
        pred = np.full(y.shape, self.init_, dtype=float)
        self.trees_: list[SimpleRegressionTree] = []
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = SimpleRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="all",
                splitter="best",
                n_thresholds=12,
                rng=np.random.default_rng(int(self.rng.integers(0, np.iinfo(np.int32).max))),
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            pred = pred + self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
        return pred


# ---------------------------------------------------------------------------
# BO engine
# ---------------------------------------------------------------------------


class SurrogateBOEngine(BaseEngine):
    """Base class for lightweight surrogate-based Bayesian optimizers.

    The engine follows the ask-and-tell spirit used by scikit-optimize:
    an initial random design is evaluated, a surrogate model is fitted to the
    accumulated observations, and each macro-step proposes one new point by
    optimizing an acquisition function over a randomized candidate set.
    """

    family = "surrogate"
    capabilities = CapabilityProfile(
        has_population=False,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=False,
    )
    _DEFAULTS = dict(
        n_initial_points=None,
        candidate_pool_size=1024,
        acquisition="ei",
        xi=0.01,
        kappa=1.96,
        local_search_fraction=0.35,
        local_search_scale=0.15,
        avoid_duplicate_tolerance=1.0e-12,
    )
    _ESTIMATOR_KIND = "base"

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        params = {**self._DEFAULTS, **config.params}
        self._params = params
        dim = int(problem.dimension)
        requested_initial = params.get("n_initial_points")
        if requested_initial is None:
            requested_initial = max(5, 2 * dim + 1)
        if config.max_evaluations is not None:
            requested_initial = min(int(requested_initial), max(1, int(config.max_evaluations)))
        self._n_initial_points = max(1, int(requested_initial))
        self._candidate_pool_size = max(16, int(params.get("candidate_pool_size", 1024)))
        self._acquisition = str(params.get("acquisition", "ei")).lower()
        self._xi = float(params.get("xi", 0.01))
        self._kappa = float(params.get("kappa", 1.96))
        self._local_fraction = float(params.get("local_search_fraction", 0.35))
        self._local_scale = float(params.get("local_search_scale", 0.15))
        self._duplicate_tol = float(params.get("avoid_duplicate_tolerance", 1.0e-12))
        self._rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            np.random.seed(config.seed)
        self._validate_parameters()

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span > 0.0, span, 1.0)

    def _validate_parameters(self) -> None:
        if self.problem.dimension < 1:
            raise ValueError(f"{self.algorithm_id} requires at least one decision variable.")
        if np.any(self._hi < self._lo):
            raise ValueError(f"{self.algorithm_id} requires max_values >= min_values for every variable.")
        if self._acquisition not in {"ei", "lcb", "pi"}:
            raise ValueError(f"{self.algorithm_id} acquisition must be one of: 'ei', 'lcb', 'pi'.")
        if self._xi < 0.0:
            raise ValueError(f"{self.algorithm_id} xi must be non-negative.")
        if self._kappa < 0.0:
            raise ValueError(f"{self.algorithm_id} kappa must be non-negative.")
        if not 0.0 <= self._local_fraction <= 1.0:
            raise ValueError(f"{self.algorithm_id} local_search_fraction must be in [0, 1].")
        if self._local_scale < 0.0:
            raise ValueError(f"{self.algorithm_id} local_search_scale must be non-negative.")
        if self._duplicate_tol < 0.0:
            raise ValueError(f"{self.algorithm_id} avoid_duplicate_tolerance must be non-negative.")

    def _unit_to_position(self, unit: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
        return self._lo + unit * self._span

    def _position_to_unit(self, position: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(position, dtype=float) - self._lo) / self._span, 0.0, 1.0)

    def _fitness_to_loss(self, fitness: np.ndarray | float) -> np.ndarray | float:
        if self.problem.objective == "min":
            return fitness
        return -np.asarray(fitness, dtype=float)

    def _sample_unit(self, n: int) -> np.ndarray:
        return self._rng.random((int(n), self.problem.dimension))

    def _evaluate_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([float(self.problem.evaluate(row.copy())) for row in positions], dtype=float)

    def _best_index(self, fitness: np.ndarray) -> int:
        return int(np.argmin(fitness) if self.problem.objective == "min" else np.argmax(fitness))

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _make_estimator(self, random_state: int):
        raise NotImplementedError

    def _fit_estimator(self, X_unit: np.ndarray, loss: np.ndarray):
        estimator = self._make_estimator(self._next_seed())
        y_mean = float(np.mean(loss))
        y_std = float(np.std(loss))
        if y_std <= 1.0e-30:
            y_std = 1.0
        y_scaled = (loss - y_mean) / y_std
        estimator.fit(X_unit, y_scaled)
        return estimator, y_mean, y_std, y_scaled

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, np.iinfo(np.int32).max))

    def _predict_scaled(self, estimator, X_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean/std in the standardized loss scale."""
        try:
            mean, std = estimator.predict(X_unit, return_std=True)
            return np.asarray(mean, dtype=float), np.maximum(np.asarray(std, dtype=float), 1.0e-12)
        except TypeError:
            pass
        mean = np.asarray(estimator.predict(X_unit), dtype=float)
        std = self._model_uncertainty(estimator, X_unit, mean)
        return mean, np.maximum(std, 1.0e-12)

    def _model_uncertainty(self, estimator, X_unit: np.ndarray, mean: np.ndarray) -> np.ndarray:
        if hasattr(estimator, "estimators_"):
            estimators = getattr(estimator, "estimators_")
            try:
                tree_preds = np.asarray([est.predict(X_unit) for est in estimators], dtype=float)
                if tree_preds.ndim >= 2:
                    return np.std(tree_preds.reshape(tree_preds.shape[0], -1), axis=0)
            except Exception:
                pass
        return self._distance_uncertainty(X_unit)

    def _distance_uncertainty(self, X_unit: np.ndarray) -> np.ndarray:
        observed = getattr(self, "_observed_unit_for_uncertainty", None)
        if observed is None or len(observed) == 0:
            return np.ones(X_unit.shape[0], dtype=float)
        diff = X_unit[:, None, :] - observed[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        nearest = np.min(dist, axis=1)
        scale = math.sqrt(max(1, self.problem.dimension))
        return np.maximum(nearest / scale, 1.0e-12)

    def _normal_pdf(self, z: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def _normal_cdf(self, z: np.ndarray) -> np.ndarray:
        try:
            from scipy.special import erf
            return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
        except Exception:
            return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    def _acquisition_values(self, mean: np.ndarray, std: np.ndarray, best_loss_scaled: float) -> np.ndarray:
        std = np.maximum(std, 1.0e-12)
        if self._acquisition == "lcb":
            return -(mean - self._kappa * std)
        improvement = best_loss_scaled - mean - self._xi
        z = improvement / std
        if self._acquisition == "pi":
            return self._normal_cdf(z)
        return improvement * self._normal_cdf(z) + std * self._normal_pdf(z)

    def _candidate_pool(self, X_unit: np.ndarray, best_unit: np.ndarray) -> np.ndarray:
        n_local = int(round(self._candidate_pool_size * self._local_fraction))
        n_global = self._candidate_pool_size - n_local
        parts = []
        if n_global > 0:
            parts.append(self._sample_unit(n_global))
        if n_local > 0:
            local = best_unit + self._rng.normal(0.0, self._local_scale, size=(n_local, self.problem.dimension))
            parts.append(np.clip(local, 0.0, 1.0))
        pool = np.vstack(parts) if parts else self._sample_unit(self._candidate_pool_size)
        if self._duplicate_tol > 0.0 and X_unit.size:
            diff = pool[:, None, :] - X_unit[None, :, :]
            too_close = np.min(np.sqrt(np.sum(diff * diff, axis=2)), axis=1) <= self._duplicate_tol
            if np.all(too_close):
                return self._sample_unit(self._candidate_pool_size)
            pool = pool[~too_close]
        return pool

    def initialize(self) -> EngineState:
        X_unit = self._sample_unit(self._n_initial_points)
        X = self._unit_to_position(X_unit)
        y = self._evaluate_positions(X)
        loss = np.asarray(self._fitness_to_loss(y), dtype=float)
        best_idx = self._best_index(y)
        return EngineState(
            step=0,
            evaluations=int(X.shape[0]),
            best_position=X[best_idx].tolist(),
            best_fitness=float(y[best_idx]),
            initialized=True,
            payload={
                "X": X,
                "X_unit": X_unit,
                "fitness": y,
                "loss": loss,
                "last_candidate": X[best_idx].copy(),
                "last_acquisition": None,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        X = np.asarray(state.payload["X"], dtype=float)
        X_unit = np.asarray(state.payload["X_unit"], dtype=float)
        fitness = np.asarray(state.payload["fitness"], dtype=float)
        loss = np.asarray(state.payload["loss"], dtype=float)
        self._observed_unit_for_uncertainty = X_unit
        estimator, _, _, y_scaled = self._fit_estimator(X_unit, loss)
        best_idx = int(np.argmin(loss))
        best_unit = X_unit[best_idx]
        pool = self._candidate_pool(X_unit, best_unit)
        mean, std = self._predict_scaled(estimator, pool)
        acquisition = self._acquisition_values(mean, std, float(np.min(y_scaled)))
        chosen_idx = int(np.argmax(acquisition))
        x_unit_new = pool[chosen_idx]
        x_new = self._unit_to_position(x_unit_new)
        y_new = float(self.problem.evaluate(x_new.copy()))
        loss_new = float(self._fitness_to_loss(y_new))

        X = np.vstack((X, x_new.reshape(1, -1)))
        X_unit = np.vstack((X_unit, x_unit_new.reshape(1, -1)))
        fitness = np.append(fitness, y_new)
        loss = np.append(loss, loss_new)

        state.step += 1
        state.evaluations += 1
        if state.best_fitness is None or self._is_better(y_new, state.best_fitness):
            state.best_fitness = float(y_new)
            state.best_position = x_new.tolist()
        state.payload = {
            "X": X,
            "X_unit": X_unit,
            "fitness": fitness,
            "loss": loss,
            "last_candidate": x_new.copy(),
            "last_acquisition": float(acquisition[chosen_idx]),
        }
        return state

    def observe(self, state: EngineState) -> dict:
        fitness = np.asarray(state.payload["fitness"], dtype=float)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "observations": int(fitness.size),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "last_acquisition": state.payload.get("last_acquisition"),
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
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
                "surrogate": self._ESTIMATOR_KIND,
                "observations": int(np.asarray(state.payload["fitness"]).size),
                "acquisition": self._acquisition,
                "elapsed_time": state.elapsed_time,
            },
        )
