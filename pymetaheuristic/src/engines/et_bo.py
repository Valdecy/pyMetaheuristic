"""pyMetaheuristic src - Extra-Trees Bayesian Optimization Engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._surrogate_bo import SurrogateBOEngine
from .protocol import CapabilityProfile, EngineState


@dataclass
class _ETNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_ETNode | None" = None
    right: "_ETNode | None" = None


class _PaperExtraTreeRegressor:
    """Regression tree using the numerical Extra-Trees split rule of Geurts et al.

    At each node, K non-constant attributes are selected without replacement;
    one random cut-point is drawn uniformly for each selected attribute; the
    candidate with maximal variance-reduction score is used. The tree is not
    pruned and has no depth limit unless an explicit safety limit is requested.
    """

    def __init__(
        self,
        *,
        k: int | str | None = None,
        n_min: int = 5,
        max_depth: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.k = k
        self.n_min = max(1, int(n_min))
        self.max_depth = None if max_depth is None else max(1, int(max_depth))
        self.rng = rng or np.random.default_rng()
        self.stats_: dict[str, int] = {
            "nodes": 0,
            "leaves": 0,
            "features_screened": 0,
            "candidate_splits": 0,
            "accepted_splits": 0,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_PaperExtraTreeRegressor":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        if self.X_.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.y_.ndim != 1 or self.y_.shape[0] != self.X_.shape[0]:
            raise ValueError("y must be a 1D array with len(y) == X.shape[0].")
        self.n_features_in_ = int(self.X_.shape[1])
        self.stats_ = {key: 0 for key in self.stats_}
        self.root_ = self._build(np.arange(self.X_.shape[0]), depth=0)
        return self

    def _resolve_k(self, n_nonconstant: int) -> int:
        if n_nonconstant <= 0:
            return 0
        raw = self.k
        if raw is None:
            # The paper's regression default is K = n.  In a node with fewer
            # non-constant attributes, all available attributes are screened.
            return int(n_nonconstant)
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in {"regression", "regression_default", "all", "n", "none"}:
                return int(n_nonconstant)
            if text in {"classification", "classification_default", "sqrt"}:
                return max(1, min(n_nonconstant, int(round(math.sqrt(max(1, self.n_features_in_))))))
            if text == "log2":
                return max(1, min(n_nonconstant, int(round(math.log2(max(2, self.n_features_in_))))))
            raw = int(text)
        if isinstance(raw, float) and 0.0 < raw <= 1.0:
            return max(1, min(n_nonconstant, int(round(raw * self.n_features_in_))))
        return max(1, min(n_nonconstant, int(raw)))

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if y.size <= 1:
            return 0.0
        centered = y - float(np.mean(y))
        return float(np.dot(centered, centered))

    def _nonconstant_features(self, X_node: np.ndarray) -> np.ndarray:
        if X_node.shape[0] <= 1:
            return np.empty(0, dtype=int)
        lo = np.min(X_node, axis=0)
        hi = np.max(X_node, axis=0)
        return np.flatnonzero(hi > lo)

    def _stop(self, X_node: np.ndarray, y_node: np.ndarray, depth: int) -> bool:
        if X_node.shape[0] < self.n_min:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if np.all(np.max(X_node, axis=0) <= np.min(X_node, axis=0)):
            return True
        if np.max(y_node) <= np.min(y_node):
            return True
        return False

    def _split_node(self, X_node: np.ndarray, y_node: np.ndarray) -> tuple[int | None, float | None]:
        nonconstant = self._nonconstant_features(X_node)
        k = self._resolve_k(int(nonconstant.size))
        if k <= 0:
            return None, None
        features = self.rng.choice(nonconstant, size=k, replace=False)
        base_sse = self._sse(y_node)
        best_gain = -math.inf
        best_feature: int | None = None
        best_threshold: float | None = None
        self.stats_["features_screened"] += int(k)

        for feature in features:
            values = X_node[:, int(feature)]
            lo = float(np.min(values))
            hi = float(np.max(values))
            if not hi > lo:
                continue
            threshold = float(self.rng.uniform(lo, hi))
            left_mask = values < threshold
            n_left = int(np.sum(left_mask))
            n_right = int(values.size - n_left)
            # Empty children are a zero-probability event for continuous data,
            # but can occur numerically or with heavily discrete inputs.
            if n_left == 0 or n_right == 0:
                continue
            self.stats_["candidate_splits"] += 1
            loss = self._sse(y_node[left_mask]) + self._sse(y_node[~left_mask])
            gain = base_sse - loss
            if gain > best_gain:
                best_gain = float(gain)
                best_feature = int(feature)
                best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return None, None
        return best_feature, best_threshold

    def _build(self, indices: np.ndarray, depth: int) -> _ETNode:
        X_node = self.X_[indices]
        y_node = self.y_[indices]
        node = _ETNode(value=float(np.mean(y_node)) if y_node.size else 0.0)
        self.stats_["nodes"] += 1
        if self._stop(X_node, y_node, depth):
            self.stats_["leaves"] += 1
            return node

        feature, threshold = self._split_node(X_node, y_node)
        if feature is None or threshold is None:
            self.stats_["leaves"] += 1
            return node

        values = X_node[:, feature]
        left_local = values < threshold
        if not np.any(left_local) or np.all(left_local):
            self.stats_["leaves"] += 1
            return node

        self.stats_["accepted_splits"] += 1
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(indices[left_local], depth + 1)
        node.right = self._build(indices[~left_local], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty(X.shape[0], dtype=float)

        def _fill(row_indices: np.ndarray, node: _ETNode) -> None:
            if row_indices.size == 0:
                return
            if node.feature is None or node.threshold is None or node.left is None or node.right is None:
                out[row_indices] = float(node.value)
                return
            mask = X[row_indices, node.feature] < node.threshold
            _fill(row_indices[mask], node.left)
            _fill(row_indices[~mask], node.right)

        _fill(np.arange(X.shape[0]), self.root_)
        return out


class PaperExtraTreesRegressor:
    """Paper-faithful numerical Extra-Trees regressor for the BO surrogate."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        k: int | str | None = None,
        n_min: int = 5,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = max(1, int(n_estimators))
        self.k = k
        self.n_min = max(1, int(n_min))
        self.max_depth = None if max_depth is None else max(1, int(max_depth))
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PaperExtraTreesRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.estimators_: list[_PaperExtraTreeRegressor] = []
        totals = {
            "nodes": 0,
            "leaves": 0,
            "features_screened": 0,
            "candidate_splits": 0,
            "accepted_splits": 0,
        }
        for _ in range(self.n_estimators):
            tree = _PaperExtraTreeRegressor(
                k=self.k,
                n_min=self.n_min,
                max_depth=self.max_depth,
                rng=np.random.default_rng(int(self.rng.integers(0, np.iinfo(np.int32).max))),
            )
            # Extra-Trees grows each tree from the full original learning sample;
            # no bootstrap replica is drawn.
            tree.fit(X, y)
            self.estimators_.append(tree)
            for key, value in tree.stats_.items():
                totals[key] += int(value)
        self.fit_stats_ = totals | {"trees": int(self.n_estimators), "n_min": int(self.n_min)}
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        preds = np.asarray([tree.predict(X) for tree in self.estimators_], dtype=float)
        mean = np.mean(preds, axis=0)
        if not return_std:
            return mean
        std = np.std(preds, axis=0)
        return mean, np.maximum(std, 1.0e-12)


class ETBOEngine(SurrogateBOEngine):
    """Extra-Trees Bayesian Optimization.

    This optimizer keeps the package's sequential model-based optimization
    interface, but replaces the surrogate with a numerical Extra-Trees regressor
    that follows the split mechanics and defaults of Geurts, Ernst, and
    Wehenkel's Extremely Randomized Trees for regression.
    """

    algorithm_id = "et_bo"
    algorithm_name = "Extra-Trees Bayesian Optimization"
    family = "surrogate"
    capabilities = CapabilityProfile(
        has_population=False,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=False,
    )
    _REFERENCE = {
        "doi": "10.1007/s10994-006-6226-1",
        "title": "Extremely randomized trees",
        "authors": "Pierre Geurts, Damien Ernst, Louis Wehenkel",
        "year": 2006,
    }
    _ESTIMATOR_KIND = "paper_extra_trees"
    _OPERATOR_LABELS = (
        "et_bo.extra_trees_surrogate_fit",
        "et_bo.random_cutpoint_screening",
        "et_bo.acquisition_search",
        "et_bo.candidate_evaluation",
        "et_bo.incumbent_update",
    )
    _DEFAULTS = {
        **SurrogateBOEngine._DEFAULTS,
        "n_estimators": 100,
        "k": "regression_default",
        "n_min": 5,
        "max_depth": None,
    }

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._last_fit_stats: dict[str, int] = {}
        self._last_operator_counts = {label: 0 for label in self._OPERATOR_LABELS}
        self._last_operator_contributions = {label: 0.0 for label in self._OPERATOR_LABELS}

    def _resolve_k_param(self) -> int | str | None:
        if "K" in self._params:
            return self._params.get("K")
        if "k" in self._params:
            return self._params.get("k")
        # Backward-compatible alias from the previous generic tree wrapper.
        legacy = self._params.get("max_features", "regression_default")
        if legacy in {None, "all"}:
            return "regression_default"
        return legacy

    def _make_estimator(self, random_state: int):
        max_depth = self._params.get("max_depth", None)
        return PaperExtraTreesRegressor(
            n_estimators=int(self._params.get("n_estimators", 100)),
            k=self._resolve_k_param(),
            n_min=int(self._params.get("n_min", self._params.get("nmin", 5))),
            max_depth=None if max_depth in {None, "none", "None"} else int(max_depth),
            random_state=random_state,
        )

    def _fit_estimator(self, X_unit: np.ndarray, loss: np.ndarray):
        estimator, y_mean, y_std, y_scaled = super()._fit_estimator(X_unit, loss)
        self._last_fit_stats = dict(getattr(estimator, "fit_stats_", {}) or {})
        return estimator, y_mean, y_std, y_scaled

    def step(self, state: EngineState) -> EngineState:
        old_best = None if state.best_fitness is None else float(state.best_fitness)
        old_evals = int(state.evaluations)
        state = super().step(state)
        new_best = None if state.best_fitness is None else float(state.best_fitness)
        if old_best is None or new_best is None:
            positive_delta = 0.0
        elif self.problem.objective == "min":
            positive_delta = max(0.0, old_best - new_best)
        else:
            positive_delta = max(0.0, new_best - old_best)

        fit_stats = dict(self._last_fit_stats)
        counts = {
            "et_bo.extra_trees_surrogate_fit": int(fit_stats.get("trees", self._params.get("n_estimators", 100))),
            "et_bo.random_cutpoint_screening": int(fit_stats.get("candidate_splits", 0)),
            "et_bo.acquisition_search": int(self._candidate_pool_size),
            "et_bo.candidate_evaluation": int(max(0, state.evaluations - old_evals)),
            "et_bo.incumbent_update": 1,
        }
        contributions = {
            "et_bo.extra_trees_surrogate_fit": 0.0,
            "et_bo.random_cutpoint_screening": 0.0,
            "et_bo.acquisition_search": 0.40 * positive_delta,
            "et_bo.candidate_evaluation": 0.30 * positive_delta,
            "et_bo.incumbent_update": 0.30 * positive_delta,
        }
        self._last_operator_counts = counts
        self._last_operator_contributions = contributions
        state.payload["last_fit_stats"] = fit_stats
        state.payload["operator_counts"] = counts
        state.payload["operator_contributions"] = contributions
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        obs = super().observe(state)
        fit_stats = dict(state.payload.get("last_fit_stats", self._last_fit_stats) or {})
        counts = dict(state.payload.get("operator_counts", self._last_operator_counts) or {})
        contributions = dict(state.payload.get("operator_contributions", self._last_operator_contributions) or {})
        obs.update(
            {
                "surrogate_trees": int(fit_stats.get("trees", self._params.get("n_estimators", 100))),
                "surrogate_nodes": int(fit_stats.get("nodes", 0)),
                "surrogate_leaves": int(fit_stats.get("leaves", 0)),
                "random_cutpoint_candidates": int(fit_stats.get("candidate_splits", 0)),
                "accepted_tree_splits": int(fit_stats.get("accepted_splits", 0)),
                "operator_counts": counts,
                "operator_contributions": contributions,
                "evomapx_fidelity": "native_paper_extra_trees_surrogate",
            }
        )
        return obs

    def finalize(self, state: EngineState):
        result = super().finalize(state)
        result.metadata.update(
            {
                "math": self._ESTIMATOR_KIND,
                "extra_trees_reference_defaults": {
                    "M": int(self._params.get("n_estimators", 100)),
                    "K": self._resolve_k_param(),
                    "n_min": int(self._params.get("n_min", self._params.get("nmin", 5))),
                    "bootstrap": False,
                    "max_depth": self._params.get("max_depth", None),
                },
                "last_fit_stats": dict(state.payload.get("last_fit_stats", {}) or {}),
            }
        )
        return result
