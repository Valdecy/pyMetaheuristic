"""pyMetaheuristic src — Random Forest Bayesian Optimization Engine"""
from __future__ import annotations

from ._surrogate_bo import SimpleTreeEnsembleRegressor, SurrogateBOEngine


class RFBOEngine(SurrogateBOEngine):
    """Random Forest Bayesian Optimization.

    Sequential model-based optimization using a lightweight random-forest
    surrogate. Predictive uncertainty is estimated from individual-tree
    dispersion.
    """

    algorithm_id = "rf_bo"
    algorithm_name = "Random Forest Bayesian Optimization"
    family = "math"
    _REFERENCE = {
        "doi": "10.1023/A:1010933404324",
        "title": "Random Forests",
        "authors": "Leo Breiman",
        "year": 2001,
    }
    _ESTIMATOR_KIND = "random_forest"
    _DEFAULTS = {
        **SurrogateBOEngine._DEFAULTS,
        "n_initial_points": 40,
        "candidate_pool_size": 384,
        "avoid_duplicate_tolerance": 0.0,
        "surrogate_train_size": 256,
        "local_search_fraction": 0.45,
        "local_search_scale": 0.22,
        "batch_size": 8,
        "polish_points": 4,
        "polish_scale": 0.10,
        "n_estimators": 8,
        "max_depth": 6,
        "max_samples": 128,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_thresholds": 8,
    }

    def _make_estimator(self, random_state: int):
        return SimpleTreeEnsembleRegressor(
            kind="random_forest",
            n_estimators=int(self._params.get("n_estimators", 8)),
            max_depth=int(self._params.get("max_depth", 6)),
            min_samples_leaf=int(self._params.get("min_samples_leaf", 1)),
            max_features=self._params.get("max_features", "sqrt"),
            n_thresholds=int(self._params.get("n_thresholds", 8)),
            bootstrap=True,
            max_samples=self._params.get("max_samples", 128),
            random_state=random_state,
        )
