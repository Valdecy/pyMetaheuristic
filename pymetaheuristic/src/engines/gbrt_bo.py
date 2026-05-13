"""pyMetaheuristic src — Gradient-Boosted Trees Bayesian Optimization Engine"""
from __future__ import annotations

from ._surrogate_bo import SimpleGBRTRegressor, SurrogateBOEngine


class GBRTBOEngine(SurrogateBOEngine):
    """Gradient-Boosted Regression Trees Bayesian Optimization.

    Sequential model-based optimization using a lightweight gradient-boosted
    tree surrogate. Since GBRT does not expose a native predictive variance,
    this engine uses a distance-to-observed-design uncertainty proxy.
    """

    algorithm_id = "gbrt_bo"
    algorithm_name = "Gradient-Boosted Regression Trees Bayesian Optimization"
    family = "math"
    _REFERENCE = {
        "doi": "10.1214/aos/1013203451",
        "title": "Greedy Function Approximation: A Gradient Boosting Machine",
        "authors": "Jerome H. Friedman",
        "year": 2001,
    }
    _ESTIMATOR_KIND = "gradient_boosted_regression_trees"
    _DEFAULTS = {
        **SurrogateBOEngine._DEFAULTS,
        "n_estimators": 64,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 1,
    }

    def _make_estimator(self, random_state: int):
        return SimpleGBRTRegressor(
            n_estimators=int(self._params.get("n_estimators", 64)),
            learning_rate=float(self._params.get("learning_rate", 0.05)),
            max_depth=int(self._params.get("max_depth", 3)),
            min_samples_leaf=int(self._params.get("min_samples_leaf", 1)),
            random_state=random_state,
        )
