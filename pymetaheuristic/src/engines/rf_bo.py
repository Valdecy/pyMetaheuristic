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
        "n_estimators": 64,
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_thresholds": 8,
    }

    def _make_estimator(self, random_state: int):
        return SimpleTreeEnsembleRegressor(
            kind="random_forest",
            n_estimators=int(self._params.get("n_estimators", 64)),
            max_depth=int(self._params.get("max_depth", 6)),
            min_samples_leaf=int(self._params.get("min_samples_leaf", 1)),
            max_features=self._params.get("max_features", "sqrt"),
            n_thresholds=int(self._params.get("n_thresholds", 8)),
            bootstrap=True,
            random_state=random_state,
        )
