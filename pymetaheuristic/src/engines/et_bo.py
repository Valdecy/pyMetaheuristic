"""pyMetaheuristic src — Extra-Trees Bayesian Optimization Engine"""
from __future__ import annotations

from ._surrogate_bo import SimpleTreeEnsembleRegressor, SurrogateBOEngine


class ETBOEngine(SurrogateBOEngine):
    """Extra-Trees Bayesian Optimization.

    Sequential model-based optimization using a lightweight Extra-Trees
    surrogate. Predictive uncertainty is estimated from the dispersion of the
    individual tree predictions.
    """

    algorithm_id = "et_bo"
    algorithm_name = "Extra-Trees Bayesian Optimization"
    family = "math"
    _REFERENCE = {
        "doi": "10.1007/s10994-006-6226-1",
        "title": "Extremely randomized trees",
        "authors": "Pierre Geurts, Damien Ernst, Louis Wehenkel",
        "year": 2006,
    }
    _ESTIMATOR_KIND = "extra_trees"
    _DEFAULTS = {
        **SurrogateBOEngine._DEFAULTS,
        "n_estimators": 48,
        "max_depth": 6,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_thresholds": 8,
    }

    def _make_estimator(self, random_state: int):
        return SimpleTreeEnsembleRegressor(
            kind="extra_trees",
            n_estimators=int(self._params.get("n_estimators", 48)),
            max_depth=int(self._params.get("max_depth", 6)),
            min_samples_leaf=int(self._params.get("min_samples_leaf", 1)),
            max_features=self._params.get("max_features", "sqrt"),
            n_thresholds=int(self._params.get("n_thresholds", 8)),
            bootstrap=False,
            random_state=random_state,
        )
