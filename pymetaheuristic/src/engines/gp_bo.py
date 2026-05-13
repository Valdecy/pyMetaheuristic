"""pyMetaheuristic src — Gaussian Process Bayesian Optimization Engine"""
from __future__ import annotations

from ._surrogate_bo import PureGaussianProcessRegressor, SurrogateBOEngine


class GPBOEngine(SurrogateBOEngine):
    """Gaussian Process Bayesian Optimization.

    A compact scikit-optimize-style sequential model-based optimizer using a
    dependency-light Gaussian-process surrogate and EI/PI/LCB acquisition over a
    randomized candidate set.
    """

    algorithm_id = "gp_bo"
    algorithm_name = "Gaussian Process Bayesian Optimization"
    family = "math"
    _REFERENCE = {
        "doi": "10.1023/A:1008306431147",
        "title": "Efficient Global Optimization of Expensive Black-Box Functions",
        "authors": "Donald R. Jones, Matthias Schonlau, William J. Welch",
        "year": 1998,
    }
    _ESTIMATOR_KIND = "gaussian_process"
    _DEFAULTS = {
        **SurrogateBOEngine._DEFAULTS,
        "alpha": 1.0e-10,
        "length_scale": "auto",
    }

    def _make_estimator(self, random_state: int):
        return PureGaussianProcessRegressor(
            alpha=float(self._params.get("alpha", 1.0e-10)),
            length_scale=self._params.get("length_scale", "auto"),
        )
