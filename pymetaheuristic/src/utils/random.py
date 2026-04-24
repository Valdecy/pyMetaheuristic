from __future__ import annotations

import math
import numpy as np

__all__ = ["levy_flight"]


def levy_flight(rng=None, alpha: float = 0.01, beta: float = 1.5, size=None):
    """Sample Lévy-flight steps using the Mantegna algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0.0, sigma, size)
    v = rng.normal(0.0, 1.0, size)
    return alpha * u / (np.abs(v) ** (1 / beta))
