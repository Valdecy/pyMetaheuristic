from __future__ import annotations

import numpy as np

__all__ = [
    "limit",
    "limit_inverse",
    "wang",
    "rand",
    "reflect",
    "AVAILABLE_REPAIR_STRATEGIES",
    "get_repair_function",
]


def _as_arrays(lower, upper):
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def limit(x, lower, upper, **_kwargs):
    x = np.asarray(x, dtype=float).copy()
    lo, hi = _as_arrays(lower, upper)
    return np.clip(x, lo, hi)


def limit_inverse(x, lower, upper, **_kwargs):
    x = np.asarray(x, dtype=float).copy()
    lo, hi = _as_arrays(lower, upper)
    ir = np.where(x < lo)
    x[ir] = hi[ir]
    ir = np.where(x > hi)
    x[ir] = lo[ir]
    return x


def wang(x, lower, upper, **_kwargs):
    x = np.asarray(x, dtype=float).copy()
    lo, hi = _as_arrays(lower, upper)
    ir = np.where(x < lo)
    x[ir] = np.minimum(hi[ir], 2 * lo[ir] - x[ir])
    ir = np.where(x > hi)
    x[ir] = np.maximum(lo[ir], 2 * hi[ir] - x[ir])
    return x


def rand(x, lower, upper, rng=None, **_kwargs):
    x = np.asarray(x, dtype=float).copy()
    lo, hi = _as_arrays(lower, upper)
    if rng is None:
        rng = np.random.default_rng()
    ir = np.where(x < lo)
    if ir[0].size:
        x[ir] = rng.uniform(lo[ir], hi[ir])
    ir = np.where(x > hi)
    if ir[0].size:
        x[ir] = rng.uniform(lo[ir], hi[ir])
    return x


def reflect(x, lower, upper, **_kwargs):
    x = np.asarray(x, dtype=float).copy()
    lo, hi = _as_arrays(lower, upper)
    span = hi - lo
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    ir = np.where(x > hi)
    x[ir] = lo[ir] + np.mod(x[ir] - hi[ir], span[ir])
    ir = np.where(x < lo)
    x[ir] = hi[ir] - np.mod(lo[ir] - x[ir], span[ir])
    return x


_REPAIR_REGISTRY = {
    "clip": limit,
    "limit": limit,
    "limit_inverse": limit_inverse,
    "wrap": limit_inverse,
    "wang": wang,
    "rand": rand,
    "random": rand,
    "reflect": reflect,
}

AVAILABLE_REPAIR_STRATEGIES = sorted(_REPAIR_REGISTRY.keys())


def get_repair_function(name: str):
    key = str(name).strip().lower()
    if key not in _REPAIR_REGISTRY:
        raise KeyError(f"Unknown repair strategy: {name}. Available: {', '.join(AVAILABLE_REPAIR_STRATEGIES)}")
    return _REPAIR_REGISTRY[key]
