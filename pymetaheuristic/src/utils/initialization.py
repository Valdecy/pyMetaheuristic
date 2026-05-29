from __future__ import annotations

from typing import Callable
import numpy as np

from .chaotic import chaotic_population

__all__ = [
    "AVAILABLE_INIT_STRATEGIES",
    "uniform_population",
    "lhs_population",
    "obl_population",
    "sobol_population",
    "chaotic_init_function",
    "get_init_function",
]


def _as_rng(rng=None):
    """Return an object exposing NumPy Generator-like random methods."""
    return np.random.default_rng() if rng is None else rng


def _bounds(min_values, max_values, dimension: int | None = None):
    lo = np.asarray(min_values, dtype=float).reshape(-1)
    hi = np.asarray(max_values, dtype=float).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(f"min_values and max_values must have the same shape; got {lo.shape} and {hi.shape}")
    if dimension is not None and lo.size != int(dimension):
        raise ValueError(f"dimension={dimension} but bounds have length {lo.size}")
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("Bounds must be finite for initialization presets.")
    if np.any(hi < lo):
        raise ValueError("Every max_values entry must be >= the corresponding min_values entry.")
    return lo, hi


def _problem_bounds(problem):
    if hasattr(problem, "min_values") and hasattr(problem, "max_values"):
        lo = problem.min_values
        hi = problem.max_values
    elif hasattr(problem, "lower") and hasattr(problem, "upper"):
        lo = problem.lower
        hi = problem.upper
    else:
        raise AttributeError("problem must expose min_values/max_values or lower/upper")
    dim = int(getattr(problem, "dimension", len(lo)))
    lo, hi = _bounds(lo, hi, dim)
    return dim, lo, hi


def _validate_shape(pop_size: int, dimension: int) -> tuple[int, int]:
    pop_size = int(pop_size)
    dimension = int(dimension)
    if pop_size <= 0:
        raise ValueError("pop_size must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    return pop_size, dimension


def uniform_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    """Uniform random population in the hyper-rectangle [min_values, max_values]."""
    rng = _as_rng(rng)
    pop_size, dimension = _validate_shape(pop_size, dimension)
    lo, hi = _bounds(min_values, max_values, dimension)
    return rng.uniform(lo, hi, size=(pop_size, dimension))


def lhs_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS) population.

    Each variable is stratified into ``pop_size`` equiprobable intervals and
    sampled once per interval; each dimension is independently permuted. This
    gives better one-dimensional space filling than independent uniform draws
    while preserving a simple NumPy-only implementation.
    """
    rng = _as_rng(rng)
    pop_size, dimension = _validate_shape(pop_size, dimension)
    lo, hi = _bounds(min_values, max_values, dimension)
    result = np.empty((pop_size, dimension), dtype=float)
    cut = np.linspace(0.0, 1.0, pop_size + 1)
    for j in range(dimension):
        # One point per stratum, then independently shuffle the strata for
        # this coordinate. This is the standard random LHS construction.
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        result[:, j] = lo[j] + u * (hi[j] - lo[j])
    return result


def obl_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    """Opposition-based population built from uniform points and their opposites."""
    rng = _as_rng(rng)
    pop_size, dimension = _validate_shape(pop_size, dimension)
    lo, hi = _bounds(min_values, max_values, dimension)
    base = rng.uniform(lo, hi, size=(pop_size, dimension))
    opp = lo + hi - base
    combined = np.vstack((base, opp))
    idx = rng.choice(combined.shape[0], size=pop_size, replace=False)
    return combined[idx]


def sobol_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    """Sobol quasi-random population; falls back to uniform if SciPy is unavailable."""
    rng = _as_rng(rng)
    pop_size, dimension = _validate_shape(pop_size, dimension)
    lo, hi = _bounds(min_values, max_values, dimension)
    try:
        from scipy.stats import qmc
        seed = int(rng.integers(0, 2**31 - 1)) if hasattr(rng, "integers") else None
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        sample = sampler.random(n=pop_size)
    except Exception:
        sample = rng.uniform(0.0, 1.0, size=(pop_size, dimension))
    return lo + sample * (hi - lo)


def chaotic_init_function(map_name: str = "logistic", seed: float = 0.7) -> Callable:
    """Return an ``init_function`` wrapper backed by ``chaotic_population``."""
    def _fn(problem, pop_size, rng=None, engine=None, **kwargs):
        dim, lo, hi = _problem_bounds(problem)
        return chaotic_population(
            pop_size=int(pop_size),
            dimension=int(dim),
            min_values=lo,
            max_values=hi,
            map_name=map_name,
            seed=seed,
        )
    _fn.__name__ = f"chaotic_{map_name}_init"
    return _fn


def _wrap_population_builder(builder: Callable, name: str | None = None):
    def _fn(problem, pop_size, rng=None, engine=None, **kwargs):
        dim, lo, hi = _problem_bounds(problem)
        return builder(
            pop_size=int(pop_size),
            dimension=int(dim),
            min_values=lo,
            max_values=hi,
            rng=rng,
        )
    _fn.__name__ = name or getattr(builder, "__name__", "init_function")
    return _fn


_CANONICAL_INIT_STRATEGIES = {
    "uniform": _wrap_population_builder(uniform_population, "uniform_init"),
    "lhs": _wrap_population_builder(lhs_population, "lhs_init"),
    "obl": _wrap_population_builder(obl_population, "obl_init"),
    "sobol": _wrap_population_builder(sobol_population, "sobol_init"),
}

_INIT_ALIASES = {
    "random": "uniform",
    "uniform_population": "uniform",
    "latin": "lhs",
    "latin_hypercube": "lhs",
    "latin-hypercube": "lhs",
    "latin hypercube": "lhs",
    "lhs_population": "lhs",
    "opposition": "obl",
    "opposition_based": "obl",
    "opposition-based": "obl",
    "opposition based": "obl",
    "obl_population": "obl",
    "sobol_population": "sobol",
    "qmc_sobol": "sobol",
}

_INIT_REGISTRY = dict(_CANONICAL_INIT_STRATEGIES)
for alias, canonical in _INIT_ALIASES.items():
    _INIT_REGISTRY[alias] = _CANONICAL_INIT_STRATEGIES[canonical]

AVAILABLE_INIT_STRATEGIES = sorted(_CANONICAL_INIT_STRATEGIES.keys())


def get_init_function(name: str):
    """
    Resolve a named initialisation preset to a callable.

    Accepted canonical names are ``uniform``, ``lhs``, ``obl`` and ``sobol``.
    Useful aliases include ``latin_hypercube`` and ``lhs_population``. Chaotic
    maps are selected with ``chaotic:<map_name>``, for example ``chaotic:tent``.
    """
    key = str(name).strip().lower().replace("_population", "_population")
    if key.startswith("chaotic:"):
        return chaotic_init_function(map_name=key.split(":", 1)[1] or "logistic")
    if key not in _INIT_REGISTRY:
        accepted = sorted(set(AVAILABLE_INIT_STRATEGIES) | set(_INIT_ALIASES.keys()) | {"chaotic:<map>"})
        raise KeyError(f"Unknown init strategy: {name}. Available: {', '.join(accepted)}")
    return _INIT_REGISTRY[key]
