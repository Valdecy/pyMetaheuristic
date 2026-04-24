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


def _bounds(min_values, max_values):
    return np.asarray(min_values, dtype=float), np.asarray(max_values, dtype=float)


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
    return dim, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def uniform_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = _bounds(min_values, max_values)
    return rng.uniform(lo, hi, size=(int(pop_size), int(dimension)))


def lhs_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    pop_size = int(pop_size)
    dimension = int(dimension)
    lo, hi = _bounds(min_values, max_values)
    result = np.empty((pop_size, dimension), dtype=float)
    cut = np.linspace(0.0, 1.0, pop_size + 1)
    for j in range(dimension):
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        result[:, j] = lo[j] + u * (hi[j] - lo[j])
    return result


def obl_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = _bounds(min_values, max_values)
    base = rng.uniform(lo, hi, size=(int(pop_size), int(dimension)))
    opp = lo + hi - base
    combined = np.vstack((base, opp))
    idx = rng.choice(combined.shape[0], size=int(pop_size), replace=False)
    return combined[idx]


def sobol_population(pop_size: int, dimension: int, min_values, max_values, rng=None) -> np.ndarray:
    lo, hi = _bounds(min_values, max_values)
    pop_size = int(pop_size)
    dimension = int(dimension)
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=None if rng is None else int(rng.integers(0, 2**31 - 1)))
        sample = sampler.random(n=pop_size)
    except Exception:
        if rng is None:
            rng = np.random.default_rng()
        sample = rng.uniform(0.0, 1.0, size=(pop_size, dimension))
    return lo + sample * (hi - lo)


def chaotic_init_function(map_name: str = "logistic", seed: float = 0.7) -> Callable:
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


def _wrap_population_builder(builder: Callable):
    def _fn(problem, pop_size, rng=None, engine=None, **kwargs):
        dim, lo, hi = _problem_bounds(problem)
        return builder(
            pop_size=int(pop_size),
            dimension=int(dim),
            min_values=lo,
            max_values=hi,
            rng=rng,
        )
    _fn.__name__ = getattr(builder, "__name__", "init_function")
    return _fn


_INIT_REGISTRY = {
    "uniform": _wrap_population_builder(uniform_population),
    "lhs": _wrap_population_builder(lhs_population),
    "obl": _wrap_population_builder(obl_population),
    "sobol": _wrap_population_builder(sobol_population),
}

AVAILABLE_INIT_STRATEGIES = sorted(_INIT_REGISTRY.keys())


def get_init_function(name: str):
    key = str(name).strip().lower()
    if key.startswith("chaotic:"):
        return chaotic_init_function(map_name=key.split(":", 1)[1] or "logistic")
    if key not in _INIT_REGISTRY:
        raise KeyError(f"Unknown init strategy: {name}. Available: {', '.join(AVAILABLE_INIT_STRATEGIES + ['chaotic:<map>'])}")
    return _INIT_REGISTRY[key]
