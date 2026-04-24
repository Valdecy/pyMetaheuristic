"""
pyMetaheuristic src — Chaotic Maps and Transfer Functions
=========================================================

chaotic.py
----------
Ten classical chaotic maps for population initialisation, perturbation,
and parameter control.  Each map takes a scalar x ∈ [0, 1] and returns
the next value in the chaotic sequence.

Usage::

    from pymetaheuristic.src.utils.chaotic import ChaoticMap, chaotic_sequence

    # Generate 100 values using the logistic map
    seq = chaotic_sequence(n=100, map_name="logistic", seed=0.42)

    # Use inside an engine for chaotic initialisation
    lo, hi = np.array(problem.min_values), np.array(problem.max_values)
    chaos  = chaotic_sequence(n=pop_size * dim, map_name="logistic", seed=0.3)
    chaos  = chaos.reshape(pop_size, dim)
    pop    = lo + chaos * (hi - lo)

transfer.py (binary/discrete optimisation adapters)
----------------------------------------------------
Eight transfer functions (4 V-shaped, 4 S-shaped) that map continuous
real-valued positions to probabilities for bit-flipping, enabling any
continuous metaheuristic to solve binary or boolean problems.

Usage::

    from pymetaheuristic.src.utils.transfer import vstf_01, sstf_02, BinaryAdapter

    # Standalone — convert position values to flip probabilities
    probs  = vstf_01(np.array([0.5, -1.2, 2.3]))

    # BinaryAdapter wraps any engine and applies a transfer function at each step
    adapter = BinaryAdapter(engine, transfer_fn="v1")
    result  = adapter.run()
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ===========================================================================
# Chaotic Maps
# ===========================================================================

class ChaoticMap:
    """
    Ten static chaotic map implementations.

    Each method receives one scalar *x* in [0, 1] (or (-1, 1) for Chebyshev)
    and returns the next iterate.  They are pure functions — no state.
    """

    @staticmethod
    def logistic(x: float, a: float = 4.0) -> float:
        """Logistic map: x_{n+1} = a·x·(1-x).  Chaotic for a = 4."""
        return a * x * (1.0 - x)

    @staticmethod
    def tent(x: float, mu: float = 0.499) -> float:
        """Tent map."""
        if x < mu:
            return x / mu
        return (1.0 - x) / (1.0 - mu)

    @staticmethod
    def bernoulli(x: float, a: float = 0.5) -> float:
        """Bernoulli shift map."""
        if 0.0 <= x <= a:
            return x / (1.0 - a)
        return (x - a) / a

    @staticmethod
    def chebyshev(x: float, a: float = 4.0) -> float:
        """Chebyshev map: cos(a·arccos(x)).  Input/output in [-1, 1]."""
        return float(np.cos(a * np.arccos(np.clip(x, -1.0, 1.0))))

    @staticmethod
    def circle(x: float, a: float = 0.5, b: float = 0.2) -> float:
        """Circle map."""
        return (x + b - (a / (2.0 * np.pi)) * np.sin(2.0 * np.pi * x)) % 1.0

    @staticmethod
    def cubic(x: float, q: float = 2.59) -> float:
        """Cubic map: q·x·(1 - x²)."""
        return q * x * (1.0 - x * x)

    @staticmethod
    def icmic(x: float, a: float = 0.7) -> float:
        """Iterative chaotic map with infinite collapses (ICMIC): |sin(a/x)|."""
        if abs(x) < 1e-12:
            return 0.1  # avoid division by zero
        return float(np.abs(np.sin(a / x)))

    @staticmethod
    def piecewise(x: float, a: float = 0.4) -> float:
        """Piecewise map."""
        if 0.0 <= x < a:
            return x / a
        if a <= x < 0.5:
            return (x - a) / (0.5 - a)
        if 0.5 <= x < 1.0 - a:
            return (1.0 - a - x) / (0.5 - a)
        return (1.0 - x) / a

    @staticmethod
    def sine(x: float, a: float = 1.0) -> float:
        """Sine map: (a/4)·sin(π·x)."""
        return (a / 4.0) * np.sin(np.pi * x)

    @staticmethod
    def gauss(x: float) -> float:
        """Gauss / mouse map."""
        if abs(x) < 1e-12:
            return 0.0
        return (1.0 / x) % 1.0


# Registry: map name → callable
_MAP_REGISTRY: dict[str, Callable[[float], float]] = {
    "logistic":   ChaoticMap.logistic,
    "tent":       ChaoticMap.tent,
    "bernoulli":  ChaoticMap.bernoulli,
    "chebyshev":  ChaoticMap.chebyshev,
    "circle":     ChaoticMap.circle,
    "cubic":      ChaoticMap.cubic,
    "icmic":      ChaoticMap.icmic,
    "piecewise":  ChaoticMap.piecewise,
    "sine":       ChaoticMap.sine,
    "gauss":      ChaoticMap.gauss,
}

AVAILABLE_CHAOTIC_MAPS: list[str] = sorted(_MAP_REGISTRY.keys())


def chaotic_sequence(
    n: int,
    map_name: str = "logistic",
    seed: float = 0.7,
    skip: int = 100,
) -> np.ndarray:
    """
    Generate *n* values from a chaotic map.

    Parameters
    ----------
    n        : Number of values to generate.
    map_name : Name of the map (see ``AVAILABLE_CHAOTIC_MAPS``).
    seed     : Initial value x₀ ∈ (0, 1) exclusive.  Must not be 0 or 1
               for most maps.
    skip     : Discard the first *skip* iterates to enter the chaotic regime.

    Returns
    -------
    numpy array of length *n* with values (approximately) in [0, 1].
    """
    if map_name not in _MAP_REGISTRY:
        raise ValueError(
            f"Unknown chaotic map '{map_name}'. "
            f"Available: {AVAILABLE_CHAOTIC_MAPS}"
        )
    fn = _MAP_REGISTRY[map_name]
    x = float(seed)
    # Ensure seed is not a fixed point for common maps
    if abs(x) < 1e-12 or abs(x - 1.0) < 1e-12:
        x = 0.7

    # Skip transient
    for _ in range(skip):
        x = fn(x)

    # Generate
    out = np.empty(n, dtype=float)
    for i in range(n):
        x = fn(x)
        out[i] = x

    # Normalise to [0, 1] (some maps like chebyshev output [-1, 1])
    lo, hi = out.min(), out.max()
    if hi - lo > 1e-12:
        out = (out - lo) / (hi - lo)
    else:
        out = np.full(n, 0.5)

    return out


def chaotic_population(
    pop_size: int,
    dimension: int,
    min_values: list[float],
    max_values: list[float],
    map_name: str = "logistic",
    seed: float = 0.7,
) -> np.ndarray:
    """
    Generate a *pop_size × dimension* initial population using a chaotic map.

    Each element is mapped from [0, 1] into the corresponding variable range.
    Use this as a drop-in replacement for ``np.random.uniform`` in engine
    ``initialize()`` methods.

    Returns
    -------
    Array of shape (pop_size, dimension) with values inside bounds.
    """
    lo = np.array(min_values, dtype=float)
    hi = np.array(max_values, dtype=float)
    total = pop_size * dimension
    seq   = chaotic_sequence(total, map_name=map_name, seed=seed)
    mat   = seq.reshape(pop_size, dimension)
    return lo + mat * (hi - lo)


# ===========================================================================
# Transfer Functions  (for binary / boolean problems)
# ===========================================================================

# ---- V-shaped (output ∈ [0, 1], symmetric about 0) ----------------------

def vstf_01(x: np.ndarray) -> np.ndarray:
    """V-shaped TF 1: |erf((√π / 2) · x)|"""
    from scipy.special import erf  # optional import
    return np.abs(erf((np.sqrt(np.pi) / 2.0) * np.asarray(x, dtype=float)))

def vstf_02(x: np.ndarray) -> np.ndarray:
    """V-shaped TF 2: |tanh(x)|"""
    return np.abs(np.tanh(np.asarray(x, dtype=float)))

def vstf_03(x: np.ndarray) -> np.ndarray:
    """V-shaped TF 3: |x / √(1 + x²)|"""
    x = np.asarray(x, dtype=float)
    return np.abs(x / np.sqrt(1.0 + np.square(x)))

def vstf_04(x: np.ndarray) -> np.ndarray:
    """V-shaped TF 4: |(2/π)·arctan((π/2)·x)|"""
    x = np.asarray(x, dtype=float)
    return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * x))


# ---- S-shaped (sigmoid-like, output ∈ [0, 1]) ---------------------------

def sstf_01(x: np.ndarray) -> np.ndarray:
    """S-shaped TF 1: 1 / (1 + exp(-2x))"""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-2.0 * x))

def sstf_02(x: np.ndarray) -> np.ndarray:
    """S-shaped TF 2: 1 / (1 + exp(-x))  (standard logistic)"""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))

def sstf_03(x: np.ndarray) -> np.ndarray:
    """S-shaped TF 3: 1 / (1 + exp(-x/3))"""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x / 3.0))

def sstf_04(x: np.ndarray) -> np.ndarray:
    """S-shaped TF 4: 1 / (1 + exp(-x/2))"""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x / 2.0))


# Registry: short name → callable
_TF_REGISTRY: dict[str, Callable] = {
    "v1": vstf_01, "v2": vstf_02, "v3": vstf_03, "v4": vstf_04,
    "s1": sstf_01, "s2": sstf_02, "s3": sstf_03, "s4": sstf_04,
    # aliases
    "vstf_01": vstf_01, "vstf_02": vstf_02,
    "vstf_03": vstf_03, "vstf_04": vstf_04,
    "sstf_01": sstf_01, "sstf_02": sstf_02,
    "sstf_03": sstf_03, "sstf_04": sstf_04,
}

AVAILABLE_TRANSFER_FUNCTIONS: list[str] = [
    "v1", "v2", "v3", "v4", "s1", "s2", "s3", "s4",
]


def apply_transfer(x: np.ndarray, fn_name: str = "v2") -> np.ndarray:
    """
    Apply a named transfer function to a continuous position array.

    Parameters
    ----------
    x       : Continuous position values (any shape).
    fn_name : Transfer function identifier (see ``AVAILABLE_TRANSFER_FUNCTIONS``).

    Returns
    -------
    Array of probabilities in [0, 1] with the same shape as *x*.
    """
    if fn_name not in _TF_REGISTRY:
        raise ValueError(
            f"Unknown transfer function '{fn_name}'. "
            f"Available: {AVAILABLE_TRANSFER_FUNCTIONS}"
        )
    return _TF_REGISTRY[fn_name](np.asarray(x, dtype=float))


def binarize(
    x: np.ndarray,
    fn_name: str = "v2",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Convert a continuous position to a binary (0/1) vector.

    Uses stochastic binarisation: bit *i* is set to 1 with probability
    equal to transfer(x[i]).

    Parameters
    ----------
    x       : Continuous position array (1-D).
    fn_name : Transfer function to apply before sampling.
    rng     : Optional numpy random generator for reproducibility.

    Returns
    -------
    Binary array of the same shape as *x* (dtype int32).
    """
    probs = apply_transfer(x, fn_name)
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(probs.shape) < probs).astype(np.int32)


# ===========================================================================
# BinaryAdapter — wraps any BaseEngine for binary optimisation
# ===========================================================================

class BinaryAdapter:
    """
    Wraps any population-based engine and applies a transfer function
    at each step to decode the continuous internal positions into
    binary solutions that are passed to the objective function.

    The internal engine still searches in the *continuous* space
    [0, 1]^n, but the objective function receives binary {0, 1}^n vectors.

    Parameters
    ----------
    engine       : A fully constructed BaseEngine instance.
    transfer_fn  : Transfer function name (default "v2" = standard sigmoid).

    Example
    -------
    ::

        def knapsack(x):
            weights   = [2, 3, 4, 5]
            values    = [3, 4, 5, 6]
            capacity  = 8
            items     = [int(b) for b in x]
            if sum(w*i for w, i in zip(weights, items)) > capacity:
                return 0.0
            return -float(sum(v*i for v, i in zip(values, items)))  # min

        from pymetaheuristic import create_optimizer
        from pymetaheuristic.src.utils.chaotic import BinaryAdapter

        engine  = create_optimizer("pso", target_function=knapsack,
                                   min_values=[0]*4, max_values=[1]*4,
                                   max_steps=200)
        adapted = BinaryAdapter(engine, transfer_fn="v2")
        result  = adapted.run()
    """

    def __init__(self, engine, transfer_fn: str = "v2") -> None:
        if transfer_fn not in _TF_REGISTRY:
            raise ValueError(
                f"Unknown transfer function '{transfer_fn}'. "
                f"Available: {AVAILABLE_TRANSFER_FUNCTIONS}"
            )
        self._engine = engine
        self._tf     = transfer_fn
        self._rng    = np.random.default_rng(getattr(engine.config, "seed", None))

    def _wrap_objective(self, original_fn):
        """Return a wrapper that binarises positions before calling the original fn."""
        tf  = self._tf
        rng = self._rng

        def _wrapped(x):
            binary = binarize(np.asarray(x, dtype=float), fn_name=tf, rng=rng)
            return original_fn(binary.tolist())

        return _wrapped

    def run(self):
        """Run the wrapped engine and return the OptimizationResult."""
        orig_fn = self._engine.problem.target_function
        self._engine.problem.target_function = self._wrap_objective(orig_fn)
        result  = self._engine.run()
        # Restore so the engine is reusable
        self._engine.problem.target_function = orig_fn
        # Decode best_position to binary for result inspection
        if result.best_position is not None:
            binary_best = binarize(
                np.asarray(result.best_position), fn_name=self._tf, rng=self._rng
            )
            result.metadata["binary_best_position"] = binary_best.tolist()
        return result
