"""
pyMetaheuristic src — Typed Variable Space
==========================================


Provides typed variable descriptors that can be composed into a
mixed-type search space.  Each descriptor knows how to:

* generate a random value within its domain
* encode a decoded (real-world) value into a continuous float representation
* decode a continuous float back to the real-world value
* clip / correct a raw float to stay within bounds

These classes integrate seamlessly with ``ProblemSpec`` through the
helper ``build_problem_spec()``, which accepts either the old flat-list
API (fully backward-compatible) or a list of typed variable objects.

Quick start
-----------
::

    from pymetaheuristic.src.utils.space import FloatVar, IntegerVar, CategoricalVar
    from pymetaheuristic.src.utils.space import build_problem_spec
    import pymetaheuristic

    bounds = [
        FloatVar  (lb=-5.0, ub=5.0,          name="x1"),
        IntegerVar(lb=1,    ub=10,            name="n_clusters"),
        CategoricalVar(options=["relu", "tanh", "sigmoid"], name="activation"),
    ]

    def my_fn(decoded):
        # decoded is a dict: {"x1": float, "n_clusters": int, "activation": str}
        x1   = decoded["x1"]
        k    = decoded["n_clusters"]
        act  = decoded["activation"]
        return x1**2 + k

    spec = build_problem_spec(
        target_function = my_fn,
        bounds          = bounds,
        objective       = "min",
    )
    result = pymetaheuristic.optimize(
        algorithm       = "pso",
        target_function = spec.target_function,
        min_values      = spec.min_values,
        max_values      = spec.max_values,
        max_steps       = 300,
    )
    # Decode the best position back to the typed domain:
    from pymetaheuristic.src.utils.space import decode_position
    best_decoded = decode_position(result.best_position, bounds)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ===========================================================================
# Abstract base
# ===========================================================================

class BaseVar(ABC):
    """
    Abstract typed variable descriptor.

    Each concrete subclass occupies one or more *internal* continuous
    dimensions (``n_dims``).  The engine always works in the continuous
    internal space; encoding/decoding translates between that space and
    the real-world domain.
    """

    def __init__(self, name: str = "var") -> None:
        self.name = name

    # Number of continuous dimensions this variable occupies internally
    @property
    @abstractmethod
    def n_dims(self) -> int: ...

    # Continuous lower/upper bounds (length == n_dims)
    @property
    @abstractmethod
    def lb(self) -> list[float]: ...

    @property
    @abstractmethod
    def ub(self) -> list[float]: ...

    @abstractmethod
    def generate(self) -> Any:
        """Draw a random value from the domain."""

    @abstractmethod
    def encode(self, value: Any) -> list[float]:
        """Encode a domain value into a continuous vector of length n_dims."""

    @abstractmethod
    def decode(self, x: list[float]) -> Any:
        """Decode a continuous vector of length n_dims to a domain value."""

    def correct(self, x: list[float]) -> list[float]:
        """Clip x to the continuous bounds."""
        return [
            float(min(max(xi, lo), hi))
            for xi, lo, hi in zip(x, self.lb, self.ub)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ===========================================================================
# FloatVar
# ===========================================================================

class FloatVar(BaseVar):
    """
    Continuous floating-point variable in [lb, ub].

    Parameters
    ----------
    lb, ub : float
        Lower and upper bounds (inclusive).
    name   : str
    """

    def __init__(self, lb: float, ub: float, name: str = "float_var") -> None:
        super().__init__(name)
        if lb >= ub:
            raise ValueError(f"FloatVar '{name}': lb ({lb}) must be < ub ({ub})")
        self._lb = float(lb)
        self._ub = float(ub)

    @property
    def n_dims(self) -> int: return 1

    @property
    def lb(self) -> list[float]: return [self._lb]

    @property
    def ub(self) -> list[float]: return [self._ub]

    def generate(self) -> float:
        return random.uniform(self._lb, self._ub)

    def encode(self, value: float) -> list[float]:
        return [float(value)]

    def decode(self, x: list[float]) -> float:
        return float(min(max(x[0], self._lb), self._ub))

    def __repr__(self) -> str:
        return f"FloatVar(name={self.name!r}, lb={self._lb}, ub={self._ub})"


# ===========================================================================
# IntegerVar
# ===========================================================================

class IntegerVar(BaseVar):
    """
    Integer variable in {lb, lb+1, …, ub} (both inclusive).

    Internally represented as a continuous float in [lb, ub]; decoded by
    rounding to the nearest integer.

    Parameters
    ----------
    lb, ub : int
    name   : str
    """

    def __init__(self, lb: int, ub: int, name: str = "int_var") -> None:
        super().__init__(name)
        if int(lb) > int(ub):
            raise ValueError(f"IntegerVar '{name}': lb ({lb}) must be <= ub ({ub})")
        self._lb = int(lb)
        self._ub = int(ub)

    @property
    def n_dims(self) -> int: return 1

    @property
    def lb(self) -> list[float]: return [float(self._lb)]

    @property
    def ub(self) -> list[float]: return [float(self._ub)]

    def generate(self) -> int:
        return random.randint(self._lb, self._ub)

    def encode(self, value: int) -> list[float]:
        return [float(int(value))]

    def decode(self, x: list[float]) -> int:
        return int(round(min(max(x[0], self._lb), self._ub)))

    def __repr__(self) -> str:
        return f"IntegerVar(name={self.name!r}, lb={self._lb}, ub={self._ub})"


# ===========================================================================
# BinaryVar
# ===========================================================================

class BinaryVar(BaseVar):
    """
    Single binary variable in {0, 1}.

    Internally: continuous float in [0, 1]; decoded by rounding.
    """

    def __init__(self, name: str = "binary_var") -> None:
        super().__init__(name)

    @property
    def n_dims(self) -> int: return 1

    @property
    def lb(self) -> list[float]: return [0.0]

    @property
    def ub(self) -> list[float]: return [1.0]

    def generate(self) -> int:
        return random.randint(0, 1)

    def encode(self, value: int) -> list[float]:
        return [float(int(bool(value)))]

    def decode(self, x: list[float]) -> int:
        return 1 if x[0] >= 0.5 else 0

    def __repr__(self) -> str:
        return f"BinaryVar(name={self.name!r})"


# ===========================================================================
# CategoricalVar
# ===========================================================================

class CategoricalVar(BaseVar):
    """
    Categorical variable over a fixed ordered list of options.

    Internally represented as a continuous float in [0, len(options)-1].
    Decoding maps the float index to the corresponding option.

    Parameters
    ----------
    options : list
        The allowed values (any hashable Python objects).
    name    : str
    """

    def __init__(self, options: list, name: str = "cat_var") -> None:
        super().__init__(name)
        if len(options) < 2:
            raise ValueError(
                f"CategoricalVar '{name}': must have at least 2 options, "
                f"got {len(options)}"
            )
        self.options = list(options)
        self._n = len(self.options)

    @property
    def n_dims(self) -> int: return 1

    @property
    def lb(self) -> list[float]: return [0.0]

    @property
    def ub(self) -> list[float]: return [float(self._n - 1)]

    def generate(self) -> Any:
        return random.choice(self.options)

    def encode(self, value: Any) -> list[float]:
        if value in self.options:
            return [float(self.options.index(value))]
        raise ValueError(
            f"CategoricalVar '{self.name}': value {value!r} not in options {self.options}"
        )

    def decode(self, x: list[float]) -> Any:
        idx = int(round(min(max(x[0], 0.0), float(self._n - 1))))
        return self.options[idx]

    def __repr__(self) -> str:
        return f"CategoricalVar(name={self.name!r}, options={self.options})"


# ===========================================================================
# PermutationVar
# ===========================================================================

class PermutationVar(BaseVar):
    """
    Permutation variable over a sequence of *n* items.

    Internally uses *n* continuous floats in [0, n-1] (one per position).
    Decoding applies argsort to obtain a valid permutation (no repeats,
    no missing indices).

    Parameters
    ----------
    n    : int  — number of elements to permute (0-indexed: 0 … n-1)
    name : str
    """

    def __init__(self, n: int, name: str = "perm_var") -> None:
        super().__init__(name)
        if n < 2:
            raise ValueError(f"PermutationVar '{name}': n must be >= 2")
        self._n = int(n)

    @property
    def n_dims(self) -> int: return self._n

    @property
    def lb(self) -> list[float]: return [0.0] * self._n

    @property
    def ub(self) -> list[float]: return [float(self._n - 1)] * self._n

    def generate(self) -> list[int]:
        perm = list(range(self._n))
        random.shuffle(perm)
        return perm

    def encode(self, value: list[int]) -> list[float]:
        if sorted(value) != list(range(self._n)):
            raise ValueError(
                f"PermutationVar '{self.name}': expected a permutation of "
                f"0..{self._n-1}, got {value}"
            )
        return [float(v) for v in value]

    def decode(self, x: list[float]) -> list[int]:
        arr = np.asarray(x[:self._n], dtype=float)
        # Add tiny noise to break ties deterministically
        tie_break = np.arange(self._n) * 1e-9
        return list(map(int, np.argsort(arr + tie_break)))

    def __repr__(self) -> str:
        return f"PermutationVar(name={self.name!r}, n={self._n})"


# ===========================================================================
# Helper functions
# ===========================================================================

def encode_position(decoded_values: list[Any], bounds: list[BaseVar]) -> list[float]:
    """
    Encode a list of real-world values (one per variable) into a flat
    continuous position vector.

    Parameters
    ----------
    decoded_values : list, one value per variable in *bounds*
    bounds         : list of BaseVar descriptors

    Returns
    -------
    Flat list of floats (length = sum of n_dims for each variable).
    """
    if len(decoded_values) != len(bounds):
        raise ValueError(
            f"encode_position: got {len(decoded_values)} values but "
            f"{len(bounds)} variable descriptors"
        )
    out: list[float] = []
    for val, var in zip(decoded_values, bounds):
        out.extend(var.encode(val))
    return out


def decode_position(
    position: list[float],
    bounds: list[BaseVar],
) -> dict[str, Any]:
    """
    Decode a flat continuous position vector into a dict mapping variable
    names to their real-world values.

    Parameters
    ----------
    position : flat continuous vector (length = sum of n_dims)
    bounds   : list of BaseVar descriptors

    Returns
    -------
    dict {name: decoded_value}
    """
    result: dict[str, Any] = {}
    idx = 0
    for var in bounds:
        chunk = position[idx: idx + var.n_dims]
        result[var.name] = var.decode(chunk)
        idx += var.n_dims
    return result


def build_problem_spec(
    target_function,
    bounds: list[BaseVar],
    objective: str = "min",
    constraints=None,
    constraint_handler=None,
    **kwargs,
):
    """
    Construct a ``ProblemSpec``-compatible namespace from typed variable
    descriptors.

    The returned object is a simple namespace that matches the fields
    expected by ``create_optimizer`` and ``optimize``:
    ``target_function``, ``min_values``, ``max_values``, ``variable_types``.

    The ``target_function`` stored in the namespace automatically decodes
    each continuous position vector back to a dict of typed values before
    calling the user's original function.

    Parameters
    ----------
    target_function : callable
        User objective.  Receives a *dict* {name: value} when bounds are
        typed, or a list when called with raw floats.
    bounds          : list[BaseVar]
        Typed variable descriptors.
    objective       : "min" or "max"
    constraints     : optional constraint list (passed through unchanged)
    constraint_handler : optional handler name (passed through unchanged)

    Returns
    -------
    A namespace with attributes: target_function, min_values, max_values,
    variable_types, objective, constraints, constraint_handler, bounds.

    Example
    -------
    ::

        spec = build_problem_spec(my_fn, [FloatVar(-5,5,"x"), IntegerVar(1,10,"k")])
        result = pymetaheuristic.optimize(
            algorithm       = "pso",
            target_function = spec.target_function,
            min_values      = spec.min_values,
            max_values      = spec.max_values,
            max_steps       = 200,
        )
        best_decoded = decode_position(result.best_position, spec.bounds)
    """
    from types import SimpleNamespace

    # Aggregate flat bounds
    min_values: list[float] = []
    max_values: list[float] = []
    variable_types: list[str] = []
    for var in bounds:
        min_values.extend(var.lb)
        max_values.extend(var.ub)
        t = type(var).__name__
        variable_types.extend([t] * var.n_dims)

    # Wrap objective to perform decoding transparently
    _bounds = list(bounds)
    _orig   = target_function

    def _decoded_fn(x):
        decoded = decode_position(list(x), _bounds)
        return _orig(decoded)

    return SimpleNamespace(
        target_function   = _decoded_fn,
        min_values        = min_values,
        max_values        = max_values,
        variable_types    = variable_types,
        objective         = objective,
        constraints       = constraints,
        constraint_handler= constraint_handler,
        bounds            = _bounds,
        _original_fn      = _orig,
    )
