"""Benchmark problem definitions and suites.

This module intentionally stays lightweight.  It wraps the existing
``pymetaheuristic.src.utils.Problem`` classes and ordinary Python callables into
a common object that the benchmark runner can execute repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..utils import Problem as UtilityProblem
from ..utils import FunctionalProblem, get_test_problem


_KNOWN_MINIMA: dict[str, float] = {
    "sphere": 0.0,
    "de_jong_1": 0.0,
    "rastrigin": 0.0,
    "ackley": 0.0,
    "rosenbrock": 0.0,
    "rosenbrocks_valley": 0.0,
    "zakharov": 0.0,
}


def _full_array(value, dimension: int) -> list[float]:
    if value is None:
        raise ValueError("Bound value cannot be None.")
    if np.isscalar(value):
        return [float(value)] * int(dimension)
    values = list(value)
    if len(values) != int(dimension):
        raise ValueError(f"Expected {dimension} bounds, received {len(values)}.")
    return [float(v) for v in values]


@dataclass
class BenchmarkProblem:
    """A problem entry used by :class:`BenchmarkStudy`.

    Parameters
    ----------
    function:
        Callable accepting a list-like position and returning a scalar fitness.
    min_values, max_values:
        Box bounds.
    name:
        Problem label used in result tables.
    objective:
        ``"min"`` or ``"max"``.
    optimum:
        Known optimum fitness.  When provided, benchmark records include
        ``error_to_optimum`` and target-hitting statistics.
    metadata:
        Free-form additional information preserved in manifests.
    """

    function: Callable[[Sequence[float]], float]
    min_values: list[float]
    max_values: list[float]
    name: str = "problem"
    objective: str = "min"
    optimum: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.min_values = [float(v) for v in self.min_values]
        self.max_values = [float(v) for v in self.max_values]
        if len(self.min_values) != len(self.max_values):
            raise ValueError("min_values and max_values must have the same length.")
        if self.objective not in {"min", "max"}:
            raise ValueError("objective must be 'min' or 'max'.")

    @property
    def dimension(self) -> int:
        return len(self.min_values)

    def evaluate(self, x) -> float:
        return float(self.function(list(x)))

    def error(self, fitness: float) -> float | None:
        if self.optimum is None:
            return None
        return abs(float(fitness) - float(self.optimum))

    def reached_target(self, fitness: float, tolerance: float = 1.0e-8) -> bool:
        err = self.error(fitness)
        return bool(err is not None and err <= float(tolerance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "min_values": list(self.min_values),
            "max_values": list(self.max_values),
            "objective": self.objective,
            "optimum": self.optimum,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_any(cls, value: Any, *, default_dimension: int | None = None, objective: str = "min") -> "BenchmarkProblem":
        """Normalize common problem specifications.

        Accepted inputs include:
        - ``BenchmarkProblem``
        - existing ``utils.Problem`` instances
        - strings accepted by ``get_test_problem``
        - ``(name, dimension)`` tuples
        - dictionaries with either ``name`` or ``function``
        - plain callables plus explicit bounds in a dictionary
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, UtilityProblem):
            key = str(value.name).lower()
            optimum = value.metadata.get("optimum", _KNOWN_MINIMA.get(key))
            return cls(
                function=value,
                min_values=list(value.lower),
                max_values=list(value.upper),
                name=value.name,
                objective=objective,
                optimum=optimum,
                metadata=dict(value.metadata or {}),
            )

        if isinstance(value, str):
            problem = get_test_problem(value, dimension=default_dimension or 2)
            return cls.from_any(problem, objective=objective)

        if isinstance(value, tuple) and len(value) in {1, 2, 3}:
            name = value[0]
            dimension = int(value[1]) if len(value) >= 2 else (default_dimension or 2)
            opts = value[2] if len(value) >= 3 else {}
            if opts is None:
                opts = {}
            if not isinstance(opts, dict):
                raise TypeError("Problem tuple third item must be a dict of options.")
            problem = get_test_problem(str(name), dimension=dimension, lower=opts.get("lower"), upper=opts.get("upper"))
            bp = cls.from_any(problem, objective=opts.get("objective", objective))
            if "optimum" in opts:
                bp.optimum = opts["optimum"]
            if "name" in opts:
                bp.name = str(opts["name"])
            return bp

        if isinstance(value, dict):
            data = dict(value)
            obj = data.get("objective", objective)
            if "problem" in data:
                bp = cls.from_any(data["problem"], default_dimension=data.get("dimension", default_dimension), objective=obj)
                if "name" in data:
                    bp.name = str(data["name"])
                if "optimum" in data:
                    bp.optimum = data["optimum"]
                return bp
            if "function" in data or "target_function" in data:
                fn = data.get("function", data.get("target_function"))
                if not callable(fn):
                    raise TypeError("function/target_function must be callable.")
                dim = int(data.get("dimension") or default_dimension or len(data.get("min_values", data.get("lower", []))))
                lower = data.get("min_values", data.get("lower"))
                upper = data.get("max_values", data.get("upper"))
                return cls(
                    function=fn,
                    min_values=_full_array(lower, dim),
                    max_values=_full_array(upper, dim),
                    name=str(data.get("name", getattr(fn, "__name__", "custom"))),
                    objective=obj,
                    optimum=data.get("optimum"),
                    metadata=dict(data.get("metadata") or {}),
                )
            if "name" in data:
                dimension = int(data.get("dimension") or default_dimension or 2)
                problem = get_test_problem(str(data["name"]), dimension=dimension, lower=data.get("lower"), upper=data.get("upper"))
                bp = cls.from_any(problem, objective=obj)
                bp.name = str(data.get("label", data.get("name", bp.name)))
                if "optimum" in data:
                    bp.optimum = data["optimum"]
                return bp
            raise TypeError("Problem dictionaries need 'name', 'problem', or 'function'.")

        if callable(value):
            if default_dimension is None:
                raise TypeError("Callable problems require bounds; pass a dict with function/min_values/max_values.")
            raise TypeError("Callable problems require explicit bounds; pass a dict with function/min_values/max_values.")

        raise TypeError(f"Unsupported problem specification: {type(value)!r}")


@dataclass
class ProblemSuite:
    """Small container for a list of benchmark problems."""

    problems: list[BenchmarkProblem]

    def __iter__(self):
        return iter(self.problems)

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> BenchmarkProblem:
        return self.problems[index]

    def to_dict(self) -> list[dict[str, Any]]:
        return [problem.to_dict() for problem in self.problems]

    @classmethod
    def from_any(cls, values: Any, *, default_dimension: int | None = None, objective: str = "min") -> "ProblemSuite":
        if isinstance(values, cls):
            return values
        if values is None:
            raise ValueError("At least one benchmark problem is required.")
        if isinstance(values, (str, BenchmarkProblem, UtilityProblem, dict)):
            values = [values]
        return cls([
            BenchmarkProblem.from_any(value, default_dimension=default_dimension, objective=objective)
            for value in list(values)
        ])

    @classmethod
    def from_names(cls, names: Iterable[str], dimensions: Iterable[int] | int = 2, **kwargs) -> "ProblemSuite":
        if isinstance(dimensions, int):
            dims = [int(dimensions)]
        else:
            dims = [int(d) for d in dimensions]
        problems: list[BenchmarkProblem] = []
        for name in names:
            for dim in dims:
                problem = get_test_problem(str(name), dimension=dim, lower=kwargs.get("lower"), upper=kwargs.get("upper"))
                bp = BenchmarkProblem.from_any(problem, objective=kwargs.get("objective", "min"))
                bp.name = f"{bp.name}_D{dim}"
                problems.append(bp)
        return cls(problems)


# Friendly aliases for the root namespace.  ``BenchmarkProblem`` avoids
# colliding with the already exported modelling ``Problem`` class.
Problem = BenchmarkProblem


__all__ = ["BenchmarkProblem", "Problem", "ProblemSuite"]
