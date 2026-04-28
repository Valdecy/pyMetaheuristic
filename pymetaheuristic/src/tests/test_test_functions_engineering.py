from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pymetaheuristic  # noqa: E402


def test_engineering_benchmarks_are_public_and_feasible():
    names = pymetaheuristic.list_engineering_benchmarks()
    assert "tension_spring" in names
    assert "gear_train" in names

    report = pymetaheuristic.validate_engineering_benchmarks()
    assert set(names) <= set(report)
    assert all(item["max_violation"] <= 1e-4 for item in report.values())


def test_engineering_benchmark_metadata_is_optimizer_ready():
    bench = pymetaheuristic.get_engineering_benchmark("tension_spring")
    assert callable(bench["objective"])
    assert bench["constraints"]
    assert len(bench["min_values"]) == 3
    assert len(bench["max_values"]) == 3
    assert pymetaheuristic.get_test_function("tension_spring") is bench["objective"]


def test_engineering_problem_wrapper_builds_problem_spec():
    problem = pymetaheuristic.get_engineering_problem("tension_spring")
    spec = problem.to_problem_spec()
    assert spec.target_function is problem
    assert spec.constraint_handler == "deb"
    assert len(spec.constraints) == 4
    assert spec.min_values == list(problem.lower)
    assert spec.max_values == list(problem.upper)


def test_test_function_helpers_include_new_metadata():
    names = pymetaheuristic.list_test_functions()
    assert "tension_spring" in names
    assert "cec_2022_f11" in names

    info = pymetaheuristic.get_test_function_info("tension_spring")
    assert "engineering" in info["name"].lower() or "spring" in info["name"].lower()
