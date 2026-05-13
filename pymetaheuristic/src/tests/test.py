from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pymetaheuristic  # noqa: E402
from pymetaheuristic.src.engines import REGISTRY  # noqa: E402


LOWER = [-5.0, -5.0]
UPPER = [5.0, 5.0]


def sphere(x):
    return float(sum(float(v) * float(v) for v in x))


def _assert_valid_result(result, algorithm_id: str) -> None:
    assert result.algorithm_id == algorithm_id
    assert result.best_position is not None
    assert len(result.best_position) == len(LOWER)
    assert math.isfinite(float(result.best_fitness))
    for value, lo, hi in zip(result.best_position, LOWER, UPPER):
        assert lo - 1.0e-8 <= float(value) <= hi + 1.0e-8


def test_registry_is_not_empty_and_matches_public_algorithm_list():
    public_ids = set(pymetaheuristic.list_algorithms())
    registry_ids = set(REGISTRY)
    assert registry_ids
    assert public_ids == registry_ids


@pytest.mark.parametrize("algorithm_id", sorted(REGISTRY), ids=sorted(REGISTRY))
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_all_registered_algorithms_run_on_sphere(algorithm_id):
    result = pymetaheuristic.optimize(
        algorithm=algorithm_id,
        target_function=sphere,
        min_values=LOWER,
        max_values=UPPER,
        objective="min",
        max_steps=2,
        max_evaluations=80,
        seed=123,
        store_history=False,
        verbose=False,
    )
    _assert_valid_result(result, algorithm_id)


@pytest.mark.parametrize("algorithm_id", sorted(REGISTRY), ids=sorted(REGISTRY))
def test_all_registered_algorithms_expose_metadata(algorithm_id):
    info = pymetaheuristic.get_algorithm_info(algorithm_id)
    assert info["algorithm_id"] == algorithm_id
    assert isinstance(info["algorithm_name"], str) and info["algorithm_name"].strip()
    assert isinstance(info["family"], str) and info["family"].strip()
    assert "capabilities" in info
    assert "defaults" in info
    assert "info" in info and algorithm_id in info["info"]
