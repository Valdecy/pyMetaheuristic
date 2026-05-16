from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pymetaheuristic as pmh  # noqa: E402


def sphere(x):
    return sum(v * v for v in x)


def test_island_system_cooperative_configuration_objects():
    system = pmh.IslandSystem(
        islands=[
            pmh.Island("de", config={"population_size": 8}, label="DE", role="explorer"),
            pmh.Island("ga", config={"population_size": 8}, label="GA", role="diversity_keeper"),
        ],
        topology=pmh.TopologyConfig(name="bidirectional_ring"),
        migration=pmh.MigrationConfig(interval=2, size=1, mode="elite", policy="push_pull"),
        max_steps=3,
        seed=7,
    )
    result = system.optimize(sphere, [-5, -5], [5, 5], mode="cooperative")
    assert result.best_fitness is not None
    assert result.metadata["topology"] == "bidirectional_ring"
    assert result.metadata["migration_policy"] == "push_pull"
    assert system.describe()["islands"][0]["role"] == "explorer"


def test_island_system_fixed_orchestration_preserves_migration_settings():
    system = pmh.IslandSystem(
        islands=[
            {"algorithm": "de", "config": {"population_size": 8}, "label": "DE"},
            {"algorithm": "ga", "config": {"population_size": 8}, "label": "GA"},
        ],
        topology={"name": "star"},
        migration={"interval": 2, "size": 2, "mode": "elite", "policy": "push_pull"},
        max_steps=3,
        seed=11,
    )
    result = system.optimize(sphere, [-5, -5], [5, 5], mode="fixed")
    assert result.best_fitness is not None
    assert result.controller_mode == "fixed"
    assert result.metadata["topology"] == "star"
    assert result.metadata["migration_policy"] == "push_pull"
    assert result.metadata["migration_size"] == 2


def test_island_system_rules_mode_runs():
    system = pmh.IslandSystem(
        islands=[
            pmh.Island("de", config={"population_size": 8}, label="DE"),
            pmh.Island("ga", config={"population_size": 8}, label="GA"),
        ],
        orchestration=pmh.OrchestrationConfig(mode="rules", checkpoint_interval=2),
        max_steps=3,
        seed=13,
    )
    result = system.optimize(sphere, [-5, -5], [5, 5], mode="rules")
    assert result.best_fitness is not None
    assert result.controller_mode == "rules"


def test_phase1_public_exports_present():
    for name in (
        "Island",
        "IslandConfig",
        "IslandSystem",
        "TopologyConfig",
        "MigrationConfig",
        "ExecutionConfig",
        "OrchestrationConfig",
    ):
        assert hasattr(pmh, name), f"missing public symbol: {name}"
