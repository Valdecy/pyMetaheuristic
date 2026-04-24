from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pymetaheuristic  # noqa: E402


def sphere(x):
    return sum(v * v for v in x)


def test_enhanced_cooperation_smoke_and_replay():
    result = pymetaheuristic.cooperative_optimize(
        islands=[
            {"label": "A1", "algorithm": "de", "config": {"population_size": 8}},
            {"label": "A2", "algorithm": "ga", "config": {"population_size": 8}},
            {"label": "A3", "algorithm": "cem", "config": {"size": 8}},
        ],
        target_function=sphere,
        min_values=[-5, -5],
        max_values=[5, 5],
        objective="min",
        max_steps=4,
        migration_interval=2,
        topology="wheel",
        migration_policy="push_pull",
        donor_strategy="improving",
        receiver_strategy="stagnated",
        adaptive_checkpointing=True,
        checkpoint_strategy="adaptive",
        seed=3,
    )
    assert result.best_fitness is not None
    assert result.metadata["topology"] == "wheel"
    assert result.metadata["migration_policy"] == "push_pull"
    assert result.metadata["adaptive_checkpointing"] is True
    assert len(result.island_telemetry) == 3
    assert all(len(v) >= 1 for v in result.island_telemetry.values())
    assert isinstance(result.replay_manifest, dict) and result.replay_manifest["runner"] == "cooperative"

    replay = pymetaheuristic.replay_cooperative_result(result, target_function=sphere)
    assert replay.best_fitness == result.best_fitness
    assert len(replay.events) == len(result.events)


def test_public_collaboration_exports_present():
    for name in (
        "replay_cooperative_result",
        "plot_island_dynamics",
        "plot_collaboration_network",
        "export_island_telemetry_csv",
        "export_replay_manifest_json",
        "summarize_cooperative_result",
    ):
        assert hasattr(pymetaheuristic, name), f"missing public symbol: {name}"
