import pymetaheuristic as pmh


def sphere(x):
    return float(sum(v * v for v in x))


def test_cooperative_result_phase2_diagnostics():
    result = pmh.cooperative_optimize(
        islands=[
            {"algorithm": "pso", "config": {"swarm_size": 8}, "label": "pso"},
            {"algorithm": "ga", "config": {"population_size": 8}, "label": "ga"},
        ],
        target_function=sphere,
        min_values=(-3, -3),
        max_values=(3, 3),
        max_steps=4,
        migration_interval=2,
        migration_size=1,
        topology="ring",
        seed=7,
    )
    matrix = result.migration_matrix()
    assert set(matrix) == {"pso", "ga"}
    assert result.topology_summary()["n_islands"] == 2
    contribution = result.island_contribution()
    assert set(contribution) == {"pso", "ga"}
    assert all("rank" in row for row in contribution.values())
    roles = result.island_roles()
    assert set(roles) == {"pso", "ga"}
    effectiveness = result.action_effectiveness()
    assert effectiveness["total_actions"] == len(result.events)
    assert "migration" in effectiveness["by_type"]
    summary = result.diagnostics_summary()
    assert "migration_matrix" in summary


def test_orchestrated_result_phase2_diagnostics():
    config = pmh.CollaborativeConfig(
        orchestration=pmh.OrchestrationSpec(
            mode="rules",
            checkpoint_interval=2,
            warmup_checkpoints=0,
            max_actions_per_checkpoint=2,
        ),
        rules=pmh.RulesConfig(stagnation_threshold=1, low_diversity_threshold=0.05),
    )
    result = pmh.orchestrated_optimize(
        islands=[
            {"algorithm": "pso", "config": {"swarm_size": 8}, "label": "pso"},
            {"algorithm": "ga", "config": {"population_size": 8}, "label": "ga"},
        ],
        target_function=sphere,
        min_values=(-3, -3),
        max_values=(3, 3),
        max_steps=5,
        seed=8,
        config=config,
    )
    assert result.checkpoints
    assert isinstance(result.migration_matrix(), dict)
    assert result.topology_summary()["n_islands"] == 2
    assert set(result.island_contribution()) == {"pso", "ga"}
    assert set(result.island_roles()) == {"pso", "ga"}
    effectiveness = result.action_effectiveness()
    assert "by_type" in effectiveness


def test_root_diagnostic_functions_match_methods():
    result = pmh.cooperative_optimize(
        islands=[
            {"algorithm": "pso", "config": {"swarm_size": 6}, "label": "pso"},
            {"algorithm": "ga", "config": {"population_size": 6}, "label": "ga"},
        ],
        target_function=sphere,
        min_values=(-2, -2),
        max_values=(2, 2),
        max_steps=3,
        migration_interval=1,
        seed=9,
    )
    assert pmh.migration_matrix(result) == result.migration_matrix()
    assert pmh.topology_summary(result) == result.topology_summary()
    assert pmh.diagnostics_summary(result)["island_roles"] == result.island_roles()
