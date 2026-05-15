import os

import matplotlib
matplotlib.use("Agg")

import pymetaheuristic as pmh


def test_benchmark_study_algorithm_core_statistics_and_plots(tmp_path):
    problems = pmh.ProblemSuite.from_names(["sphere", "rastrigin"], dimensions=[2])
    study = pmh.BenchmarkStudy(
        algorithms=[
            {"algorithm": "pso", "config": {"swarm_size": 5}},
            {"algorithm": "ga", "config": {"population_size": 8}},
            {"algorithm": "sa", "config": {"temperature": 1.0}},
        ],
        problems=problems,
        n_trials=2,
        max_steps=5,
        max_evaluations=200,
        seed=7,
        target_tolerance=1.0e-4,
    )
    result = study.run()
    assert len(result) == 12
    df = result.to_dataframe()
    assert {"candidate", "problem", "trial", "best_fitness", "error_to_optimum"}.issubset(df.columns)
    assert df["candidate"].nunique() == 3

    summary = result.summary()
    ranks = result.rank_table()
    friedman = result.friedman_test()
    wilcoxon = result.wilcoxon_pairwise()
    assert not summary.empty
    assert not ranks.empty
    assert friedman["n_candidates"] == 3
    assert len(wilcoxon) == 3

    # Plot methods should return axes.
    assert result.plot_convergence() is not None
    assert result.plot_ecdf() is not None
    assert result.plot_performance_profile() is not None
    assert result.plot_rank_heatmap() is not None

    saved = tmp_path / "study.json"
    result.save(saved)
    loaded = pmh.load_benchmark(saved)
    assert len(loaded) == len(result)


def test_benchmark_study_supports_island_system_candidate():
    system = pmh.IslandSystem(
        islands=[
            pmh.Island("pso", config={"swarm_size": 5}, label="pso"),
            pmh.Island("ga", config={"population_size": 8}, label="ga"),
        ],
        topology=pmh.TopologyConfig(name="ring"),
        migration=pmh.MigrationConfig(interval=2, size=1),
        orchestration=pmh.OrchestrationConfig(mode="rules", checkpoint_interval=2, warmup_checkpoints=1),
        max_steps=4,
        seed=11,
    )
    study = pmh.BenchmarkStudy(
        candidates=[{"name": "rules_system", "type": "island_system", "system": system, "mode": "rules"}],
        algorithms=[{"algorithm": "pso", "config": {"swarm_size": 5}}],
        problems=pmh.ProblemSuite.from_names(["sphere"], dimensions=[2]),
        n_trials=1,
        max_steps=4,
        seed=3,
    )
    result = study.run()
    df = result.to_dataframe()
    assert set(df["candidate"]) == {"rules_system", "pso"}
    row = df[df["candidate"] == "rules_system"].iloc[0]
    assert row["candidate_type"] == "island_system"
    assert row["controller_mode"] == "rules"
    assert row["n_checkpoints"] >= 1
