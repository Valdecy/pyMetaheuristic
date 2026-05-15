import numpy as np

import pymetaheuristic as pmh


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def _islands():
    return [
        {"label": "pso", "algorithm": "pso", "config": {"swarm_size": 8}},
        {"label": "ga", "algorithm": "ga", "config": {"population_size": 8}},
        {"label": "sa", "algorithm": "sa", "config": {"temperature": 5.0}},
    ]


def test_bandit_orchestration_mode_runs_and_records_decisions():
    cfg = pmh.CollaborativeConfig(
        orchestration=pmh.OrchestrationSpec(
            mode="bandit",
            checkpoint_interval=2,
            warmup_checkpoints=1,
            max_actions_per_checkpoint=2,
        ),
        rules=pmh.RulesConfig(stagnation_threshold=1, low_diversity_threshold=0.5),
        bandit=pmh.BanditConfig(policy="ucb", exploration=0.5),
    )
    result = pmh.orchestrated_optimize(
        islands=_islands(),
        target_function=sphere,
        min_values=(-5, -5),
        max_values=(5, 5),
        max_steps=6,
        seed=123,
        config=cfg,
    )
    assert result.controller_mode == "bandit"
    assert result.best_fitness is not None
    assert result.decisions
    assert all(decision.controller_mode == "bandit" for decision in result.decisions)
    assert "bandit_updates" in result.decisions[-1].diagnostics


def test_portfolio_adaptive_mode_runs_and_reports_phase():
    cfg = pmh.CollaborativeConfig(
        orchestration=pmh.OrchestrationSpec(
            mode="portfolio_adaptive",
            checkpoint_interval=2,
            warmup_checkpoints=1,
            max_actions_per_checkpoint=2,
        ),
        rules=pmh.RulesConfig(stagnation_threshold=1, low_diversity_threshold=0.5),
        portfolio=pmh.PortfolioConfig(stagnation_threshold=1, diversity_threshold=0.5),
    )
    result = pmh.orchestrated_optimize(
        islands=_islands(),
        target_function=sphere,
        min_values=(-5, -5),
        max_values=(5, 5),
        max_steps=6,
        seed=123,
        config=cfg,
    )
    assert result.controller_mode == "portfolio_adaptive"
    assert result.best_fitness is not None
    assert result.decisions
    assert all(decision.controller_mode == "portfolio_adaptive" for decision in result.decisions)
    assert "phase" in result.decisions[-1].diagnostics


def test_island_system_accepts_new_phase4_modes():
    system = pmh.IslandSystem(
        islands=_islands(),
        orchestration={"checkpoint_interval": 2, "warmup_checkpoints": 1, "max_actions_per_checkpoint": 1},
        rules={"stagnation_threshold": 1, "low_diversity_threshold": 0.5},
        max_steps=6,
        seed=321,
    )
    bandit_result = system.optimize(sphere, (-5, -5), (5, 5), mode="bandit")
    portfolio_result = system.optimize(sphere, (-5, -5), (5, 5), mode="portfolio_adaptive")
    assert bandit_result.controller_mode == "bandit"
    assert portfolio_result.controller_mode == "portfolio_adaptive"
