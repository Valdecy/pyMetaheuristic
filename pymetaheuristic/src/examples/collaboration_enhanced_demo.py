from __future__ import annotations

from pymetaheuristic import (
    cooperative_optimize,
    plot_collaboration_network,
    plot_island_dynamics,
    export_island_telemetry_csv,
)


def sphere(x):
    return sum(v * v for v in x)


if __name__ == "__main__":
    result = cooperative_optimize(
        islands=[
            {"label": "DE", "algorithm": "de", "config": {"population_size": 15}},
            {"label": "GA", "algorithm": "ga", "config": {"population_size": 15}},
            {"label": "CEM", "algorithm": "cem", "config": {"size": 15}},
            {"label": "PSO", "algorithm": "pso", "config": {"swarm_size": 15}},
        ],
        target_function=sphere,
        min_values=[-5, -5],
        max_values=[5, 5],
        objective="min",
        max_steps=20,
        migration_interval=3,
        topology="wheel",
        migration_policy="adaptive",
        donor_strategy="improving",
        receiver_strategy="stagnated",
        adaptive_checkpointing=True,
        checkpoint_strategy="adaptive",
        seed=7,
    )
    print("Best fitness:", result.best_fitness)
    export_island_telemetry_csv(result, "island_telemetry.csv")
    plot_island_dynamics(result, metric="best_fitness", filepath="island_dynamics.html")
    plot_collaboration_network(result, filepath="collaboration_network.html")
