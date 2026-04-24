import numpy as np

from pymetaheuristic import cooperative_optimize


def easom(x=[0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


if __name__ == "__main__":
    res = cooperative_optimize(
        islands=[
            {"algorithm": "pso", "config": {"swarm_size": 25}},
            {"algorithm": "ga", "config": {"population_size": 30}},
            {"algorithm": "sa", "config": {"temperature_iterations": 20}},
            {"algorithm": "abco", "config": {"swarm_size": 20}},
        ],
        target_function=easom,
        min_values=(-5, -5),
        max_values=(5, 5),
        max_steps=20,
        migration_interval=5,
        migration_size=2,
        topology="ring",
        seed=42,
    )

    print("Cooperative best:", res.best_fitness)
    print("Events:", len(res.events))
    for label, result in res.island_results.items():
        print(label, result.best_fitness)
