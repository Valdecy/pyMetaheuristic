import numpy as np

from pymetaheuristic import optimize, summarize_result


def easom(x=[0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


if __name__ == "__main__":
    result = optimize(
        algorithm="pso",
        target_function=easom,
        min_values=(-5, -5),
        max_values=(5, 5),
        max_steps=20,
        seed=42,
        swarm_size=30,
        store_history=True,
        store_population_snapshots=True,
    )

    print("Best:", result.best_fitness)
    print("History points:", len(result.history))
    print(summarize_result(result))
