from pymetaheuristic import optimize


def sphere(x):
    return x[0] ** 2 + x[1] ** 2


def g1(x):
    return 1.0 - x[0] - x[1]


if __name__ == "__main__":
    result = optimize(
        algorithm="pso",
        target_function=sphere,
        min_values=(-2, -2),
        max_values=(2, 2),
        constraints=[g1],
        constraint_handler="deb",
        max_steps=50,
        seed=42,
        swarm_size=30,
        store_history=True,
    )

    print("Best position:", result.best_position)
    print("Effective score:", result.best_fitness)
    print("Raw objective:", result.metadata.get("best_raw_fitness"))
    print("Violation:", result.metadata.get("best_violation"))
    print("Feasible:", result.metadata.get("best_is_feasible"))
