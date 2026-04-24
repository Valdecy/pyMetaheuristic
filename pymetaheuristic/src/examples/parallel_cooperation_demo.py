from pymetaheuristic import cooperative_optimize, get_test_function

ackley = get_test_function("ackley")


def main():
    result = cooperative_optimize(
        islands=[
            {"label": "DE_1", "algorithm": "de", "config": {"population_size": 15}},
            {"label": "PSO_1", "algorithm": "pso", "config": {"swarm_size": 15}},
            {"label": "SA_1", "algorithm": "sa", "config": {"temperature": 10.0}},
        ],
        target_function=ackley,
        min_values=[-5.0, -5.0],
        max_values=[5.0, 5.0],
        max_steps=20,
        migration_interval=5,
        migration_size=1,
        execution_backend="process",
        n_jobs=3,
        verbose=True,
    )
    print("best fitness:", result.best_fitness)
    print("backend used:", result.metadata.get("execution_backend_used"))
    print("warning:", result.metadata.get("parallel_warning"))


if __name__ == "__main__":
    main()
