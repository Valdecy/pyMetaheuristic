import numpy as np

from pymetaheuristic import CollaborativeConfig, OrchestrationSpec, RulesConfig, orchestrated_optimize


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    cfg = CollaborativeConfig(
        orchestration=OrchestrationSpec(mode="rules", checkpoint_interval=5),
        rules=RulesConfig(
            stagnation_threshold=4,
            low_diversity_threshold=0.08,
            perturbation_sigma=0.05,
        ),
    )

    res = orchestrated_optimize(
        islands=[
            {"algorithm": "pso", "config": {"swarm_size": 20}},
            {"algorithm": "ga", "config": {"population_size": 20}},
            {"algorithm": "sa", "config": {"temperature_iterations": 20}},
        ],
        target_function=rastrigin,
        min_values=[-5.12, -5.12],
        max_values=[5.12, 5.12],
        max_steps=20,
        config=cfg,
        seed=42,
    )

    print("Best fitness:", res.best_fitness)
    print("Controller mode:", res.controller_mode)
    print("Decisions:", len(res.decisions))
    if res.decisions:
        print("Last reasoning:", res.decisions[-1].reasoning)
