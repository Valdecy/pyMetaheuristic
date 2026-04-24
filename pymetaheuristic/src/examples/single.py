from __future__ import annotations

from pathlib import Path

from pymetaheuristic import optimize
from pymetaheuristic.src.graphs import (
    compare_convergence,
    plot_benchmark_summary,
    plot_convergence,
    plot_function_contour,
    plot_function_surface,
    plot_population_snapshot,
)
from pymetaheuristic.src.test_functions import ackley, easom, rastrigin

OUTPUT_DIR = Path("single_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_single_demo():
    problems = {
        "ackley": {
            "func": ackley,
            "bounds": ([-5, -5], [5, 5]),
            "algorithm": "pso",
            "params": {"swarm_size": 30},
            "steps": 60,
        },
        "easom": {
            "func": easom,
            "bounds": ([-10, -10], [10, 10]),
            "algorithm": "ga",
            "params": {"population_size": 40},
            "steps": 60,
        },
        "rastrigin": {
            "func": rastrigin,
            "bounds": ([-5.12, -5.12], [5.12, 5.12]),
            "algorithm": "gwo",
            "params": {"pack_size": 30},
            "steps": 60,
        },
    }

    results = {}
    for name, spec in problems.items():
        min_values, max_values = spec["bounds"]
        result = optimize(
            algorithm=spec["algorithm"],
            target_function=spec["func"],
            min_values=min_values,
            max_values=max_values,
            max_steps=spec["steps"],
            seed=42,
            store_history=True,
            store_population_snapshots=True,
            snapshot_interval=5,
            **spec["params"],
        )
        results[name] = result

        print(f"\n{name.upper()} :: {spec['algorithm']}")
        print("  best_position:", [round(v, 6) for v in result.best_position])
        print("  best_fitness:", round(result.best_fitness, 10))
        print("  steps:", result.steps)
        print("  evaluations:", result.evaluations)

        plot_convergence(result, filepath=OUTPUT_DIR / f"{name}_convergence.png")
        plot_function_contour(
            spec["func"],
            min_values,
            max_values,
            best_position=result.best_position,
            filepath=OUTPUT_DIR / f"{name}_contour.png",
            title=f"{name.title()} contour",
            grid_points=150,
        )
        plot_function_surface(
            spec["func"],
            min_values,
            max_values,
            best_position=result.best_position,
            filepath=OUTPUT_DIR / f"{name}_surface.png",
            title=f"{name.title()} surface",
            grid_points=80,
        )
        try:
            plot_population_snapshot(result, filepath=OUTPUT_DIR / f"{name}_population.png")
        except Exception:
            pass

    compare_convergence(
        list(results.values()),
        labels=[f"{k}:{problems[k]['algorithm']}" for k in results],
        filepath=OUTPUT_DIR / "comparison_convergence.png",
    )
    plot_benchmark_summary(results, filepath=OUTPUT_DIR / "benchmark_summary.png")
    return results


if __name__ == "__main__":
    run_single_demo()
