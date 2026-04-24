"""
pyMetaheuristic src — HyperparameterTuner and BenchmarkRunner
=============================================================

Feature 7 — HyperparameterTuner
---------------------------------
Grid-search or random-search over algorithm hyperparameters.

Usage::

    from pymetaheuristic.src.tuner import HyperparameterTuner, Termination

    tuner = HyperparameterTuner(
        algorithm       = "pso",
        param_grid      = {"swarm_size": [20, 50], "w": [0.5, 0.9]},
        target_function = sphere,
        min_values      = [-5] * 5,
        max_values      = [ 5] * 5,
        termination     = Termination(max_steps=200),
        n_trials        = 5,
        objective       = "min",
        seed            = 42,
    )
    df = tuner.run()        # returns pandas DataFrame (or list of dicts)
    print(tuner.best_params)

Feature 8 — BenchmarkRunner
-----------------------------
Run multiple algorithms × multiple problems × multiple trials.

Usage::

    from pymetaheuristic.src.tuner import BenchmarkRunner, Termination

    runner = BenchmarkRunner(
        algorithms      = ["pso", "de", "ga"],
        problems        = [
            {"name": "sphere",  "target_function": sphere,  "min_values": [-5]*5, "max_values": [5]*5},
            {"name": "easom",   "target_function": easom,   "min_values": [-5,-5], "max_values": [5,5]},
        ],
        termination     = Termination(max_steps=300),
        n_trials        = 10,
        seed            = 0,
        n_jobs          = 4,
    )
    df = runner.run()       # returns pandas DataFrame (or list of dicts)
"""

from __future__ import annotations

import ast
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_single(
    algorithm: str,
    target_function: Callable,
    min_values,
    max_values,
    params: dict,
    termination,
    objective: str,
    seed: int | None,
    store_history: bool = False,
) -> dict[str, Any]:
    """Run one trial and return a flat result dict."""
    from .api import optimize

    t0 = time.perf_counter()
    try:
        result = optimize(
            algorithm       = algorithm,
            target_function = target_function,
            min_values      = min_values,
            max_values      = max_values,
            objective       = objective,
            termination     = termination,
            seed            = seed,
            store_history   = store_history,
            verbose         = False,
            **params,
        )
        elapsed = time.perf_counter() - t0
        return {
            "algorithm":    algorithm,
            "best_fitness": result.best_fitness,
            "steps":        result.steps,
            "evaluations":  result.evaluations,
            "termination":  result.termination_reason,
            "elapsed_s":    round(elapsed, 4),
            "error":        None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "algorithm":    algorithm,
            "best_fitness": None,
            "steps":        None,
            "evaluations":  None,
            "termination":  None,
            "elapsed_s":    round(elapsed, 4),
            "error":        str(exc),
        }


def _run_tuner_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    Worker for a single HyperparameterTuner trial.

    Must be module-level to remain pickleable by ProcessPoolExecutor.
    """
    row = _run_single(
        algorithm       = job["algorithm"],
        target_function = job["target_function"],
        min_values      = job["min_values"],
        max_values      = job["max_values"],
        params          = job["params"],
        termination     = job["termination"],
        objective       = job["objective"],
        seed            = job["seed"],
        store_history   = False,
    )
    row.update(job["params"])
    row["trial"] = job["trial"]
    row["_order"] = job["idx"]
    return row


def _run_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    """
    Worker for a single BenchmarkRunner job.

    Must be module-level to remain pickleable by ProcessPoolExecutor.
    """
    from .api import optimize

    idx         = job["idx"]
    algorithm   = job["algorithm"]
    problem     = job["problem"]
    trial       = job["trial"]
    termination = job["termination"]
    seed        = job["seed"]

    prob_name = problem.get("name", "problem")
    fn        = problem["target_function"]
    lb        = problem["min_values"]
    ub        = problem["max_values"]
    obj       = problem.get("objective", "min")
    constr    = problem.get("constraints")
    handler   = problem.get("constraint_handler")

    extra: dict[str, Any] = {}
    if constr:
        extra["constraints"] = constr
    if handler:
        extra["constraint_handler"] = handler

    t0 = time.perf_counter()
    try:
        result = optimize(
            algorithm       = algorithm,
            target_function = fn,
            min_values      = lb,
            max_values      = ub,
            objective       = obj,
            termination     = termination,
            seed            = seed,
            store_history   = False,
            verbose         = False,
            **extra,
        )
        elapsed = time.perf_counter() - t0
        return {
            "_order":       idx,
            "algorithm":    algorithm,
            "problem":      prob_name,
            "trial":        trial,
            "best_fitness": result.best_fitness,
            "steps":        result.steps,
            "evaluations":  result.evaluations,
            "elapsed_s":    round(elapsed, 4),
            "error":        None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "_order":       idx,
            "algorithm":    algorithm,
            "problem":      prob_name,
            "trial":        trial,
            "best_fitness": None,
            "steps":        None,
            "evaluations":  None,
            "elapsed_s":    round(elapsed, 4),
            "error":        str(exc),
        }


# ===========================================================================
# HyperparameterTuner
# ===========================================================================

class HyperparameterTuner:
    """
    Grid or random search over algorithm hyperparameters.

    Parameters
    ----------
    algorithm       : Algorithm identifier (e.g. "pso").
    param_grid      : Dict mapping parameter names to lists of candidate values.
                      Example: ``{"swarm_size": [20, 50, 100], "w": [0.4, 0.7, 0.9]}``.
    target_function : Objective function.
    min_values      : Lower bounds.
    max_values      : Upper bounds.
    termination     : ``Termination`` instance or dict. Applied to every trial.
    n_trials        : Number of independent runs per parameter combination.
    objective       : "min" or "max".
    seed            : Base random seed (incremented per trial for reproducibility).
    search          : "grid" (all combinations) or "random" (random sample of *max_configs*).
    max_configs     : Maximum number of parameter combinations when ``search="random"``.
    n_jobs          : Number of parallel workers.
                      1 = serial (default and safest).

    Notes
    -----
    For multiprocessing to work reliably, target functions should be defined at
    module top level (not lambdas or nested functions), so they can be pickled
    by worker processes.
    """

    def __init__(
        self,
        algorithm: str,
        param_grid: dict[str, list],
        target_function: Callable,
        min_values,
        max_values,
        termination=None,
        n_trials: int = 3,
        objective: str = "min",
        seed: int | None = 42,
        search: str = "grid",
        max_configs: int = 20,
        n_jobs: int = 1,
    ) -> None:
        self.algorithm       = algorithm
        self.param_grid      = {k: list(v) for k, v in param_grid.items()}
        self.target_function = target_function
        self.min_values      = list(min_values)
        self.max_values      = list(max_values)
        self.termination     = termination
        self.n_trials        = max(1, int(n_trials))
        self.objective       = objective
        self.seed            = seed
        self.search          = search
        self.max_configs     = max(1, int(max_configs))
        self.n_jobs          = max(1, int(n_jobs))

        self._rows: list[dict] = []
        self.best_params: dict | None = None
        self.best_fitness: float | None = None

    def _all_combinations(self) -> list[dict]:
        keys   = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combos = [dict(zip(keys, combo)) for combo in product(*values)]
        if self.search == "random" and len(combos) > self.max_configs:
            rng    = random.Random(self.seed)
            combos = rng.sample(combos, self.max_configs)
        return combos

    def _build_jobs(self) -> list[dict]:
        jobs: list[dict] = []
        combos = self._all_combinations()
        counter = 0

        for combo in combos:
            for t in range(self.n_trials):
                seed = (self.seed + counter) if self.seed is not None else None
                jobs.append({
                    "idx":             counter,
                    "algorithm":       self.algorithm,
                    "target_function": self.target_function,
                    "min_values":      self.min_values,
                    "max_values":      self.max_values,
                    "params":          dict(combo),
                    "termination":     self.termination,
                    "objective":       self.objective,
                    "trial":           t + 1,
                    "seed":            seed,
                })
                counter += 1

        return jobs

    def _update_best_from_rows(self, rows: list[dict]) -> None:
        self.best_params = None
        self.best_fitness = None

        key_cols = list(self.param_grid.keys())
        combos_seen: dict[str, list[float]] = {}
        combo_map: dict[str, dict[str, Any]] = {}

        for row in rows:
            f = row.get("best_fitness")
            if f is None:
                continue
            combo = {k: row.get(k) for k in key_cols}
            combo_key = str(combo)
            combos_seen.setdefault(combo_key, []).append(f)
            combo_map[combo_key] = combo

        for combo_key, fitnesses in combos_seen.items():
            mean_f = float(np.mean(fitnesses))
            is_better = (
                self.best_fitness is None
                or (self.objective == "min" and mean_f < self.best_fitness)
                or (self.objective == "max" and mean_f > self.best_fitness)
            )
            if is_better:
                self.best_fitness = mean_f
                self.best_params = dict(combo_map[combo_key])

    def run(self, show_progress: bool = True):
        """
        Execute the search and return results.

        Returns
        -------
        pandas.DataFrame if pandas is available, else list[dict].
        """
        combos = self._all_combinations()
        jobs = self._build_jobs()
        total = len(jobs)

        if show_progress:
            print(
                f"[HyperparameterTuner] {self.algorithm}: "
                f"{len(combos)} configs × {self.n_trials} trials = {total} runs "
                f"| n_jobs={self.n_jobs}"
            )

        all_rows: list[dict] = []

        # ---------------------------------------------------------------
        # Serial path
        # ---------------------------------------------------------------
        if self.n_jobs == 1:
            for i, job in enumerate(jobs, start=1):
                row = _run_tuner_job(job)
                all_rows.append(row)
                if show_progress:
                    params_str = ", ".join(f"{k}={v}" for k, v in job["params"].items())
                    print(f"  [{i}/{total}] {params_str} | best={row['best_fitness']}")

        # ---------------------------------------------------------------
        # Parallel path
        # ---------------------------------------------------------------
        else:
            completed = 0
            ordered_rows: dict[int, dict] = {}

            with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
                future_to_job = {
                    ex.submit(_run_tuner_job, job): job
                    for job in jobs
                }

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "_order":       job["idx"],
                            "algorithm":    job["algorithm"],
                            "best_fitness": None,
                            "steps":        None,
                            "evaluations":  None,
                            "termination":  None,
                            "elapsed_s":    None,
                            "error":        f"worker crash: {exc}",
                            "trial":        job["trial"],
                            **job["params"],
                        }
                    ordered_rows[row["_order"]] = row
                    completed += 1

                    if show_progress:
                        params_str = ", ".join(f"{k}={v}" for k, v in job["params"].items())
                        print(f"  [{completed}/{total}] {params_str} | best={row['best_fitness']}")

            all_rows = [ordered_rows[i] for i in sorted(ordered_rows)]

        for row in all_rows:
            row.pop("_order", None)

        self._rows = all_rows
        self._update_best_from_rows(all_rows)

        if show_progress:
            print(f"\n[HyperparameterTuner] Best config: {self.best_params}")
            print(f"[HyperparameterTuner] Best mean fitness: {self.best_fitness}")

        try:
            import pandas as pd
            return pd.DataFrame(all_rows)
        except ImportError:
            return all_rows

    def summary(self):
        """Return a per-configuration summary (mean / std / min / max fitness)."""
        if not self._rows:
            return []

        combos_seen: dict[str, list[float]] = {}
        key_cols = list(self.param_grid.keys())
        for row in self._rows:
            combo_key = str({k: row.get(k) for k in key_cols})
            f         = row.get("best_fitness")
            if f is not None:
                combos_seen.setdefault(combo_key, []).append(f)

        summaries = []
        for combo_key, fitnesses in combos_seen.items():
            combo = ast.literal_eval(combo_key)
            summaries.append({
                **combo,
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness":  float(np.std(fitnesses)),
                "min_fitness":  float(np.min(fitnesses)),
                "max_fitness":  float(np.max(fitnesses)),
                "n_trials":     len(fitnesses),
            })

        summaries.sort(
            key=lambda r: r["mean_fitness"],
            reverse=(self.objective == "max"),
        )
        try:
            import pandas as pd
            return pd.DataFrame(summaries)
        except ImportError:
            return summaries


# ===========================================================================
# BenchmarkRunner
# ===========================================================================

class BenchmarkRunner:
    """
    Run multiple algorithms × multiple problems × multiple trials.

    Parameters
    ----------
    algorithms  : List of algorithm IDs (e.g. ["pso", "de", "gwo"]).
    problems    : List of problem dicts, each with keys:
                    - "name"              : str (human-readable label)
                    - "target_function"   : callable
                    - "min_values"        : sequence
                    - "max_values"        : sequence
                    - "objective"         : "min" | "max"  (optional, default "min")
                    - "constraints"       : list of callables (optional)
                    - "constraint_handler": str (optional)
    termination : ``Termination`` instance or dict. Applied to every run.
    n_trials    : Number of independent runs per (algorithm, problem) pair.
    seed        : Base random seed.
    n_jobs      : Parallel workers (1 = serial, default).

    Notes
    -----
    For multiprocessing to work reliably, target functions and constraint
    functions should be defined at module top level (not lambdas or nested
    functions), so they can be pickled by worker processes.

    Example
    -------
    ::

        runner = BenchmarkRunner(
            algorithms  = ["pso", "de", "gwo"],
            problems    = [
                {"name": "sphere",  "target_function": sphere,
                 "min_values": [-5]*10, "max_values": [5]*10},
            ],
            termination = Termination(max_steps=500),
            n_trials    = 10,
            seed        = 0,
            n_jobs      = 4,
        )
        df = runner.run()
        print(df.groupby(["algorithm", "problem"])["best_fitness"].agg(["mean", "std"]))
    """

    def __init__(
        self,
        algorithms: list[str],
        problems: list[dict],
        termination=None,
        n_trials: int = 5,
        seed: int | None = 0,
        n_jobs: int = 1,
    ) -> None:
        self.algorithms  = list(algorithms)
        self.problems    = list(problems)
        self.termination = termination
        self.n_trials    = max(1, int(n_trials))
        self.seed        = seed
        self.n_jobs      = max(1, int(n_jobs))

        self._rows: list[dict] = []

    def _build_jobs(self) -> list[dict]:
        jobs: list[dict] = []
        counter = 0

        for prob in self.problems:
            for alg in self.algorithms:
                for t in range(self.n_trials):
                    seed = (self.seed + counter) if self.seed is not None else None
                    jobs.append({
                        "idx":         counter,
                        "algorithm":   alg,
                        "problem":     prob,
                        "trial":       t + 1,
                        "termination": self.termination,
                        "seed":        seed,
                    })
                    counter += 1

        return jobs

    def run(self, show_progress: bool = True):
        """
        Execute all (algorithm, problem, trial) combinations.

        Returns
        -------
        pandas.DataFrame if pandas is available, else list[dict].
        """
        jobs = self._build_jobs()
        total = len(jobs)

        if show_progress:
            print(
                f"[BenchmarkRunner] {len(self.algorithms)} algorithms × "
                f"{len(self.problems)} problems × {self.n_trials} trials "
                f"= {total} runs | n_jobs={self.n_jobs}"
            )

        all_rows: list[dict] = []

        # ---------------------------------------------------------------
        # Serial path
        # ---------------------------------------------------------------
        if self.n_jobs == 1:
            for i, job in enumerate(jobs, start=1):
                row = _run_benchmark_job(job)
                all_rows.append(row)
                if show_progress:
                    print(
                        f"  [{i}/{total}] {row['algorithm']} | {row['problem']} | "
                        f"trial={row['trial']} | best={row['best_fitness']}"
                    )

        # ---------------------------------------------------------------
        # Parallel path
        # ---------------------------------------------------------------
        else:
            completed = 0
            ordered_rows: dict[int, dict] = {}

            with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
                future_to_job = {
                    ex.submit(_run_benchmark_job, job): job
                    for job in jobs
                }

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "_order":       job["idx"],
                            "algorithm":    job["algorithm"],
                            "problem":      job["problem"].get("name", "problem"),
                            "trial":        job["trial"],
                            "best_fitness": None,
                            "steps":        None,
                            "evaluations":  None,
                            "elapsed_s":    None,
                            "error":        f"worker crash: {exc}",
                        }
                    ordered_rows[row["_order"]] = row
                    completed += 1

                    if show_progress:
                        print(
                            f"  [{completed}/{total}] {row['algorithm']} | "
                            f"{row['problem']} | trial={row['trial']} | "
                            f"best={row['best_fitness']}"
                        )

            all_rows = [ordered_rows[i] for i in sorted(ordered_rows)]

        for row in all_rows:
            row.pop("_order", None)

        self._rows = all_rows
        try:
            import pandas as pd
            return pd.DataFrame(all_rows)
        except ImportError:
            return all_rows

    def summary(self):
        """
        Aggregate mean / std / min / max fitness per (algorithm, problem).

        Returns
        -------
        pandas.DataFrame or list[dict], sorted by (problem, mean_fitness).
        """
        if not self._rows:
            return []

        from collections import defaultdict
        grouped: dict[tuple, list] = defaultdict(list)
        for row in self._rows:
            if row.get("best_fitness") is not None:
                grouped[(row["algorithm"], row["problem"])].append(row["best_fitness"])

        summaries = []
        for (alg, prob), fitnesses in grouped.items():
            summaries.append({
                "algorithm":    alg,
                "problem":      prob,
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness":  float(np.std(fitnesses)),
                "min_fitness":  float(np.min(fitnesses)),
                "max_fitness":  float(np.max(fitnesses)),
                "n_trials":     len(fitnesses),
            })

        summaries.sort(key=lambda r: (r["problem"], r["mean_fitness"]))
        try:
            import pandas as pd
            return pd.DataFrame(summaries)
        except ImportError:
            return summaries
