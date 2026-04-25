
<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/logo_pmh_.png" alt="Logo" width="300" height="300"/>
</p>

# pymetaheuristic

## A. **Version Note**

This README targets **pymetaheuristic-v5+**. It can be installed with:

```bash
pip install pymetaheuristic
```

For legacy, the old library can still be installed with:

```bash
pip install pymetaheuristic==1.9.5
```

## B. **Summary** 

1. [Introduction](#1-introduction)
2. [Installation and Package Overview](#2-installation-and-package-overview)
   - [2.1 Installation](#21-installation)
   - [2.2 Package Overview](#22-package-overview)
   - [2.3 Optimization, Telemetry, Export, and Plotting Example](#23-optimization-telemetry-export-and-plotting-example) --- [[Colab Demo]](https://colab.research.google.com/drive/11lPwLf13mav4UWSqNolMaKPbqt2lsq4x?usp=sharing) ---
   - [2.4 Termination Criteria](#24-termination-criteria)
   - [2.5 Constraint Handling Example](#25-constraint-handling-example) --- [[Colab Demo]](https://colab.research.google.com/drive/1T8ltBcunERKd7N3q12rW2MdnzSTdsOGs?usp=sharing) ---  
   - [2.6 Cooperative Multi-island Example](#26-cooperative-multi-island-example) --- [[Colab Demo] ](https://colab.research.google.com/drive/1DteFWUIqpZZNV4nUM7FGAHfqZN5Vabse?usp=sharing) ---
   - [2.7 Orchestrated Cooperation Example](#27-orchestrated-cooperation-example) --- [[Colab Demo]](https://colab.research.google.com/drive/1j4RbtBjFyxAVuVTMNaJw9ALbREWiIBmn?usp=sharing) --- 
   - [2.8 Chaotic Maps and Transfer Functions](#28-chaotic-maps-and-transfer-functions) --- [[Colab Demo]](https://colab.research.google.com/drive/1cvrahJ5Bp4E4vU7I-O6Uqru9SK2hxMXX?usp=sharing) ---
   - [2.9 Hyperparameter Tuner](#29-hyperparameter-tuner) --- [[Colab Demo] ](https://colab.research.google.com/drive/13pZQyrMDyegRAcYUJRO6cSwvQ7pZvDKs?usp=sharing) ---
   - [2.10 Save, Load, and Checkpoint](#210-save-load-and-checkpoint)
3. [Algorithm Details](#3-algorithm-details)
4. [Test Functions](#4-test-functions) --- [[Colab Demo]](https://colab.research.google.com/drive/132-yqoaJKkJ4gf6yqjrV1siXVvZ3ZgE7?usp=sharing) ---
5. [Other Libraries](#5-other-libraries)

## 1. **Introduction**

[Back to Summary](#b-summary)

**pymetaheuristic** is a Python optimization library built around metaheuristics, benchmark functions, stepwise execution, telemetry, cooperation, rule-based orchestration, constraint-aware evaluation, composable termination criteria, typed variable spaces, chaotic initialization, transfer functions, hyperparameter tuning, and benchmark sweeps. The package provides:

- a broad collection of metaheuristic algorithms
- benchmark functions for testing and visualization
- a stepwise engine API for controlled execution
- telemetry, export helpers, evaluation-indexed convergence data, and save/load for experiments
- cooperative multi-island optimization
- rule-based orchestration for collaborative optimization
- built-in constrained optimization support plus named repair strategies (`clip`, `wang`, `reflect`, `rand`, `limit_inverse`)
- composable `Termination` object with four independent stopping conditions
- automatic per-step diversity and exploration/exploitation tracking in history
- matplotlib-based diversity, convergence, runtime, and explore/exploit charts, including evaluation-indexed convergence plots
- typed variable space (`FloatVar`, `IntegerVar`, `CategoricalVar`, `PermutationVar`, `BinaryVar`)
- ten chaotic maps plus `lhs`, `obl`, and `sobol` population initialization presets
- eight transfer functions and `BinaryAdapter` for binary/discrete optimization
- `HyperparameterTuner` for grid/random hyperparameter search
- `BenchmarkRunner` for multi-algorithm × multi-problem sweeps
- `save_result`, `load_result`, `save_checkpoint`, `load_checkpoint` for persistence
- callback system with lifecycle hooks and callback-driven early stopping
- object-based `Problem` API with parametrized bounds, `latex_code()`, and curated test-problem wrappers
- reusable `levy_flight()` utility and human-readable `algorithm.info()` metadata

---
## 2. **Installation and Package Overview**

### 2.1 **Installation**

Standard installation:

```bash
pip install pymetaheuristic
```

### 2.2 **Package Overview**

[Back to Summary](#b-summary)

| Area | Main objects / functions | What it covers |
|---|---|---|
| Core Optimization | `optimize`, `list_algorithms`, `get_algorithm_info`, `create_optimizer` | Single-algorithm optimization, algorithm discovery, and inspection of default parameters |
| Termination | `Termination`, `EarlyStopping`, callbacks | Composable stopping criteria: max_steps, max_evaluations, max_time, max_early_stop, target_fitness, and callback-driven stops |
| Constraints and Feasibility | `optimize(..., constraints=..., constraint_handler=...)` | Constrained optimization with inequality/equality constraints, feasibility-aware evaluation |
| Benchmarks and Plots (Plotly) | `FUNCTIONS`, `get_test_function`, `plot_function`, `plot_convergence`, `compare_convergence`, `plot_benchmark_summary`, `plot_island_dynamics`, `plot_collaboration_network`, `plot_population_snapshot` | Built-in benchmark functions and Plotly-based landscape, convergence, and cooperation visualizations |
| History Charts (Matplotlib) | `plot_global_best_chart`, `plot_diversity_chart`, `plot_explore_exploit_chart`, `plot_runtime_chart`, `plot_run_dashboard`, `plot_diversity_comparison` | Per-step diversity, exploration/exploitation, runtime, and convergence charts using matplotlib |
| Telemetry and Export | `summarize_result`, `export_history_csv`, `export_population_snapshots_json`, `convergence_data` | Experiment summarization, evaluation-indexed convergence extraction, and export of history and snapshots |
| IO (Persistence) | `save_result`, `load_result`, `save_checkpoint`, `load_checkpoint`, `result_to_json`, `result_from_json` | Save and restore results; checkpoint-and-resume for long runs |
| Typed Variable Space | `FloatVar`, `IntegerVar`, `BinaryVar`, `CategoricalVar`, `PermutationVar`, `build_problem_spec`, `decode_position`, `encode_position` | Define mixed-type search spaces; automatic encode/decode to/from continuous representation |
| Problem Objects | `Problem`, `FunctionalProblem`, `SphereProblem`, `RastriginProblem`, `AckleyProblem`, `RosenbrockProblem`, `ZakharovProblem`, `get_test_problem` | N-dimensional object-based problem definitions with parametrized bounds and `latex_code()` |
| Chaotic Maps | `ChaoticMap`, `chaotic_sequence`, `chaotic_population`, `AVAILABLE_CHAOTIC_MAPS` | Ten chaotic maps for diversity-preserving population initialisation and perturbation |
| Initialisation Presets | `uniform_population`, `lhs_population`, `obl_population`, `sobol_population`, `get_init_function`, `AVAILABLE_INIT_STRATEGIES` | Composable initialisation strategies for any algorithm through `init_function=` or `init_name=` |
| Transfer Functions | `apply_transfer`, `binarize`, `BinaryAdapter`, `vstf_01`–`vstf_04`, `sstf_01`–`sstf_04`, `AVAILABLE_TRANSFER_FUNCTIONS` | Eight transfer functions mapping continuous positions to binary probabilities for binary optimization |
| Repair and Random Utilities | `limit`, `limit_inverse`, `wang`, `rand`, `reflect`, `get_repair_function`, `levy_flight` | Named bound-repair policies and a reusable Lévy-flight sampler |
| Hyperparameter Tuner | `HyperparameterTuner` | Grid or random search over algorithm hyperparameters across multiple trials |
| Benchmark Runner | `BenchmarkRunner` | Multi-algorithm × multi-problem sweeps with statistical aggregation |
| Cooperation | `cooperative_optimize`, `replay_cooperative_result` | Multi-island cooperative optimization |
| Orchestration | `orchestrated_optimize`, `OrchestrationSpec`, `CollaborativeConfig`, `RulesConfig` | Checkpoint-driven cooperation with fixed or rule-based orchestration |
| Reference | `print_root_exports`, `print_reference`, `search_reference` | Programmatic argument reference for all callables |

To quickly inspect parameters:

```python
import pymetaheuristic

# List
pymetaheuristic.print_root_exports()

# Detail
pymetaheuristic.print_reference("optimize")
```

### 2.3 **Optimization, Telemetry, Export, and Plotting Example**

[Back to Summary](#b-summary)

`optimize` is the main high-level entry point for running a single metaheuristic on a user-defined objective function. The user specifies the algorithm, search bounds, and computational budget, while optional keyword arguments configure the selected optimizer and control diagnostics, such as history storage and population snapshots. The function returns a structured result object containing the best solution found, its objective value, and optional run traces that can later be summarized, exported, or plotted. In the example below, `optimize` applies Particle Swarm Optimization (PSO) to the Easom function over a bounded two-dimensional domain, stores the optimization trajectory, and then summarizes the run with `summarize_result`.

When `store_history` and `store_population_snapshots` are enabled, the returned result object contains enough information to support post-run analysis, reproducibility, and visualization. The history can be exported as a tabular CSV file, population states can be saved as JSON snapshots for later inspection, and convergence can be visualized directly with the built-in plotting utilities. In the example below, PSO is applied to the Sphere function, the optimization trace is exported to disk, and the convergence behavior is plotted for immediate inspection.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/11lPwLf13mav4UWSqNolMaKPbqt2lsq4x?usp=sharing)

```python
import numpy as np
import pymetaheuristic

# To use a built-in test function instead, uncomment the next line:
# easom = pymetaheuristic.get_test_function("easom")

# Or define your own objective function.
# The input must be a list (or array-like) of variable values,
# and its length corresponds to the problem dimension.

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

result = pymetaheuristic.optimize(
					algorithm                  = "pso",
					target_function            = easom,
					min_values                 = (-5, -5),
					max_values                 = ( 5,  5),
					max_steps                  = 30,
					seed                       = 42,
					store_history              = True,
					store_population_snapshots = True,
				)

print(result.best_fitness)
print(len(result.history))
print(pymetaheuristic.summarize_result(result))

export_history_csv(result, "population_history.csv")
export_population_snapshots_json(result, "population_snapshots.json")
fig = plot_convergence(result)
fig.show()
```

### 2.4 **Termination Criteria**

[Back to Summary](#b-summary)

`Termination` is a composable stopping-criteria object that replaces (or extends) the individual `max_steps`, `max_evaluations`, `target_fitness`, and `timeout_seconds` keyword arguments. The first condition that triggers ends the run.

Four independent condition types are supported:
- **MG** (`max_steps`): maximum number of macro-steps / iterations.
- **FE** (`max_evaluations`): maximum number of objective-function evaluations.
- **TB** (`max_time`): wall-clock time bound in seconds.
- **ES** (`max_early_stop`): early stopping — halt if the global best has not improved by more than `epsilon` for this many consecutive steps.

`Termination` can also be passed as a plain dict.

```python
import numpy as np
import pymetaheuristic

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

# Build a composable termination with multiple conditions
# The run stops as soon as ANY condition is triggered.
term = pymetaheuristic.Termination(
									max_steps       = 1000,
									max_evaluations = 50000,
									max_time        = 30.0,       # 30-second wall-clock limit
									max_early_stop  = 25,         # stop if no improvement for 25 steps
									epsilon         = 1e-8,
								   )

result = pymetaheuristic.optimize(
									algorithm       = "pso",
									target_function = easom,
									min_values      = (-5, -5),
									max_values      = ( 5,  5),
									termination     = term,
									seed            = 42,
								 )

print(f"Best fitness:        {result.best_fitness:.6f}")
print(f"Steps run:           {result.steps}")
print(f"Evaluations:         {result.evaluations}")
print(f"Termination reason:  {result.termination_reason}")

```

### 2.5 **Constraint Handling Example**

[Back to Summary](#b-summary)

This example illustrates how `optimize` can be applied to constrained optimization problems. The user provides one or more constraint functions alongside the objective, and the solver evaluates candidate solutions by combining objective quality with constraint satisfaction according to the selected handling strategy. In this case, the `"deb"` constraint handler applies feasibility-based comparison rules, so feasible solutions are preferred over infeasible ones, and among infeasible candidates, those with smaller violations are favored. The returned result, therefore, includes not only the best position and penalized search outcome but also metadata describing the raw objective value, the magnitude of constraint violation, and whether the final solution is feasible.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1T8ltBcunERKd7N3q12rW2MdnzSTdsOGs?usp=sharing)

```python
import pymetaheuristic 

# ─────────────────────────────────────────────────────────────────────────────
# Variables: 1) wire diameter d, 2) mean coil diameter D, 3) number of coils N
# Solution:  f* ≈ 0.012665
# ─────────────────────────────────────────────────────────────────────────────

def tension_spring(x = [0, 0, 0]):
    d, D, N = x[0], x[1], x[2]
    return (N + 2) * D * d**2

constraints = [
                  lambda x: 1 - (x[1]**3 * x[2]) / (71785 * x[0]**4),
                  lambda x: (4*x[1]**2 - x[0]*x[1]) / (12566*(x[1]*x[0]**3 - x[0]**4)) + 1/(5108*x[0]**2) - 1,
                  lambda x: 1 - 140.45*x[0] / (x[1]**2 * x[2]),
                  lambda x: (x[0] + x[1]) / 1.5 - 1,
              ]

result = pymetaheuristic.optimize(
					algorithm          = "pso",
					target_function    = tension_spring,
                    min_values         = (0.05, 0.25,  2.0),
                    max_values         = (2.00, 1.30, 15.0),
					constraints        = constraints,
					constraint_handler = "deb",
					max_steps          = 2500,
					seed               = 42,
				 )

print(result.best_position)
print(result.best_fitness)
print(result.metadata["best_raw_fitness"])
print(result.metadata["best_violation"])
print(result.metadata["best_is_feasible"])
```

Other constraints examples:

```python
constraint  = [lambda x: x[0] + x[1] - 1.0   # means x0 + x1 <= 1]
constraints = [
				lambda x:  x[0]**2 + x[1]**2 - 4.0,            # x0^2 + x1^2 <= 4
				lambda x: -x[0],                               # x0 >= 0
				lambda x: -x[1],                               # x1 >= 0
				lambda x:  x[2] - 5.0,                         # x2 <= 5
				lambda x:  2.0 - x[2],                         # x2 >= 2
				lambda x:  abs(x[0] - x[1]) - 0.5,             # |x0 - x1| <= 0.5
				lambda x:  max(x[0], x[1]) - 3.0,              # max(x0, x1) <= 3
				lambda x:  x[0]*x[1] - 2.0,                    # x0*x1 <= 2
				lambda x:  np.sin(x[0]) + x[1] - 1.5,          # sin(x0) + x1 <= 1.5
				lambda x: {"type": "eq", "value": x[0] - x[1]} # x0 = x1
              ]

def c1(x):
    return x[0] + x[1] - 1.0                     # x0 + x1 <= 1

def c2(x):
    return -x[0]                                 # x0 >= 0

def c3(x):
    return {"type": "eq", "value": x[0] - x[1]}  # x0 = x1

constraints = [c1, c2, c3]
```

---

### 2.6 **Cooperative Multi-island Example**

[Back to Summary](#b-summary)

`cooperative_optimize` extends the framework from single-optimizer execution to a collaborative multi-island setting, where several heterogeneous metaheuristics explore the same search space in parallel and periodically exchange information. This interface is useful when the user wants to combine complementary search behaviors—for example, swarm-based, evolutionary, and trajectory-based methods—within a single optimization run. The migration mechanism controls when candidate solutions are shared, how many are transferred, and how communication is structured through a topology such as a ring. In the example below, PSO, GA, SA, and ABCO are executed as cooperating islands on the Easom function, with periodic migration events that allow promising solutions discovered by one method to influence the others.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1DteFWUIqpZZNV4nUM7FGAHfqZN5Vabse?usp=sharing)

```python
import numpy as np
import pymetaheuristic

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

result = pymetaheuristic.ooperative_optimize(
					islands            = [
											{"algorithm": "pso",  "config": {"swarm_size": 25}},
											{"algorithm": "ga",   "config": {}},
											{"algorithm": "sa",   "config": {"temperature_iterations": 20}},
											{"algorithm": "abco", "config": {}},
										  ],
					target_function    = easom,
					min_values         = (-5, -5),
					max_values         = ( 5,  5),
					max_steps          = 20,
					migration_interval = 5,
					migration_size     = 2,
					topology           = "ring",
					seed               = 42,
			   )

print(result.best_fitness)
print(len(result.events))
```

### 2.7 **Orchestrated Cooperation Example**

[Back to Summary](#b-summary)

`orchestrated_optimize` adds an adaptive decision layer atop cooperative multi-island optimization. Instead of relying only on fixed migration schedules, the run is periodically inspected at predefined checkpoints, and an orchestration policy decides whether corrective actions such as rebalancing, perturbation, restarting, or waiting should be applied. This interface is useful when the user wants cooperation to become state-aware and responsive to signals such as stagnation, loss of diversity, or uneven progress across islands. In the example below, PSO, GA, and SA cooperate on the Easom function under a rule-based orchestration policy, and the resulting object records not only the best solution found but also the sequence of checkpoints and the decisions taken during the run.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1j4RbtBjFyxAVuVTMNaJw9ALbREWiIBmn?usp=sharing)

```python
import numpy as np
import pymetaheuristic  

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

config = pymetaheuristic.CollaborativeConfig(
					orchestration = pymetaheuristic.OrchestrationSpec(
																		mode                       = "rules",
																		checkpoint_interval        = 5,
																		max_actions_per_checkpoint = 2,
																		warmup_checkpoints         = 1,
																	 ),
					rules         = pymetaheuristic.RulesConfig(
																	stagnation_threshold     = 4,
																	low_diversity_threshold  = 0.05,
																	high_diversity_threshold = 0.25,
																	perturbation_sigma       = 0.05,
															   ),
				)

result    = pymetaheuristic.orchestrated_optimize(
					islands         = [
										{"label": "pso", "algorithm": "pso", "config": {"swarm_size": 20}},
										{"label": "ga",  "algorithm": "ga",  "config": {"population_size": 20}},
										{"label": "sa",  "algorithm": "sa",  "config": {"temperature": 10.0}},
									  ],
					target_function = easom,
					min_values      = (-5, -5),
					max_values      = ( 5,  5),
					max_steps       = 20,
					seed            = 42,
					config          = config,
				  )

print(result.best_fitness)
print(len(result.checkpoints))
print(len(result.decisions))
```

---

### 2.8 **Chaotic Maps and Transfer Functions**

[Back to Summary](#b-summary)

**Chaotic maps** are initializations based on deterministic chaotic sequences that improve early population diversity and help avoid premature convergence. Ten maps are available: `logistic`, `tent`, `bernoulli`, `chebyshev`, `circle`, `cubic`, `icmic`, `piecewise`, `sine`, `gauss`. The default for population initialization is random. **Transfer functions** map continuous positions to bit-flip probabilities, enabling any continuous metaheuristic to solve binary or Boolean problems. Four V-shaped (`v1`–`v4`) and four S-shaped (`s1`–`s4`) functions are available. `BinaryAdapter` wraps any algorithm and automatically applies the transfer function.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1cvrahJ5Bp4E4vU7I-O6Uqru9SK2hxMXX?usp=sharing)

```python
import itertools
import numpy as np
import pymetaheuristic

# Knapsack Instance
weights  = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82], dtype = int)
values   = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72], dtype = int)
capacity = 165
n_items  = len(weights)

# Known Optimum
# x      = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
# profit = 309
# weight = 165

# Target Function:
def knapsack(bits):
    bits    = np.asarray(bits, dtype = int)
    total_w = np.sum(weights * bits)
    total_v = np.sum(values  * bits)
    if total_w > capacity:
        return 1000.0 + (total_w - capacity)  
    return -float(total_v)  

# Optimize
engine = pymetaheuristic.create_optimizer(
                                          algorithm       = "ga",
                                          target_function = knapsack,
                                          min_values      = [0.0] * n_items,
                                          max_values      = [1.0] * n_items,
                                          population_size = 15,    
                                          max_steps       = 300,
                                          seed            = 42,
                                          init_name       = "chaotic:tent",
                                         )

# Results
adapter      = pymetaheuristic.BinaryAdapter(engine, transfer_fn = "v2")
result       = adapter.run()
found_profit = -result.best_fitness
print("\nMetaheuristic result")
print("Best profit:", found_profit)
print("Binary solution reported:", result.metadata.get("binary_best_position"))

```

---

### 2.9 **Hyperparameter Tuner**

[Back to Summary](#b-summary)

`HyperparameterTuner` performs grid or random search over an algorithm's hyperparameters. It runs each configuration for `n_trials` independent trials, aggregates results, and returns a DataFrame (if pandas is available) or a list of dicts. The `best_params` and `best_fitness` attributes summarise the optimal configuration found.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/13pZQyrMDyegRAcYUJRO6cSwvQ7pZvDKs?usp=sharing)

```python
import numpy as np
import pymetaheuristic

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

tuner = pymetaheuristic.HyperparameterTuner(
                                            algorithm       = "pso",
                                            param_grid      = {
                                                                  "swarm_size": [20, 50, 100],
                                                                  "w":          [0.4, 0.7, 0.9],
                                                                  "c1":         [1.5, 2.0],
                                                                  "c2":         [1.5, 2.0],
                                                                  "init_name":  ["uniform", "chaotic:tent"],
                                                              },
                                            target_function = easom,
                                            min_values      = [-5, -5],
                                            max_values      = [ 5,  5],
                                            termination     = pymetaheuristic.Termination(max_steps = 200),
                                            n_trials        = 5,
                                            objective       = "min",
                                            seed            = 42,
                                            search          = "grid",
                                         )

df      = tuner.run()
summary = tuner.summary()

print(f"Best params:  {tuner.best_params}")
print(f"Best fitness: {tuner.best_fitness:.6f}")
print(summary.head())

```

---

### 2.10 **Save, Load, and Checkpoint**

[Back to Summary](#b-summary)

The IO module provides four functions for persisting results and resuming interrupted runs.

- `save_result` / `load_result`: pickle a completed `OptimizationResult` to disk.
- `result_to_json` / `result_from_json`: export a human-readable JSON summary.
- `save_checkpoint` / `load_checkpoint`: pickle a running `(engine, state)` pair; resume by calling `engine.step(state)` in a loop.

```python
import numpy as np
import pymetaheuristic

# Easom: 
def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)

# Optimize - Run
result = pymetaheuristic.optimize(
                                  algorithm                  = algorithm_id,
                                  target_function            = easom,
                                  min_values                 = (-5, -5),
                                  max_values                 = ( 5,  5),
                                  max_steps                  = 25, # iterations
                                  seed                       = 42,
                                  store_history              = True,
                                  store_population_snapshots = True,
                                )

# Save & Load a Completed Result
pymetaheuristic.save_result(result, "easom_ga.pkl")
r2 = pymetaheuristic.load_result("easom_ga.pkl")
print(f"Reloaded best fitness:  {r2.best_fitness:.6f}")
print(f"Reloaded best position: {r2.best_position}")

# Export and Read a JSON Summary
pymetaheuristic.result_to_json(result, "easom_ga.json")
summary = pymetaheuristic.result_from_json("easom_ga.json")
print(f"JSON best_fitness:  {summary['best_fitness']}")
print(f"JSON best_position: {summary['best_position']}")

# Checkpoint and Resume
engine = pymetaheuristic.create_optimizer(
                                            algorithm                  = algorithm_id,
                                            target_function            = easom,
                                            min_values                 = (-5, -5),
                                            max_values                 = ( 5,  5),
                                            max_steps                  = 25, # iterations
                                            seed                       = 42,
                                            store_history              = True,
                                            store_population_snapshots = True,
                                          )

state = engine.initialize()

# Run
for _ in range(0, 100):
    state = engine.step(state)

pymetaheuristic.save_checkpoint(engine, state, "easom_checkpoint.pkl")
print(f"Checkpoint saved at step {state.step}, best = {state.best_fitness:.6f}")

# Resume from Checkpoint
engine2, state2 = pymetaheuristic.load_checkpoint("easom_checkpoint.pkl")

while not engine2.should_stop(state2):
    state2 = engine2.step(state2)

result_resumed = engine2.finalize(state2)
print(f"Resumed best fitness:  {result_resumed.best_fitness:.6f}")
print(f"Resumed best position: {result_resumed.best_position}")
```

---
## 3. **Algorithm Details**

[Back to Summary](#b-summary)

You can inspect the default parameters of any metaheuristic in the library using `get_algorithm_info()`.

```python
import pymetaheuristic
from pprint import pprint

# Get Info
algorithm_id = "pso"   # change this to any ID from the table, e.g. "de", "ga", "gwo", "woa"
info         = pymetaheuristic.get_algorithm_info(algorithm_id)

# Results
print("Algorithm ID:",   algo_info["algorithm_id"])
print("Algorithm Name:", algo_info["algorithm_name"])
print("")
print("Default Parameters:")
pprint(algo_info["defaults"])

Algorithm ID:   pso
Algorithm Name: Particle Swarm Optimization

Default Parameters:
{'c1': 2.0, 'c2': 2.0, 'decay': 0, 'swarm_size': 30, 'w': 0.9}
```

The table below summarizes the optimization engines currently available in the library. The **Algorithm** column reports the conventional algorithm name, **ID** gives the identifier used in the codebase, **Family** provides a coarse methodological grouping, **Population** indicates whether the algorithm maintains an explicit candidate population, **Candidate Injection** indicates whether the algorithm is currently marked as able to absorb external candidates during cooperative or orchestrated workflows, **Restart** shows whether native restart support is declared, **Checkpoint** reports whether the algorithm state can be serialized and restored, **Snapshot Fit** is a practical recommendation for using store_population_snapshots in the current implementation, **Constraint Support** indicates whether constraints are handled natively by the method or through the framework-level penalty/repair machinery. Finally, **Origin** points to the primary reference or original source associated with the algorithm.

---

| Algorithm                                       | ID              | Family       | Population | Candidate Injection | Restart | Checkpoint | Snapshot Fit | Constraint Support | Origin                                                                                                                                 |
| ----------------------------------------------- | --------------- | ------------ | ---------- | ------------------- | ------- | ---------- | ------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Adaptive Chaotic Grey Wolf Optimizer            | `acgwo`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s42835-023-01621-w)                                                                                    |
| Adaptive Random Search                          | `ars`           | trajectory   | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.1623&rep=rep1&type=pdf)                                           |

<details>
<summary><b>🔍 View complete Metaheuristic reference table</b></summary>
<br/>


| Algorithm | ID | Family | Population | Candidate Injection | Restart | Checkpoint | Snapshot Fit | Constraint Support | Origin |
|---|---|---|---|---|---|---|---|---|---|
| Adaptive Chaotic Grey Wolf Optimizer | `acgwo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s42835-023-01621-w) |
| Adaptive Exploration State-Space Particle Swarm Optimization | `aesspso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.swevo.2025.101868) |
| Adaptive Random Search | `ars` | trajectory | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.1623&rep=rep1&type=pdf) |
| Affix Optimization | `aft` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-021-06392-x) |
| African Vultures Optimization Algorithm | `avoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S0360835221005507) |
| Anarchic Society Optimization | `aso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Anarchic+Society+Optimization+A+human-inspired+method) |
| Ant Colony Optimization | `aco` | swarm | Yes | No | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Ant+colony+optimization+a+new+meta-heuristic+Dorigo+1999) |
| Ant Colony Optimization (Continuous) | `acor` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10732-008-9062-4) |
| Ant Lion Optimizer | `alo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2015.01.010) |
| Aquila Optimizer | `ao` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.cie.2021.107250) |
| Archimedes Optimization Algorithm | `arch_oa` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10489-020-01893-z) |
| Arithmetic Optimization Algorithm | `aoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.cma.2020.113609) |
| Artificial Bee Colony Optimization | `abco` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf) |
| Artificial Ecosystem Optimization | `aeo` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705119302953) |
| Artificial Fish Swarm Algorithm | `afsa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sysengi.com/EN/10.12011/1000-6788(2002)11-32) |
| Artificial Gorilla Troops Optimizer | `agto` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1002/int.22535) |
| Artificial Rabbits Optimization | `aro` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.engappai.2022.105082) |
| Automated Design of Variation Operators | `autov` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.23919/CJE.2022.00.038) |
| Bacterial Chemotaxis Optimizer | `bco` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/MCS.2002.1004010) |
| Bacterial Foraging Optimization | `bfo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/MCS.2002.1004010) |
| Bald Eagle Search | `bes` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10462-019-09732-5) |
| Barnacles Mating Optimizer | `bmo` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ICOICA.2019.8895393) |
| Bat Algorithm | `bat_a` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://arxiv.org/abs/1004.4170) |
| Battle Royale Optimization | `bro` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-020-05004-4) |
| Bees Algorithm | `bea` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=The+bees+algorithm+a+novel+tool+for+complex+optimisation+problems) |
| BFGS Quasi-Newton Method | `bfgs` | math | No | No | No | Yes | No | Framework | [Paper](https://scholar.google.com/scholar?q=Conditioning+of+quasi-Newton+methods+Shanno+1970) |
| Binary Space Partition Tree Genetic Algorithm | `bspga` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.ins.2019.11.055) |
| Biogeography-Based Optimization | `bbo` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2008.919004) |
| Bird Swarm Algorithm | `bsa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1080/0952813X.2015.1042530) |
| Black Widow Optimization | `bwo` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.engappai.2019.103249) |
| Brain Storm Optimization | `bso` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/978-3-642-21515-5_36) |
| Brown-Bear Optimization Algorithm | `bboa` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S0957417423005353) |
| Camel Algorithm | `camel` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.iasj.net/iasj?func=fulltext&aId=118375) |
| Cat Swarm Optimization | `cat_so` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/978-3-540-36668-3_94) |
| Chaos Game Optimization | `cgo` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10462-020-09867-w) |
| Cheetah Based Optimization | `cddo` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s13369-021-05928-6) |
| Cheetah Optimizer | `cdo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S2667305322000448) |
| Chicken Swarm Optimization | `chicken_so` | swarm | Yes | No | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/978-3-319-11857-4_10) |
| Circle-Based Search Algorithm | `circle_sa` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417421009994) |
| Clonal Selection Algorithm | `clonalg` | immune | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.researchgate.net/publication/2917410_Parallelizing_an_Immune-Inspired_Algorithm_for_Efficient_Pattern_Recognition) |
| Coati Optimization Algorithm | `coati_oa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2022.110011) |
| Cockroach Swarm Optimization | `cockroach_so` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ICCET.2010.5485993) |
| Competitive Swarm Optimizer | `cso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TCYB.2014.2314537) |
| Coral Reefs Optimization | `cro` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1155/2014/739768) |
| Coronavirus Herd Immunity Optimization | `chio` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-020-05296-6) |
| Covariance Matrix Adaptation Evolution Strategy | `cmaes` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1162/106365602760972767) |
| Coyote Optimization Algorithm | `coa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/7981604) |
| Cross Entropy Method | `cem` | distribution | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/S0377-2217(96)00385-2) |
| Crow Search Algorithm | `csa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.compstruc.2016.03.001) |
| Cuckoo Search | `cuckoo_s` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://arxiv.org/abs/1003.1594v1) |
| Cultural Algorithm | `ca` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1142/9789814534116) |
| Deer Hunting Optimization Algorithm | `doa` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.hindawi.com/journals/cin/2021/8824610/) |
| Differential Evolution | `de` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1023%2FA%3A1008202821328) |
| Differential Evolution MTS | `hde` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/4983179/) |
| Dispersive Fly Optimization | `dfo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](http://dx.doi.org/10.15439/2014F142) |
| Dragonfly Algorithm | `da` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-015-1920-1) |
| Dwarf Mongoose Optimization Algorithm | `dmoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.cma.2022.114570) |
| Dynamic Virtual Bats Algorithm | `dvba` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/INCoS.2014.40) |
| Earthworm Optimization Algorithm | `eoa` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1504/IJBIC.2015.10004283) |
| Efficient Global Optimization | `ego` | distribution | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1023/A:1008306431147) |
| Egret Swarm Optimization Algorithm | `esoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.hindawi.com/journals/cin/2022/6319430/) |
| Electric Charged Particles Optimization | `ecpo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10462-020-09920-8) |
| Electric Squirrel Optimizer | `eso` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.3390/make7010024) |
| Electromagnetic Field Optimization | `efo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2015.10.048) |
| Elephant Herding Optimization | `eho` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ISCBI.2015.8) |
| Emperor Penguin Colony | `epc` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2018.06.001) |
| Energy Valley Optimizer | `evo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.nature.com/articles/s41598-022-27818-9) |
| Enzyme Activity Optimizer | `eao` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/2227-7390/12/21/3326) |
| Equilibrium Optimizer | `eo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2019.105190) |
| Evolution Strategy (mu + lambda) | `es` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.wiley.com/en-us/Multi+Objective+Optimization+using+Evolutionary+Algorithms-p-9780471873396) |
| Evolutionary Programming | `ep` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/book/5263042) |
| Fast Evolutionary Programming | `fep` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/4235.771163) |
| Feasibility Rule with Objective Function Information | `frofi` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TCYB.2015.2493239) |
| Fennec Fox Optimizer | `ffo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/2227-7390/11/4/1016) |
| Fick's Law Algorithm | `fla` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s11831-022-09849-3) |
| Firefly Algorithm | `firefly_a` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms) |
| Fireworks Algorithm | `fwa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2017.10.046) |
| Fish School Search | `fss` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ICSMC.2008.4811695) |
| Fletcher-Reeves Conjugate Gradient | `frcg` | math | No | No | No | Yes | No | Framework | [Paper](https://doi.org/10.1093/comjnl/7.2.149) |
| Flow Direction Algorithm | `fda` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.cie.2021.107224) |
| Flower Pollination Algorithm | `fpa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms) |
| Flying Dobsonflies Optimizer | `fdo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s11831-022-09849-3) |
| Forensic-Based Investigation Optimization | `fbio` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2020.106339) |
| Forest Optimization Algorithm | `foa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.eswa.2014.05.009) |
| Fox Optimizer | `fox` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10489-022-03533-0) |
| Fruit-Fly Algorithm | `ffa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S0950705114002366) |
| Gaining-Sharing Knowledge Algorithm | `gska` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s13042-019-01053-x) |
| Genetic Algorithm | `ga` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/book/6267401) |
| Geometric Mean Optimizer | `gmo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00500-023-08202-z) |
| Germinal Center Optimization | `gco` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1002/int.21892) |
| Giant Trevally Optimizer | `gto` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/9982508) |
| Glowworm Swarm Optimization | `gso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.springer.com/gp/book/9783319515946) |
| Golden Jackal Optimizer | `gjo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X) |
| Gradient-Based Optimizer | `gbo` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s00500-020-05180-6) |
| Gradient-Based Particle Swarm Optimization | `gpso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2011.10.007) |
| Grasshopper Optimization Algorithm | `goa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2017.01.004) |
| Gravitational Search Algorithm | `gsa` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.ins.2009.03.004) |
| Grey Wolf Optimizer | `gwo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2013.12.007) |
| Harmony Search Algorithm | `hsa` | trajectory | Yes | No | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1177/003754970107600201) |
| Harris Hawks Optimization | `hho` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.future.2019.02.028) |
| Heap-Based Optimizer | `hbo` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s11831-020-09444-y) |
| Henry Gas Solubility Optimization | `hgso` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557) |
| Hill Climb Algorithm | `hc` | trajectory | No | No | No | Yes | No | Framework | [Paper](https://en.wikipedia.org/wiki/Hill_climbing) |
| Human Conception Optimizer | `hco` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417420305017) |
| Hunger Games Search | `hgs` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s11831-021-09537-0) |
| Hunting Search Algorithm | `hus` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ICSCCW.2009.5379451) |
| Hybrid Bat Algorithm | `hba` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=A+Hybrid+Bat+Algorithm+Fister+Yang) |
| Hybrid Self-Adaptive Bat Algorithm | `hsaba` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.hindawi.com/journals/tswj/2014/709738/cta/) |
| Imperialist Competitive Algorithm | `ica` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/4425083) |
| Improved Grey Wolf Optimizer | `i_gwo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.eswa.2020.113917) |
| Improved L-SHADE | `ilshade` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/CEC.2016.7744312) |
| Improved Multi-Operator Differential Evolution | `imode` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Improved+multi-operator+differential+evolution+Sallam+2020) |
| Improved Whale Optimization Algorithm | `i_woa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.jcde.2019.02.002) |
| Invasive Weed Optimization | `iwo` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S0925231206002366) |
| Jaya Algorithm | `jy` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf) |
| Jellyfish Search Optimizer | `jso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.amc.2020.125535) |
| Komodo Mlipir Algorithm | `kma` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2022.108043) |
| Krill Herd Algorithm | `kha` | swarm | Yes | No | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2016.08.041) |
| Life Choice-Based Optimizer | `lco` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00500-019-04443-z) |
| Linear Subspace Surrogate Modeling Evolutionary Algorithm | `l2smea` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2024.3354543) |
| Lion Optimization Algorithm | `loa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.jcde.2015.06.003) |
| Manta Ray Foraging Optimization | `mrfo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.engappai.2019.103300) |
| Mantis Shrimp Optimization Algorithm | `mshoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.3390/math13091500) |
| Marine Predators Algorithm | `mpa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025) |
| Memetic Algorithm | `memetic_a` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9474&rep=rep1&type=pdf) |
| Mirage-Search Optimizer | `mso` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417422021510) |
| Monarch Butterfly Optimization | `mbo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-015-1923-y) |
| Monkey King Evolution V1 | `mke` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2016.01.009) |
| Most Valuable Player Algorithm | `mvpa` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s12351-017-0307-5) |
| Moth Flame Algorithm | `mfa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2015.07.006) |
| Moth Search Algorithm | `msa_e` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s12293-016-0212-3) |
| Mountain Gazelle Optimizer | `mgo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2022.103282) |
| Multi-Surrogate-Assisted Ant Colony Optimization | `misaco` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TCYB.2020.3035521) |
| Multi-Verse Optimizer | `mvo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-015-1870-7) |
| Multifactorial Evolutionary Algorithm | `mfea` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2015.2458037) |
| Multifactorial Evolutionary Algorithm II | `mfea2` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2019.2904771) |
| Multiple Trajectory Search | `mts` | trajectory | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/4631210/) |
| Multiswarm-Assisted Expensive Optimization | `samso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TCYB.2019.2950169) |
| Naked Mole-Rat Algorithm | `nmra` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s00521-017-3287-8) |
| Nelder-Mead Method | `nmm` | trajectory | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) |
| Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO) | `nndrea_so` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2024.3378530) |
| Northern Goshawk Optimization | `ngo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/9638548) |
| Nuclear Reaction Optimization | `nro` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/8612324) |
| Optimal Foraging Algorithm | `ofa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2017.01.006) |
| Osprey Optimization Algorithm | `ooa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.frontiersin.org/articles/10.3389/fmech.2022.1126450/full) |
| Parameter-Free Bat Algorithm | `plba` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Towards+the+development+of+a+parameter-free+bat+algorithm) |
| Parent-Centric Crossover (G3-PCX style) | `pcx` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1162/106365602760972767) |
| Particle Swarm Optimization | `pso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/ICNN.1995.488968) |
| Pathfinder Algorithm | `pfa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2019.03.012) |
| Pelican Optimization Algorithm | `poa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/1424-8220/22/3/855) |
| Population-Based Incremental Learning | `pbil` | distribution | No | No | No | Yes | No | Framework | [Paper](https://apps.dtic.mil/sti/pdfs/ADA282654.pdf) |
| Prominent Space Search | `pss` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s00500-020-05274-3) |
| Queuing Search Algorithm | `qsa` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s12652-020-02849-4) |
| Random Search | `random_s` | trajectory | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1080/01621459.1953.10501200) |
| RIME-ice Algorithm | `rime` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.neucom.2023.02.010) |
| RMSProp | `rmsprop` | math | No | No | No | Yes | No | Framework | [Paper](https://scholar.google.com/scholar?q=Divide+the+gradient+by+a+running+average+Tieleman+Hinton+2012) |
| RUNge Kutta Optimizer | `run` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.eswa.2021.115079) |
| Sailfish Optimizer | `sfo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.engappai.2019.01.001) |
| Salp Swarm Algorithm | `ssa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2017.07.002) |
| Sammon Mapping Assisted Differential Evolution | `sade_sammon` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Surrogate-assisted+evolutionary+algorithm+dimensionality+reduction+Sammon+Chen+2020) |
| Sand Cat Swarm Optimization | `scso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00366-022-01604-x) |
| Satin Bowerbird Optimizer | `sbo` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.engappai.2017.01.006) |
| Sea Lion Optimization | `slo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.14569/IJACSA.2019.0100548) |
| Seagull Optimization Algorithm | `soa` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2018.11.024) |
| Seahorse Optimizer | `seaho` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s10489-022-03994-3) |
| Search And Rescue Optimization | `saro` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1155/2019/2482543) |
| Search Space Independent Operator Based Deep Reinforcement Learning | `ssio_rl` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/JAS.2025.125018) |
| Self-Adaptive Bat Algorithm | `saba` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=A+Hybrid+Bat+Algorithm+Fister+Yang) |
| Self-Adaptive Differential Evolution | `jde` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2006.872133) |
| Sequential Quadratic Programming | `sqp` | math | No | No | No | Yes | No | Framework | [Paper](https://scholar.google.com/scholar?q=Sequential+quadratic+programming+Boggs+Tolle+1995) |
| Serval Optimization Algorithm | `serval_oa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/2227-7390/10/3/339) |
| Shuffle-based Runner-Root Algorithm | `srsr` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2017.02.028) |
| Siberian Tiger Optimization | `sto` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/9989374) |
| Simulated Annealing | `sa` | trajectory | No | Yes | Yes | Yes | No | Framework | [Paper](https://www.jstor.org/stable/1690046) |
| Sine Cosine Algorithm | `sine_cosine_a` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.knosys.2015.12.022) |
| Slime Mould Algorithm | `sma` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.future.2020.03.055) |
| Social Ski-Driver Optimization | `ssdo` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s00521-019-04159-z) |
| Social Spider Algorithm | `sspider_a` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.asoc.2015.02.014) |
| Social Spider Swarm Optimizer | `sso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/pii/S2210650215000632) |
| Spider Monkey Optimization | `smo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/s12293-013-0128-0) |
| Spotted Hyena Inspired Optimizer | `shio` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417421006461) |
| Spotted Hyena Optimizer | `sho` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2017.05.014) |
| Squirrel Search Algorithm | `squirrel_sa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.swevo.2018.02.013) |
| Star Oscillator Optimization | `soo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/2227-7390/11/11/2536) |
| Starfish Optimization Algorithm | `sfoa` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.swevo.2023.101262) |
| Steepest Descent | `sd` | math | No | No | No | Yes | No | Framework | [Paper](https://scholar.google.com/scholar?q=The+origin+of+the+method+of+steepest+descent+Petrova+1997) |
| Student Psychology Based Optimization | `spbo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2020.102804) |
| Success-History Adaptive Differential Evolution | `shade` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/CEC.2014.6900380) |
| Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II | `sacc_eam2` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/CEC.2019.8790061) |
| Surrogate-Assisted Cooperative Swarm Optimization | `sacoso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2017.2674885) |
| Surrogate-Assisted DE with Adaptive Multi-Subspace Search | `sade_amss` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1109/TEVC.2022.3168745) |
| Surrogate-Assisted DE with Adaptive Training Data Selection Criterion | `sade_atdsc` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://scholar.google.com/scholar?q=Surrogate-assisted+differential+evolution+adaptation+training+data+Nishihara+2022) |
| Surrogate-Assisted Partial Optimization | `sapo` | evolutionary | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1007/978-3-031-70085-9_22) |
| Symbiotic Organisms Search | `sos` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.compstruc.2014.03.007) |
| Tabu Search | `ts` | trajectory | No | No | No | Yes | No | Framework | [Paper](https://en.wikipedia.org/wiki/Tabu_search) |
| Tasmanian Devil Optimization | `tdo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/9761116) |
| Teaching Learning Based Optimization | `tlbo` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.cad.2010.12.015) |
| Teamwork Optimization Algorithm | `toa` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s13042-021-01432-3) |
| Tianji Horse Racing Optimizer | `thro` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025522014955) |
| Tree Physiology Optimization | `tpo` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2017.1305421) |
| Tug of War Optimization | `two` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.procs.2020.03.063) |
| Tuna Swarm Optimization | `tso` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.hindawi.com/journals/cin/2021/9210050/) |
| Tunicate Swarm Algorithm | `tsa` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197620301573) |
| Virus Colony Search | `vcs` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2015.11.004) |
| Walrus Optimization Algorithm | `waoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://www.mdpi.com/2227-7390/11/12/2807) |
| War Strategy Optimization | `warso` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://link.springer.com/article/10.1007/s11831-022-09822-0) |
| Water Cycle Algorithm | `wca` | human | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.compstruc.2012.07.010) |
| Weighting and Inertia Random Walk Optimizer | `info` | math | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.eswa.2022.116516) |
| Whale Optimization Algorithm | `woa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.1016/j.advengsoft.2016.01.008) |
| Wildebeest Herd Optimization | `who` | bio | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://doi.org/10.3233/JIFS-190495) |
| Wind Driven Optimization | `wdo` | physics | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/6316175) |
| Zebra Optimization Algorithm | `zoa` | swarm | Yes | Yes | No | Yes | Yes | Framework | [Paper](https://ieeexplore.ieee.org/document/9768862) || Adam (Adaptive Moment Estimation) | `adam` | math | No | No | No | Yes | No | Framework | [Paper](https://arxiv.org/abs/1412.6980) |



<br/>
</details>

---
## 4. **Test Functions**

[Back to Summary](#b-summary)

The graph module can be used with the built-in benchmark functions or with any user-defined scalar objective function that follows the same interface `f(x) -> float`. The unified plotting function automatically adapts the visualization to the number of variables:

```
**1D**: Line Plot                                  (1  Variable, `plot_function_1d`)
**2D**: Contour Map and Heatmap                    (2  Variables,`plot_function_2d`)
**3D**: Interactive Surface Plot                   (2  Variables,`plot_function_3d`)
**ND**: Parallel-coordinates Plot & PCA Projection (3+ Variables,`plot_function_nd`)
```

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/132-yqoaJKkJ4gf6yqjrV1siXVvZ3ZgE7?usp=sharing)

```python

import pymetaheuristic

rastrigin = pymetaheuristic.get_test_function("rastrigin")

# Plot
pymetaheuristic.plot_function_3d(
								  rastrigin,
                                  min_values = (-5.12, -5.12),
                                  max_values = ( 5.12,  5.12),
                                  solutions  = ([   0,    0]),
								  title      = "Rastrigin",
								  filepath   = "out.html",  # also supports .png / .svg / .pdf  
							    )
```

The table below summarizes the benchmark functions currently available in the library. The **Test Function** column reports the conventional function name, **ID** gives the callable identifier used in the codebase (when importing from `pymetaheuristic.src.test_functions`), **Optimal Solution** describes the known global optimum in terms of objective value and, when applicable, the corresponding decision vector, and **Origin** points to the main reference or official source associated with that benchmark.

| Test Function | ID | Optimal Solution | Origin |
|---|---|---|---|
| Ackley | `ackley` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Alpine 1 | `alpine_1` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |

<details>
<summary><b>🔍 View complete Test Functions reference table</b></summary>
<br/>


| Test Function | ID | Optimal Solution | Origin |
|---|---|---|---|
| Ackley | `ackley` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Alpine 1 | `alpine_1` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Alpine 2 | `alpine_2` | *f(x<sub>i</sub>)* = (2.808)<sup>n</sup>; *x<sub>i</sub>* ≈ 7.917 | [Paper](https://arxiv.org/abs/1308.4008) |
| Axis Parallel Hyper-Ellipsoid | `axis_parallel_hyper_ellipsoid` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Beale | `beale` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (3,0.5) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bent Cigar | `bent_cigar` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Bohachevsky F1 | `bohachevsky_1` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bohachevsky F2 | `bohachevsky_2` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bohachevsky F3 | `bohachevsky_3` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Booth | `booth` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (1,3) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Branin RCOS | `branin_rcos` | *f(x<sub>i</sub>)* ≈ 0.397887 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bukin F6 | `bukin_6` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (-10,1) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Chung-Reynolds | `chung_reynolds` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Cosine Mixture | `cosine_mixture` | **Note<sup>1</sup>** | [Paper](http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture) |
| Cross in Tray | `cross_in_tray` | *f(x<sub>i</sub>)* = -2.06261 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Csendes | `csendes` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| De Jong F1 | `de_jong_1` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Discus | `discus` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Dixon-Price | `dixon_price` | **Note<sup>2</sup>** | [Paper](https://www.sfu.ca/~ssurjano/dixonpr.html) |
| Drop Wave | `drop_wave` | *f(x<sub>i</sub>)* = -1; *x<sub>i</sub>* = 0 | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Easom | `easom` | *f(x<sub>i</sub>)* = -1; (π,π) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Eggholder | `eggholder` | *f(x<sub>i</sub>)* = -959.64 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Elliptic | `elliptic` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Expanded Griewank plus Rosenbrock | `expanded_griewank_plus_rosenbrock` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 1 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Goldstein-Price | `goldstein_price` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 3; (0,-1) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Griewangk F8 | `griewangk_8` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Happy Cat | `happy_cat` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = -1 | [Paper](http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf) |
| HGBat | `hgbat` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = -1 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Himmelblau | `himmelblau` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (3,2) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Holder Table | `holder_table` | *f(x<sub>i</sub>)* ≈ -19.2085 | [Paper](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf) |
| Katsuura | `katsuura` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://www.geocities.ws/eadorio/mvf.pdf) |
| Levi | `levy` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 1 | [Paper](https://www.sfu.ca/~ssurjano/levy.html) |
| Levi F13 | `levi_13` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (1,1) | [Paper](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf) |
| Matyas | `matyas` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| McCormick | `mccormick` | *f(x<sub>i</sub>)* ≈ -1.9133 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Michalewicz | `michalewicz` | **Note<sup>3</sup>** | [Paper](https://www.sfu.ca/~ssurjano/michal.html) |
| Modified Schwefel | `modified_schwefel` | **Note<sup>4</sup>** | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Perm | `perm` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 1/*i* | [Paper](https://www.sfu.ca/~ssurjano/perm0db.html) |
| Pinter | `pinter` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Powell | `powell` | **Note<sup>5</sup>** | [Paper](https://www.sfu.ca/~ssurjano/powell.html) |
| Qing | `qing` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = ±√*i* | [Paper](https://arxiv.org/abs/1308.4008) |
| Quintic | `quintic` | **Note<sup>6</sup>** | [Paper](https://arxiv.org/abs/1308.4008) |
| Rastrigin | `rastrigin` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://doi.org/10.1007/978-3-031-14721-0_35) |
| Ridge | `ridge` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html) |
| Rosenbrocks Valley | `rosenbrocks_valley` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 1 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Salomon | `salomon` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Schaffer F2 | `schaffer_2` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schaffer F4 | `schaffer_4` | *f(x<sub>i</sub>)* = 0.292579 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schaffer F6 | `schaffer_6` | *f(x<sub>1</sub>,x<sub>2</sub>)* = 0; (0,0) | [Paper](http://dx.doi.org/10.1016/j.cam.2017.04.047) |
| Schumer-Steiglitz | `schumer_steiglitz` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Schwefel | `schwefel` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* ≈ 420.97 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schwefel 2.21 | `schwefel_221` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Schwefel 2.22 | `schwefel_222` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Six Hump Camel Back | `six_hump_camel_back` | *f(x<sub>i</sub>)* ≈ -1.0316 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Sphere 2 (Sum of Different Powers) | `sphere_2` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://www.sfu.ca/~ssurjano/sumpow.html) |
| Sphere 3 (Rotated Hyper-Ellipsoid) | `sphere_3` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://www.sfu.ca/~ssurjano/rothyp.html) |
| Step | `step` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Step 2 | `step_2` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = -0.5 | [Paper](https://arxiv.org/abs/1308.4008) |
| Step 3 | `step_3` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/abs/1308.4008) |
| Stepint | `stepint` | **Note<sup>7</sup>** | [Paper](https://arxiv.org/abs/1308.4008) |
| Styblinski-Tang | `styblinski_tang` | *f(x<sub>i</sub>)* ≈ -39.166*n*; *x<sub>i</sub>* ≈ -2.904 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Three Hump Camel Back | `three_hump_camel_back` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Trid | `trid` | **Note<sup>8</sup>** | [Paper](https://www.sfu.ca/~ssurjano/trid.html) |
| Weierstrass | `weierstrass` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Whitley | `whitley` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 1 | [Paper](https://arxiv.org/abs/1308.4008) |
| Zakharov | `zakharov` | *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = 0 | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| CEC 2022 F1 | `cec_2022_f01` | *f(x<sub>i</sub>)* = 300; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F2 | `cec_2022_f02` | *f(x<sub>i</sub>)* = 400; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F3 | `cec_2022_f03` | *f(x<sub>i</sub>)* = 600; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F4 | `cec_2022_f04` | *f(x<sub>i</sub>)* = 800; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F5 | `cec_2022_f05` | *f(x<sub>i</sub>)* = 900; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F6 | `cec_2022_f06` | *f(x<sub>i</sub>)* = 1800; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F7 | `cec_2022_f07` | *f(x<sub>i</sub>)* = 2000; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F8 | `cec_2022_f08` | *f(x<sub>i</sub>)* = 2200; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F9 | `cec_2022_f09` | *f(x<sub>i</sub>)* = 2300; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F10| `cec_2022_f10` | *f(x<sub>i</sub>)* = 2400; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F11| `cec_2022_f11` | *f(x<sub>i</sub>)* = 2500; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F12| `cec_2022_f12` | *f(x<sub>i</sub>)* = 2700; **Note<sup>9</sup>** | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |

#### Notes:
* **Note<sup>1</sup>:** *f(x<sub>i</sub>)* = multimodal; *x<sub>i</sub>* = peak-seeking style test
* **Note<sup>2</sup>:** *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = follows recursive optimum
* **Note<sup>3</sup>:** *f(x<sub>i</sub>)* < 0; *x<sub>i</sub>* = known minima depend on *n* and *m*
* **Note<sup>4</sup>:** *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = shifted Schwefel optimum
* **Note<sup>5</sup>:** *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = blockwise optimum at zero
* **Note<sup>6</sup>:** *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = multiple roots
* **Note<sup>7</sup>:** *f(x<sub>i</sub>)* = 0; *x<sub>i</sub>* = piecewise integer floor optimum
* **Note<sup>8</sup>:** *f(x<sub>i</sub>)* = depends on dimension
* **Note<sup>9</sup>:** CEC 2022 functions are shifted and biased benchmark instances. The optimum point *x<sub>i</sub>* is defined by the official benchmark shift data. F1–F5 and F9–F12 support dimensions {2,10,20}; F6–F8 support only {10,20}.

<br/>
</details>

## 5. **Other Libraries**

[Back to Summary](#b-summary)

* For Multiobjective Optimization or Many Objectives Optimization, try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)
* For Traveling Salesman Problems (TSP), try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

### **Acknowledgement**

This section is dedicated to everyone who helped improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)


