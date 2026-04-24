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
<summary><b>🔍 View complete metaheuristic reference table</b></summary>
<br/>


| Algorithm                                       | ID              | Family       | Population | Candidate Injection | Restart | Checkpoint | Snapshot Fit | Constraint Support | Origin                                                                                                                                 |
| ----------------------------------------------- | --------------- | ------------ | ---------- | ------------------- | ------- | ---------- | ------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Adaptive Chaotic Grey Wolf Optimizer            | `acgwo`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s42835-023-01621-w)                                                                                    |
| Adaptive Random Search                          | `ars`           | trajectory   | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.1623&rep=rep1&type=pdf)                                           |
| Anarchic Society Optimization                   | `aso`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://scholar.google.com/scholar?q=Anarchic+Society+Optimization+A+human-inspired+method)                                    |
| Ant Lion Optimizer                              | `alo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2015.01.010)                                                                              |
| Arithmetic Optimization Algorithm               | `aoa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.cma.2020.113609)                                                                                    |
| Artificial Bee Colony Optimization              | `abco`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf)                                                                                  |
| Artificial Fish Swarm Algorithm                 | `afsa`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.sysengi.com/EN/10.12011/1000-6788%282002%2911-32)                                                                  |
| Bacterial Foraging Optimization                 | `bfo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/MCS.2002.1004010)                                                                                      |
| Bat Algorithm                                   | `bat_a`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://arxiv.org/abs/1004.4170)                                                                                               |
| Bees Algorithm                                  | `bea`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://scholar.google.com/scholar?q=The+bees+algorithm+a+novel+tool+for+complex+optimisation+problems)                        |
| Biogeography-Based Optimization                 | `bbo`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/TEVC.2008.919004)                                                                                      |
| Camel Algorithm                                 | `camel`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.iasj.net/iasj?func=fulltext&aId=118375)                                                                            |
| Cat Swarm Optimization                          | `cat_so`        | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/978-3-540-36668-3_94)                                                                                  |
| Chicken Swarm Optimization                      | `chicken_so`    | swarm        | Yes        | No                  | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/978-3-319-11857-4_10)                                                                                  |
| Clonal Selection Algorithm                      | `clonalg`       | immune       | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.researchgate.net/publication/2917410_Parallelizing_an_Immune-Inspired_Algorithm_for_Efficient_Pattern_Recognition) |
| Coati Optimization Algorithm                    | `coati_oa`      | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.knosys.2022.110011)                                                                                  |
| Cockroach Swarm Optimization                    | `cockroach_so`  | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/ICCET.2010.5485993)                                                                                    |
| Coral Reefs Optimization                        | `cro`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1155/2014/739768)                                                                                           |
| Cross Entropy Method                            | `cem`           | distribution | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/S0377-2217%2896%2900385-2)                                                                             |
| Crow Search Algorithm                           | `csa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.compstruc.2016.03.001)                                                                               |
| Cuckoo Search                                   | `cuckoo_s`      | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://arxiv.org/abs/1003.1594v1)                                                                                             |
| Cultural Algorithm                              | `ca`            | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1142/9789814534116)                                                                                         |
| Differential Evolution                          | `de`            | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1023%2FA%3A1008202821328)                                                                                   |
| Differential Evolution MTS                      | `hde`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://ieeexplore.ieee.org/document/4983179/)                                                                                 |
| Dispersive Fly Optimization                     | `dfo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](http://dx.doi.org/10.15439/2014F142)                                                                                           |
| Dragonfly Algorithm                             | `da`            | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s00521-015-1920-1)                                                                                     |
| Dynamic Virtual Bats Algorithm                  | `dvba`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/INCoS.2014.40)                                                                                         |
| Elephant Herding Optimization                   | `eho`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/ISCBI.2015.8)                                                                                          |
| Evolution Strategy (mu + lambda)                | `es`            | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.wiley.com/en-us/Multi+Objective+Optimization+using+Evolutionary+Algorithms-p-9780471873396)                        |
| Firefly Algorithm                               | `firefly_a`     | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms)                                      |
| Fireworks Algorithm                             | `fwa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.asoc.2017.10.046)                                                                                    |
| Fish School Search                              | `fss`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/ICSMC.2008.4811695)                                                                                    |
| Flow Direction Algorithm                        | `fda`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.cie.2021.107224)                                                                                     |
| Flower Pollination Algorithm                    | `fpa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms)                                      |
| Forest Optimization Algorithm                   | `foa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.eswa.2014.05.009)                                                                                    |
| Genetic Algorithm                               | `ga`            | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://ieeexplore.ieee.org/book/6267401)                                                                                      |
| Geometric Mean Optimizer                        | `gmo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s00500-023-08202-z)                                                                                    |
| Glowworm Swarm Optimization                     | `gso`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.springer.com/gp/book/9783319515946)                                                                                |
| Grasshopper Optimization Algorithm              | `goa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2017.01.004)                                                                              |
| Gravitational Search Algorithm                  | `gsa`           | physics      | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.ins.2009.03.004)                                                                                     |
| Grey Wolf Optimizer                             | `gwo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2013.12.007)                                                                              |
| Harmony Search Algorithm                        | `hsa`           | trajectory   | Yes        | No                  | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1177/003754970107600201)                                                                                    |
| Harris Hawks Optimization                       | `hho`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.future.2019.02.028)                                                                                  |
| Hill Climb Algorithm                            | `hc`            | trajectory   | No         | No                  | No      | Yes        | No           | Framework          | [Paper](https://en.wikipedia.org/wiki/Hill_climbing)                                                                                   |
| Hunting Search Algorithm                        | `hus`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/ICSCCW.2009.5379451)                                                                                   |
| Hybrid Bat Algorithm                            | `hba`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://scholar.google.com/scholar?q=A+Hybrid+Bat+Algorithm+Fister+Yang)                                                       |
| Hybrid Self-Adaptive Bat Algorithm              | `hsaba`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://www.hindawi.com/journals/tswj/2014/709738/cta/)                                                                        |
| Improved Grey Wolf Optimizer                    | `i_gwo`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.eswa.2020.113917)                                                                                    |
| Improved L-SHADE                                | `ilshade`       | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/CEC.2016.7744312)                                                                                      |
| Improved Whale Optimization Algorithm           | `i_woa`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.jcde.2019.02.002)                                                                                    |
| Jaya Algorithm                                  | `jy`            | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf)                                                                    |
| Jellyfish Search Optimizer                      | `jso`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.amc.2020.125535)                                                                                     |
| Krill Herd Algorithm                            | `kha`           | swarm        | Yes        | No                  | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.asoc.2016.08.041)                                                                                    |
| Lion Optimization Algorithm                     | `loa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.jcde.2015.06.003)                                                                                    |
| Mantis Shrimp Optimization Algorithm            | `mshoa`         | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.3390/math13091500)                                                                                          |
| Memetic Algorithm                               | `memetic_a`     | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9474&rep=rep1&type=pdf)                                           |
| Monarch Butterfly Optimization                  | `mbo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s00521-015-1923-y)                                                                                     |
| Monkey King Evolution V1                        | `mke`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.knosys.2016.01.009)                                                                                  |
| Moth Flame Algorithm                            | `mfa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.knosys.2015.07.006)                                                                                  |
| Multi-Verse Optimizer                           | `mvo`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1007/s00521-015-1870-7)                                                                                     |
| Multiple Trajectory Search                      | `mts`           | trajectory   | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://ieeexplore.ieee.org/document/4631210/)                                                                                 |
| Nelder-Mead Method                              | `nmm`           | trajectory   | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)                                                                      |
| Parameter-Free Bat Algorithm                    | `plba`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://scholar.google.com/scholar?q=Towards+the+development+of+a+parameter-free+bat+algorithm)                                |
| Parent-Centric Crossover (G3-PCX style)         | `pcx`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1162/106365602760972767)                                                                                    |
| Particle Swarm Optimization                     | `pso`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/ICNN.1995.488968)                                                                                      |
| Pathfinder Algorithm                            | `pfa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.asoc.2019.03.012)                                                                                    |
| Population-Based Incremental Learning           | `pbil`          | distribution | No         | No                  | No      | Yes        | No           | Framework          | [Paper](https://apps.dtic.mil/sti/pdfs/ADA282654.pdf)                                                                                  |
| Random Search                                   | `random_s`      | trajectory   | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1080/01621459.1953.10501200)                                                                                |
| Salp Swarm Algorithm                            | `ssa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2017.07.002)                                                                              |
| Self-Adaptive Bat Algorithm                     | `saba`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://scholar.google.com/scholar?q=A+Hybrid+Bat+Algorithm+Fister+Yang)                                                       |
| Self-Adaptive Differential Evolution            | `jde`           | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/TEVC.2006.872133)                                                                                      |
| Simulated Annealing                             | `sa`            | trajectory   | No         | Yes                 | Yes     | Yes        | No           | Framework          | [Paper](https://www.jstor.org/stable/1690046)                                                                                          |
| Sine Cosine Algorithm                           | `sine_cosine_a` | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.knosys.2015.12.022)                                                                                  |
| Student Psychology Based Optimization           | `spbo`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2020.102804)                                                                              |
| Success-History Adaptive Differential Evolution | `shade`         | evolutionary | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1109/CEC.2014.6900380)                                                                                      |
| Symbiotic Organisms Search                      | `sos`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.compstruc.2014.03.007)                                                                               |
| Teaching Learning Based Optimization            | `tlbo`          | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.cad.2010.12.015)                                                                                     |
| Whale Optimization Algorithm                    | `woa`           | swarm        | Yes        | Yes                 | No      | Yes        | Yes          | Framework          | [Paper](https://doi.org/10.1016/j.advengsoft.2016.01.008)                                                                              |

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
| Ackley | `ackley` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Alpine 1 | `alpine_1` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Alpine 2 | `alpine_2` | $f(x_i) = (2.808)^n$; $x_i ≈ 7.917$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Axis Parallel Hyper-Ellipsoid | `axis_parallel_hyper_ellipsoid` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Beale | `beale` | $f(x_1,x_2) = 0$; $(3,0.5)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bent Cigar | `bent_cigar` | $f(x_i) = 0$; $x_i = 0$  | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Bohachevsky F1 | `bohachevsky_1` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bohachevsky F2 | `bohachevsky_2` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bohachevsky F3 | `bohachevsky_3` | $f(x_1,x_2) = 0$; $(0,0)$  | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Booth | `booth` | $f(x_1,x_2) = 0$; $(1,3)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Branin RCOS | `branin_rcos` | $f(x_i) ≈ 0.397887$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Bukin F6 | `bukin_6` | $f(x_1,x_2) = 0$; $(-10,1)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Chung-Reynolds | `chung_reynolds` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Cosine Mixture | `cosine_mixture` | $Note^1$ | [Paper](http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture) |
| Cross in Tray | `cross_in_tray` | $f(x_i) = -2.06261$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Csendes | `csendes` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| De Jong F1 | `de_jong_1` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Discus | `discus` | $f(x_i) = 0$; $x_i = 0$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Dixon-Price | `dixon_price` | $Note^2$ | [Paper](https://www.sfu.ca/~ssurjano/dixonpr.html) |
| Drop Wave | `drop_wave` | $f(x_i) = -1$; $x_i = 0$ | [Paper](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf) |
| Easom | `easom` | $f(x_i) = -1$; $(π,π)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Eggholder | `eggholder` | $f(x_i) = -959.64$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Elliptic | `elliptic` | $f(x_i) = 0$; $x_i = 0$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Expanded Griewank plus Rosenbrock | `expanded_griewank_plus_rosenbrock` | $f(x_i) = 0$; $x_i = 1$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Goldstein-Price | `goldstein_price` | $f(x_1,x_2) = 3$; $(0,-1)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Griewangk F8 | `griewangk_8` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Happy Cat | `happy_cat` | $f(x_i) = 0$; $x_i = -1$ | [Paper](http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf) |
| HGBat | `hgbat` | $f(x_i) = 0$; $x_i = -1$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Himmelblau | `himmelblau` | $f(x_1,x_2) = 0$; $(3,2)$  | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Holder Table | `holder_table` | $f(x_i) ≈ -19.2085$ | [Paper](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf) |
| Katsuura | `katsuura` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://www.geocities.ws/eadorio/mvf.pdf) |
| Levi | `levy` | $f(x_i) = 0$; $x_i = 1$ | [Paper](https://www.sfu.ca/~ssurjano/levy.html) |
| Levi F13 | `levi_13` | $f(x_1,x_2) = 0$; $(1,1)$ | [Paper](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf) |
| Matyas | `matyas` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| McCormick | `mccormick` |$f(x_i) ≈ -1.9133$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Michalewicz | `michalewicz` | $Note^3$ | [Paper](https://www.sfu.ca/~ssurjano/michal.html) |
| Modified Schwefel | `modified_schwefel` | $Note^4$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Perm | `perm` | $f(x_i) = 0$; $x_i = 1/i$ | [Paper](https://www.sfu.ca/~ssurjano/perm0db.html) |
| Pinter | `pinter` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Powell | `powell` | $Note^5$ | [Paper](https://www.sfu.ca/~ssurjano/powell.html) |
| Qing | `qing` | $f(x_i) = 0$; $x_i = ±√i$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Quintic | `quintic` | $Note^6$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Rastrigin | `rastrigin` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://doi.org/10.1007/978-3-031-14721-0_35) |
| Ridge | `ridge` | $f(x_i) = 0$; $x_i = 0$ | [Paper](http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html) |
| Rosenbrocks Valley | `rosenbrocks_valley` | $f(x_i) = 0$; $x_i = 1$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Salomon | `salomon` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Schaffer F2 | `schaffer_2` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schaffer F4 | `schaffer_4` | $f(x_i) = 0.292579$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schaffer F6 | `schaffer_6` | $f(x_1,x_2) = 0$; $(0,0)$ | [Paper](http://dx.doi.org/10.1016/j.cam.2017.04.047) |
| Schumer-Steiglitz | `schumer_steiglitz` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Schwefel | `schwefel` | $f(x_i) = 0$; $x_i ≈ 420.97$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Schwefel 2.21 | `schwefel_221` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Schwefel 2.22 | `schwefel_222` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Six Hump Camel Back | `six_hump_camel_back` | $f(x_i) ≈ -1.0316$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Sphere 2 (Sum of Different Powers) | `sphere_2` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://www.sfu.ca/~ssurjano/sumpow.html) |
| Sphere 3 (Rotated Hyper-Ellipsoid) | `sphere_3` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://www.sfu.ca/~ssurjano/rothyp.html) |
| Step | `step` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Step 2 | `step_2` | $f(x_i) = 0$; $x_i = -0.5$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Step 3 | `step_3` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Stepint | `stepint` | $Note^7$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Styblinski-Tang | `styblinski_tang` |$f(x_i) ≈ -39.166n$; $x_i ≈ -2.904$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Three Hump Camel Back | `three_hump_camel_back` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| Trid | `trid` | $Note^8$ | [Paper](https://www.sfu.ca/~ssurjano/trid.html) |
| Weierstrass | `weierstrass` | $f(x_i) = 0$; $x_i = 0$ | [Paper](http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf) |
| Whitley | `whitley` | $f(x_i) = 0$; $x_i = 1$ | [Paper](https://arxiv.org/abs/1308.4008) |
| Zakharov | `zakharov` | $f(x_i) = 0$; $x_i = 0$ | [Paper](https://arxiv.org/pdf/1308.4008.pdf) |
| CEC 2022 F1 | `cec_2022_f01` | $f(x_i) = 300$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F2 | `cec_2022_f02` | $f(x_i) = 400$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F3 | `cec_2022_f03` | $f(x_i) = 600$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F4 | `cec_2022_f04` | $f(x_i) = 800$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F5 | `cec_2022_f05` | $f(x_i) = 900$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F6 | `cec_2022_f06` | $f(x_i) = 1800$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F7 | `cec_2022_f07` | $f(x_i) = 2000$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F8 | `cec_2022_f08` | $f(x_i) = 2200$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F9 | `cec_2022_f09` | $f(x_i) = 2300$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F10| `cec_2022_f10` | $f(x_i) = 2400$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F11| `cec_2022_f11` | $f(x_i) = 2500$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |
| CEC 2022 F12| `cec_2022_f12` | $f(x_i) = 2700$; $Note^9$ | [Paper](https://github.com/P-N-Suganthan/2022-SO-BO) |

#### Notes:
* $Note^1$: $f(x_i) =$ multimodal; $x_i =$ peak-seeking style test
* $Note^2$: $f(x_i) = 0$; $x_i =$ follows recursive optimum
* $Note^3$: $f(x_i) < 0$; $x_i =$ known minima depend on $n$ and $m$
* $Note^4$: $f(x_i) = 0$; $x_i =$ shifted Schwefel optimum
* $Note^5$: $f(x_i) = 0$; $x_i =$ blockwise optimum at zero
* $Note^6$: $f(x_i) = 0$; $x_i =$ multiple roots
* $Note^7$: $f(x_i) = 0$; $x_i =$ piecewise integer floor optimum
* $Note^8$: $f(x_i) =$ depends on dimension
* $Note^9$: CEC 2022 functions are shifted and biased benchmark instances. The optimum point $x_i$ is defined by the official benchmark shift data. F1–F5 and F9–F12 support dimensions {2,10,20}; F6–F8 support only {10,20}.

## 5. **Other Libraries**

[Back to Summary](#b-summary)

* For Multiobjective Optimization or Many Objectives Optimization, try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)
* For Traveling Salesman Problems (TSP), try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

### **Acknowledgement**

This section is dedicated to everyone who helped improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)


