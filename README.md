
<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/logo_pmh_.png" alt="Logo" width="300" height="300"/>
</p>



# pymetaheuristic

A Python library for metaheuristic optimization and collaborative search, bringing together **307 optimization algorithms** across swarm, evolutionary, trajectory, physics-inspired, nature-inspired, human-inspired, and mathematical families. 

## A. **Version Note**

This README targets **pymetaheuristic-v5+**. It can be installed with:

```bash
pip install pymetaheuristic
```

For legacy, the old library can still be installed with:

```bash
pip install pymetaheuristic==1.9.5
```

## B. **pymetaheuristic Lab**

New to Python or prefer a graphical interface? The **pymetaheuristic Lab** provides a convenient Web App to run optimizations without writing extensive code.

```python
import pymetaheuristic

# Start the web service using:
pymetaheuristic.web_app()

# Terminate the web service using:
pymetaheuristic.web.web_stop()
```

<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/lab_.png" alt="Lab" width="700"/>
</p>

* [Preview -- **pyMetaheuristic Lab** -- in Google Colab](https://colab.research.google.com/drive/1ouIGIrD0QNMuTPC2diTqGoY4v6gxb0xt?usp=sharing)

_This Google Colab Demo is intended for quick demos only. For the best experience, run the Web UI locally or open it directly in a full browser._

## C. **Summary** 

1. [Introduction](#1-introduction)
2. [Installation and Package Overview](#2-installation-and-package-overview)
   - [2.1 Installation](#21-installation)
   - [2.2 Package Overview](#22-package-overview)
   - [2.3 Optimization, Telemetry, Export, and Plotting Example](#23-optimization-telemetry-export-and-plotting-example) --- [[Colab Demo]](https://colab.research.google.com/drive/11lPwLf13mav4UWSqNolMaKPbqt2lsq4x?usp=sharing) ---
   - [2.4 Termination Criteria](#24-termination-criteria) --- [[Colab Demo]](https://colab.research.google.com/drive/1GVIsdruPnozKHE0Rd972pk8tgXAKGKFs?usp=sharing) ---  
   - [2.5 Constraint Handling Example](#25-constraint-handling-example) --- [[Colab Demo]](https://colab.research.google.com/drive/1T8ltBcunERKd7N3q12rW2MdnzSTdsOGs?usp=sharing) ---  
   - [2.6 Cooperative Multi-island Example](#26-cooperative-multi-island-example) --- [[Colab Demo] ](https://colab.research.google.com/drive/1DteFWUIqpZZNV4nUM7FGAHfqZN5Vabse?usp=sharing) ---
   - [2.7 Orchestrated Cooperation Example](#27-orchestrated-cooperation-example) --- [[Colab Demo]](https://colab.research.google.com/drive/1j4RbtBjFyxAVuVTMNaJw9ALbREWiIBmn?usp=sharing) --- 
   - [2.8 Chaotic Maps and Transfer Functions](#28-chaotic-maps-and-transfer-functions) --- [[Colab Demo]](https://colab.research.google.com/drive/1cvrahJ5Bp4E4vU7I-O6Uqru9SK2hxMXX?usp=sharing) ---
   - [2.9 Hyperparameter Tuner](#29-hyperparameter-tuner) --- [[Colab Demo] ](https://colab.research.google.com/drive/13pZQyrMDyegRAcYUJRO6cSwvQ7pZvDKs?usp=sharing) ---
   - [2.10 Save, Load, and Checkpoint](#210-save-load-and-checkpoint) --- [[Colab Demo] ](https://colab.research.google.com/drive/1detpXqDFMO-rNUpCSiN0RnuljUt5xD-E?usp=sharing) ---
   - [2.11 Benchmark Runner](#211-benchmark-runner) --- [[Colab Demo] ](https://colab.research.google.com/drive/1ZMw5RLFIU-EBPJoNp3kNyXg1KCU1KlFA?usp=sharing) ---
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

pymetaheuristic.export_history_csv(result, "population_history.csv")
pymetaheuristic.export_population_snapshots_json(result, "population_snapshots.json")
fig = pymetaheuristic.plot_convergence(result)
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

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1GVIsdruPnozKHE0Rd972pk8tgXAKGKFs?usp=sharing)

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
constraint  = [lambda x: x[0] + x[1] - 1.0]                    # x0 + x1 <= 1
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

result = pymetaheuristic.cooperative_optimize(
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

The IO module provides a set of functions for persisting results and resuming interrupted runs.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1detpXqDFMO-rNUpCSiN0RnuljUt5xD-E?usp=sharing)

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

### 2.11 **Benchmark Runner**

[Back to Summary](#b-summary)

`BenchmarkRunner` performs multi-algorithm × multi-problem comparative sweeps. It executes every algorithm on every problem for a configurable number of independent trials, records the best fitness and wall-clock time for each run, and captures failed trials without interrupting the sweep. The raw results are returned as a tidy DataFrame that can be aggregated into summary statistics, rank tables, and publication-quality compact tables. Parallel execution across trials is available through the `n_jobs` argument. After calling `.run()`, the five dedicated Plotly-based visualisation functions — `plot_benchmark_barplots`, `plot_benchmark_boxplots`, `plot_benchmark_rank_heatmap`, `plot_benchmark_runtime`, and `plot_benchmark_convergence` — produce interactive charts. All five return `go.Figure` objects that can be further customised, displayed inline in Jupyter or Colab, or saved to HTML, PNG, SVG, or PDF.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1ZMw5RLFIU-EBPJoNp3kNyXg1KCU1KlFA?usp=sharing)

```python
import pandas as pd
import pymetaheuristic

# Algorithms
algorithms = ["acgwo", "gwo", "i_gwo", "fox", "tlbo"]

# Problems
rastrigin  = pymetaheuristic.get_test_function("rastrigin")
rosenbrock = pymetaheuristic.get_test_function("rosenbrocks_valley")

problems = [
               {
                   "name":            "Rastrigin-5D",
                   "target_function": rastrigin,
                   "min_values":      [-5.12] * 5,
                   "max_values":      [ 5.12] * 5,
                   "objective":       "min",
               },
               {
                   "name":            "Rosenbrock-5D",
                   "target_function": rosenbrock,
                   "min_values":      [-30.0] * 5,
                   "max_values":      [ 30.0] * 5,
                   "objective":       "min",
               },
           ]

# Runner
termination = pymetaheuristic.Termination(max_steps = 250)
runner      = pymetaheuristic.BenchmarkRunner(
                                               algorithms  = algorithms,
                                               problems    = problems,
                                               termination = termination,
                                               n_trials    = 5,
                                               seed        = 42,
                                               n_jobs      = 1,
                                             ) 
raw_df = runner.run(show_progress = True)

# Raw Results
failed_df  = raw_df[raw_df["error"].notna()].copy()
valid_df   = raw_df[raw_df["error"].isna()].copy()
summary_df = runner.summary().copy()

# Rank Table
rank_table                 = summary_df.pivot(index = "algorithm", columns = "problem", values = "rank")
rank_table["average_rank"] = rank_table.mean(axis = 1)
rank_table                 = rank_table.sort_values("average_rank")

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

The table below summarizes the optimization engines currently available in the library. The **Algorithm** column reports the conventional algorithm name, **ID** gives the identifier used in the codebase, **Family** provides a coarse methodological grouping, **Population** indicates whether the algorithm maintains an explicit candidate population, **Candidate Injection** indicates whether the algorithm is currently marked as able to absorb external candidates during cooperative or orchestrated workflows, **Restart** shows whether native restart support is declared, and **Snapshot Fit** provides a practical recommendation for using store_population_snapshots in the current implementation. Click the algorithm name to open its primary reference or original source.
 All algorithms support checkpointing through the library framework, and all constraint handling is available through the framework-level constraint machinery.

---

| Algorithm | ID | Family | Population | Candidate Injection | Restart | Snapshot Fit |
| --- | --- | --- | --- | --- | --- | --- |
| [Adam (Adaptive Moment Estimation)](https://doi.org/10.48550/arXiv.1412.6980) | `adam` | math | No | No | No | No |
| [Adaptive Chaotic Grey Wolf Optimizer](https://doi.org/10.1007/s42835-023-01621-w) | `acgwo` | swarm | Yes | Yes | No | Yes |

<details>
<summary><b>🔍 View complete Metaheuristic reference table</b></summary>
<br/>

| Algorithm | ID | Family | Population | Candidate Injection | Restart | Snapshot Fit |
| --- | --- | --- | --- | --- | --- | --- |
| [Adam (Adaptive Moment Estimation)](https://doi.org/10.48550/arXiv.1412.6980) | `adam` | math | No | No | No | No |
| [Adaptive Chaotic Grey Wolf Optimizer](https://doi.org/10.1007/s42835-023-01621-w) | `acgwo` | swarm | Yes | Yes | No | Yes |
| [Adaptive Exploration State-Space Particle Swarm Optimization](https://doi.org/10.1016/j.swevo.2025.101868) | `aesspso` | swarm | Yes | Yes | No | Yes |
| [Adaptive Random Search](https://doi.org/10.1002/nav.20422) | `ars` | trajectory | Yes | Yes | No | Yes |
| [African Vultures Optimization Algorithm](https://doi.org/10.1016/j.cie.2021.107408) | `avoa` | swarm | Yes | Yes | No | Yes |
| [Ali Baba and the Forty Thieves](https://doi.org/10.1007/s00521-021-06392-x) | `aft` | human | Yes | Yes | No | Yes |
| [Anarchic Society Optimization](https://doi.org/10.1109/CEC.2011.5949940) | `aso` | swarm | Yes | Yes | No | Yes |
| [Ant Colony Optimization](https://doi.org/10.1109/3477.484436) | `aco` | swarm | Yes | No | No | Yes |
| [Ant Colony Optimization (Continuous)](https://doi.org/10.1016/j.ejor.2006.06.046) | `acor` | swarm | Yes | Yes | No | Yes |
| [Ant Lion Optimizer](https://doi.org/10.1016/j.advengsoft.2015.01.010) | `alo` | swarm | Yes | Yes | No | Yes |
| [Aquila Optimizer](https://doi.org/10.1016/j.cie.2021.107250) | `ao` | swarm | Yes | Yes | No | Yes |
| [Archerfish Hunting Optimizer](https://doi.org/10.1016/j.engappai.2024.108081) | `aho` | swarm | Yes | Yes | No | Yes |
| [Archimedes Optimization Algorithm](https://doi.org/10.1007/s10489-020-01893-z) | `arch_oa` | physics | Yes | Yes | No | Yes |
| [Arithmetic Optimization Algorithm](https://doi.org/10.1016/j.cma.2020.113609) | `aoa` | swarm | Yes | Yes | No | Yes |
| [Artemisinin Optimization](https://doi.org/10.1016/j.displa.2024.102740) | `artemisinin_o` | nature | Yes | Yes | No | Yes |
| [Artificial Algae Algorithm](https://doi.org/10.1016/j.asoc.2015.03.003) | `aaa` | swarm | Yes | No | Yes | Yes |
| [Artificial Bee Colony Optimization](https://doi.org/10.1007/s10898-007-9149-x) | `abco` | swarm | Yes | Yes | No | Yes |
| [Artificial Ecosystem Optimization](https://doi.org/10.1007/s00521-019-04452-x) | `aeo` | human | Yes | Yes | No | Yes |
| [Artificial Electric Field Algorithm](https://doi.org/10.1016/j.swevo.2019.03.013) | `aefa` | physics | Yes | Yes | No | Yes |
| [Artificial Fish Swarm Algorithm](https://doi.org/10.1007/s10462-012-9342-2) | `afsa` | swarm | Yes | Yes | No | Yes |
| [Artificial Gorilla Troops Optimizer](https://doi.org/10.1002/int.22535) | `agto` | swarm | Yes | Yes | No | Yes |
| [Artificial Hummingbird Algorithm](https://doi.org/10.1016/j.cma.2021.114194) | `aha` | swarm | Yes | Yes | No | Yes |
| [Artificial Lemming Algorithm](https://doi.org/10.1007/s10462-024-11023-7) | `ala` | swarm | Yes | Yes | No | Yes |
| [Artificial Protozoa Optimizer](https://doi.org/10.1016/j.knosys.2024.111737) | `apo` | swarm | Yes | Yes | No | Yes |
| [Artificial Rabbits Optimization](https://doi.org/10.1016/j.engappai.2022.105082) | `aro` | swarm | Yes | Yes | No | Yes |
| [Atom Search Optimization](https://doi.org/10.1016/j.knosys.2018.08.030) | `aso_atom` | physics | Yes | Yes | No | Yes |
| [Automated Design of Variation Operators](https://doi.org/10.1145/3712256.3726456) | `autov` | evolutionary | Yes | Yes | No | Yes |
| [Bacterial Chemotaxis Optimizer](https://doi.org/10.1007/s13369-025-10749-y) | `bco` | nature | Yes | Yes | No | Yes |
| [Bacterial Foraging Optimization](https://doi.org/10.1109/MCS.2002.1004010) | `bfo` | swarm | Yes | Yes | No | Yes |
| [Bald Eagle Search](https://doi.org/10.1007/s10462-019-09732-5) | `bes` | swarm | Yes | Yes | No | Yes |
| [Barnacles Mating Optimizer](https://doi.org/10.1016/j.engappai.2019.103330) | `bmo` | swarm | Yes | Yes | No | Yes |
| [Bat Algorithm](https://doi.org/10.1007/978-3-642-12538-6_6) | `bat_a` | swarm | Yes | Yes | No | Yes |
| [Battle Royale Optimization](https://doi.org/10.1007/s00521-020-05004-4) | `bro` | human | Yes | Yes | No | Yes |
| [Bees Algorithm](https://doi.org/10.1016/B978-008045157-2/50081-X) | `bea` | swarm | Yes | Yes | No | Yes |
| [BFGS Quasi-Newton Method](https://doi.org/10.1090/S0025-5718-1970-0274029-X) | `bfgs` | math | No | No | No | No |
| [Binary Space Partition Tree Genetic Algorithm](https://doi.org/10.1016/j.ins.2019.10.016) | `bspga` | evolutionary | Yes | Yes | No | Yes |
| [Biogeography-Based Optimization](https://doi.org/10.1109/TEVC.2008.919004) | `bbo` | evolutionary | Yes | Yes | No | Yes |
| [Bird Swarm Algorithm](https://doi.org/10.1080/0952813X.2015.1042530) | `bsa` | swarm | Yes | Yes | No | Yes |
| [Black Widow Optimization](https://doi.org/10.1016/j.engappai.2019.103249) | `bwo` | evolutionary | Yes | Yes | No | Yes |
| [Black-winged Kite Algorithm](https://doi.org/10.1007/s10462-024-10723-4) | `bka` | swarm | Yes | Yes | No | Yes |
| [Bonobo Optimizer](https://doi.org/10.1007/s10489-021-02444-w) | `bono` | swarm | Yes | Yes | No | Yes |
| [Brain Storm Optimization](https://doi.org/10.1007/978-3-642-21515-5_36) | `bso` | human | Yes | Yes | No | Yes |
| [Brown-Bear Optimization Algorithm](https://doi.org/10.1201/9781003337003-6) | `bboa` | swarm | Yes | Yes | No | Yes |
| [Butterfly Optimization Algorithm](https://doi.org/10.1007/s00500-018-3102-4) | `boa` | swarm | Yes | Yes | No | Yes |
| [Camel Algorithm](https://doi.org/10.13140/RG.2.2.21814.56649) | `camel` | swarm | Yes | Yes | No | Yes |
| [Capuchin Search Algorithm](https://doi.org/10.1007/s00521-020-05145-6) | `capsa` | swarm | Yes | Yes | No | Yes |
| [Cat Swarm Optimization](https://doi.org/10.1007/978-3-540-36668-3_94) | `cat_so` | swarm | Yes | Yes | No | Yes |
| [Chameleon Swarm Algorithm](https://doi.org/10.1016/j.eswa.2021.114685) | `chameleon_sa` | swarm | Yes | Yes | No | Yes |
| [Chaos Game Optimization](https://doi.org/10.1007/s10462-020-09867-w) | `cgo` | math | Yes | Yes | No | Yes |
| [Cheetah Based Optimization](https://doi.org/10.1038/s41598-022-14338-z) | `cddo` | swarm | Yes | Yes | No | Yes |
| [Cheetah Optimizer](https://doi.org/10.1038/s41598-022-14338-z) | `cdo` | swarm | Yes | Yes | No | Yes |
| [Chicken Swarm Optimization](https://doi.org/10.1007/978-3-319-11857-4_10) | `chicken_so` | swarm | Yes | No | No | Yes |
| [Child Drawing Development Optimization Algorithm](https://doi.org/10.1016/j.knosys.2024.111558) | `cddo_child` | human | Yes | Yes | No | Yes |
| [Chimp Optimization Algorithm](https://doi.org/10.1016/j.eswa.2020.113338) | `choa` | swarm | Yes | Yes | No | Yes |
| [Chernobyl Disaster Optimizer](https://doi.org/10.1016/j.compstruc.2023.107488) | `cdo_chernobyl` | physics | Yes | Yes | No | Yes |
| [Circle-Based Search Algorithm](https://doi.org/10.3390/math10101626) | `circle_sa` | math | Yes | Yes | No | Yes |
| [Circulatory System Based Optimization](https://doi.org/10.1016/j.egyr.2025.04.007) | `csbo` | swarm | Yes | Yes | No | Yes |
| [Clonal Selection Algorithm](https://doi.org/10.1109/TEVC.2002.1011539) | `clonalg` | evolutionary | Yes | Yes | No | Yes |
| [Coati Optimization Algorithm](https://doi.org/10.1016/j.knosys.2022.110011) | `coati_oa` | swarm | Yes | Yes | No | Yes |
| [Cockroach Swarm Optimization](https://doi.org/10.1109/ICCET.2010.5485993) | `cockroach_so` | swarm | Yes | Yes | No | Yes |
| [Competitive Swarm Optimizer](https://doi.org/10.1016/j.swevo.2024.101543) | `cso` | swarm | Yes | Yes | No | Yes |
| [COOT Bird Optimization](https://doi.org/10.1016/j.eswa.2021.115352) | `coot` | swarm | Yes | Yes | No | Yes |
| [Coral Reefs Optimization](https://doi.org/10.1155/2014/739768) | `cro` | evolutionary | Yes | Yes | No | Yes |
| [Coronavirus Herd Immunity Optimization](https://doi.org/10.1007/s00521-020-05296-6) | `chio` | human | Yes | Yes | No | Yes |
| [Cosmic Evolution Optimization](https://doi.org/10.1007/s00521-025-11234-6) | `ceo_cosmic` | physics | Yes | Yes | No | Yes |
| [Covariance Matrix Adaptation Evolution Strategy](https://doi.org/10.1109/ICEC.1996.542381) | `cmaes` | evolutionary | Yes | Yes | No | Yes |
| [Coyote Optimization Algorithm](https://doi.org/10.1109/CEC.2018.8477769) | `coa` | swarm | Yes | Yes | No | Yes |
| [Crayfish Optimization Algorithm](https://doi.org/10.1007/s10462-023-10567-4) | `crayfish_oa` | swarm | Yes | Yes | No | Yes |
| [Cross Entropy Method](https://doi.org/10.1007/978-1-4757-4321-0) | `cem` | distribution | Yes | Yes | No | Yes |
| [Crow Search Algorithm](https://doi.org/10.1016/j.compstruc.2016.03.001) | `csa` | swarm | Yes | Yes | No | Yes |
| [Cuckoo Search](https://doi.org/10.1109/NABIC.2009.5393690) | `cuckoo_s` | swarm | Yes | Yes | No | Yes |
| [Cultural Algorithm](https://doi.org/10.1080/00207160.2015.1067309) | `ca` | evolutionary | Yes | Yes | No | Yes |
| [Dandelion Optimizer](https://doi.org/10.1016/j.engappai.2022.105075) | `do_dandelion` | physics | Yes | Yes | No | Yes |
| [Deep Sleep Optimiser](https://doi.org/10.1109/ACCESS.2023.3298105) | `dso` | human | Yes | Yes | No | Yes |
| [Deer Hunting Optimization Algorithm](https://doi.org/10.1093/comjnl/bxy133) | `doa` | human | Yes | Yes | No | Yes |
| [Differential Evolution](https://doi.org/10.1023/A:1008202821328) | `de` | evolutionary | Yes | Yes | No | Yes |
| [Differential Evolution MTS](https://doi.org/10.1109/CEC.2009.4983179) | `hde` | evolutionary | Yes | Yes | No | Yes |
| [Dispersive Fly Optimization](https://doi.org/10.15439/2014F142) | `dfo` | swarm | Yes | Yes | No | Yes |
| [Dolphin Echolocation Optimization](https://doi.org/10.1016/j.advengsoft.2016.05.002) | `deo_dolphin` | swarm | Yes | Yes | No | Yes |
| [Dragonfly Algorithm](https://doi.org/10.1007/s00521-015-1920-1) | `da` | swarm | Yes | Yes | No | Yes |
| [Dung Beetle Optimizer](https://doi.org/10.1007/s11227-022-04959-6) | `dbo` | swarm | Yes | Yes | No | Yes |
| [Dwarf Mongoose Optimization Algorithm](https://doi.org/10.1016/j.cma.2022.114570) | `dmoa` | swarm | Yes | Yes | No | Yes |
| [Dynamic Differential Annealed Optimization](https://doi.org/10.1016/j.asoc.2020.106392) | `ddao` | physics | Yes | Yes | No | Yes |
| [Dynamic Virtual Bats Algorithm](https://doi.org/10.1109/INCoS.2014.40) | `dvba` | swarm | Yes | Yes | No | Yes |
| [Earthworm Optimization Algorithm](https://doi.org/10.1504/IJBIC.2015.10004283) | `eoa` | swarm | Yes | Yes | No | Yes |
| [Ecological Cycle Optimizer](https://doi.org/10.48550/arXiv.2508.20458) | `ecological_cycle_o` | swarm | Yes | Yes | No | Yes |
| [Educational Competition Optimizer](https://doi.org/10.3390/biomimetics10030176) | `eco` | human | Yes | Yes | No | Yes |
| [Efficient Global Optimization](https://doi.org/10.1023/A:1008306431147) | `ego` | distribution | Yes | Yes | No | Yes |
| [Egret Swarm Optimization Algorithm](https://doi.org/10.3390/biomimetics7040144) | `esoa` | swarm | Yes | Yes | No | Yes |
| [Electric Charged Particles Optimization](https://doi.org/10.1007/s10462-020-09890-x) | `ecpo` | physics | Yes | Yes | No | Yes |
| [Electrical Storm Optimization](https://doi.org/10.3390/make7010024) | `eso` | physics | Yes | Yes | No | Yes |
| [Electromagnetic Field Optimization](https://doi.org/10.1016/j.swevo.2015.07.002) | `efo` | physics | Yes | Yes | No | Yes |
| [Elephant Herding Optimization](https://doi.org/10.1109/ISCBI.2015.8) | `eho` | swarm | Yes | Yes | No | Yes |
| [Elk Herd Optimizer](https://doi.org/10.1007/s10462-023-10680-4) | `elk_ho` | swarm | Yes | Yes | No | Yes |
| [Emperor Penguin Colony](https://doi.org/10.1016/j.knosys.2018.06.001) | `epc` | swarm | Yes | Yes | No | Yes |
| [Energy Valley Optimizer](https://doi.org/10.1038/s41598-022-27344-y) | `evo` | physics | Yes | Yes | No | Yes |
| [Enzyme Activity Optimizer](https://doi.org/10.1007/s11227-025-07052-w) | `eao` | nature | Yes | Yes | No | Yes |
| [Equilibrium Optimizer](https://doi.org/10.1016/j.knosys.2019.105190) | `eo` | physics | Yes | Yes | No | Yes |
| [Escape Algorithm](https://doi.org/10.1007/s10462-024-11008-6) | `esc` | human | Yes | Yes | No | Yes |
| [Evolution Strategy (mu + lambda)](https://doi.org/10.1023/A:1015059928466) | `es` | evolutionary | Yes | Yes | No | Yes |
| [Evolutionary Programming](https://doi.org/10.1007/BF00175356) | `ep` | evolutionary | Yes | Yes | No | Yes |
| [Exponential Distribution Optimizer](https://doi.org/10.1007/s10462-023-10403-9) | `edo` | math | Yes | Yes | No | Yes |
| [Exponential-Trigonometric Optimization](https://doi.org/10.1016/j.cma.2024.117411) | `eto` | math | Yes | Yes | No | Yes |
| [Fast Evolutionary Programming](https://doi.org/10.1109/4235.771163) | `fep` | evolutionary | Yes | Yes | No | Yes |
| [FATA Geophysics Optimizer](https://doi.org/10.1016/j.neucom.2024.128289) | `fata` | physics | Yes | Yes | No | Yes |
| [Feasibility Rule with Objective Function Information](https://doi.org/10.1109/TCYB.2015.2493239) | `frofi` | evolutionary | Yes | Yes | No | Yes |
| [Fennec Fox Optimizer](https://doi.org/10.1109/ACCESS.2022.3197745) | `ffo` | swarm | Yes | Yes | No | Yes |
| [Fick's Law Algorithm](https://doi.org/10.1016/j.knosys.2022.110146) | `fla` | physics | Yes | Yes | No | Yes |
| [Firefly Algorithm](https://doi.org/10.1504/IJBIC.2010.032124) | `firefly_a` | swarm | Yes | Yes | No | Yes |
| [Fireworks Algorithm](https://doi.org/10.1016/j.asoc.2017.10.046) | `fwa` | swarm | Yes | Yes | No | Yes |
| [Fish School Search](https://doi.org/10.1109/ICSMC.2008.4811695) | `fss` | swarm | Yes | Yes | No | Yes |
| [Fitness Dependent Optimizer](https://doi.org/10.1109/ACCESS.2019.2907012) | `fdo` | swarm | Yes | Yes | No | Yes |
| [Fletcher-Reeves Conjugate Gradient](https://doi.org/10.1002/er.8067) | `frcg` | math | No | No | No | No |
| [Flood Algorithm](https://doi.org/10.1007/s11227-024-06291-7) | `flood_a` | physics | Yes | Yes | No | Yes |
| [Flow Direction Algorithm](https://doi.org/10.1016/j.cie.2021.107224) | `fda` | swarm | Yes | Yes | No | Yes |
| [Flower Pollination Algorithm](https://doi.org/10.1007/978-3-642-32894-7_27) | `fpa` | swarm | Yes | Yes | No | Yes |
| [Forensic-Based Investigation Optimization](https://doi.org/10.1016/j.asoc.2020.106339) | `fbio` | human | Yes | Yes | No | Yes |
| [Forest Optimization Algorithm](https://doi.org/10.1016/j.eswa.2014.05.009) | `foa` | swarm | Yes | Yes | No | Yes |
| [Fossa Optimization Algorithm](https://doi.org/10.1007/s10462-024-10953-0) | `foa_fossa` | swarm | Yes | Yes | No | Yes |
| [Fox Optimizer](https://doi.org/10.1007/s10489-022-03533-0) | `fox` | swarm | Yes | Yes | No | Yes |
| [Fruit-Fly Algorithm](https://doi.org/10.1016/j.knosys.2011.07.001) | `ffa` | swarm | Yes | Yes | No | Yes |
| [Gaining-Sharing Knowledge Algorithm](https://doi.org/10.1007/s13042-019-01053-x) | `gska` | human | Yes | Yes | No | Yes |
| [Gazelle Optimization Algorithm](https://doi.org/10.1007/s00521-022-07854-6) | `gazelle_oa` | swarm | Yes | Yes | No | Yes |
| [Gekko Japonicus Algorithm](https://doi.org/10.1016/j.eswa.2025.127982) | `gja` | swarm | Yes | Yes | No | Yes |
| [Generalized Normal Distribution Optimizer](https://doi.org/10.1016/j.enconman.2020.113301) | `gndo` | math | Yes | Yes | No | Yes |
| [Genetic Algorithm](https://doi.org/10.7551/mitpress/1090.001.0001) | `ga` | evolutionary | Yes | Yes | No | Yes |
| [Genghis Khan Shark Optimizer](https://doi.org/10.1016/j.aei.2023.102210) | `gkso` | swarm | Yes | Yes | No | Yes |
| [Geometric Mean Optimizer](https://doi.org/10.1007/s00500-023-08202-z) | `gmo` | swarm | Yes | Yes | No | Yes |
| [Germinal Center Optimization](https://doi.org/10.1016/j.ifacol.2018.07.300) | `gco` | human | Yes | Yes | No | Yes |
| [Geyser Inspired Algorithm](https://doi.org/10.1007/s42235-023-00437-8) | `gea` | physics | Yes | Yes | No | Yes |
| [Giant Pacific Octopus Optimizer](https://doi.org/10.1007/s12065-024-00945-4) | `gpoo`  | swarm  | Yes | No | No | Yes |
| [Giant Trevally Optimizer](https://doi.org/10.1109/ACCESS.2022.3223388) | `gto` | swarm | Yes | Yes | No | Yes |
| [Glider Snake Optimization](https://doi.org/10.1007/s10462-026-11504-x) | `gso_glider_snake` | swarm | Yes | No | No | No |
| [Glowworm Swarm Optimization](https://doi.org/10.1007/978-3-319-51595-3) | `gso` | swarm | Yes | Yes | No | Yes |
| [Golden Jackal Optimizer](https://doi.org/10.1016/j.eswa.2022.116924) | `gjo` | swarm | Yes | Yes | No | Yes |
| [Gradient-Based Optimizer](https://doi.org/10.1007/s11831-022-09872-y) | `gbo` | math | Yes | Yes | No | Yes |
| [Gradient-Based Particle Swarm Optimization](https://doi.org/10.48550/arXiv.2312.09703) | `gpso` | swarm | Yes | Yes | No | Yes |
| [Grasshopper Optimization Algorithm](https://doi.org/10.1016/j.advengsoft.2017.01.004) | `goa` | swarm | Yes | Yes | No | Yes |
| [Gravitational Search Algorithm](https://doi.org/10.1016/j.ins.2009.03.004) | `gsa` | physics | Yes | Yes | No | Yes |
| [Grey Wolf Optimizer](https://doi.org/10.1016/j.advengsoft.2013.12.007) | `gwo` | swarm | Yes | Yes | No | Yes |
| [Greylag Goose Optimization](https://doi.org/10.1016/j.eswa.2023.122147) | `ggo` | swarm | Yes | Yes | No | Yes |
| [Growth Optimizer](https://doi.org/10.1016/j.knosys.2022.110206) | `go_growth` | swarm | Yes | Yes | No | Yes |
| [Harmony Search Algorithm](https://doi.org/10.1177/003754970107600201) | `hsa` | trajectory | Yes | No | No | Yes |
| [Harris Hawks Optimization](https://doi.org/10.1016/j.future.2019.02.028) | `hho` | swarm | Yes | Yes | No | Yes |
| [Heap-Based Optimizer](https://doi.org/10.1016/j.eswa.2020.113702) | `hbo` | human | Yes | Yes | No | Yes |
| [Henry Gas Solubility Optimization](https://doi.org/10.1016/j.future.2019.07.015) | `hgso` | physics | Yes | Yes | No | Yes |
| [Hiking Optimization Algorithm](https://doi.org/10.1016/j.knosys.2024.111880) | `hiking_oa` | human | Yes | Yes | No | Yes |
| [Hill Climb Algorithm](https://doi.org/10.1007/978-3-540-75256-1_52) | `hc` | trajectory | No | No | No | No |
| [Hippopotamus Optimization Algorithm](https://doi.org/10.1038/s41598-024-54910-3) | `ho_hippo` | swarm | Yes | Yes | No | Yes |
| [Honey Badger Algorithm](https://doi.org/10.1016/j.matcom.2021.08.013) | `hba_honey` | swarm | Yes | Yes | No | Yes |
| [Horse Herd Optimization Algorithm](https://doi.org/10.1016/j.knosys.2020.106711) | `horse_oa` | swarm | Yes | Yes | No | Yes |
| [Human Conception Optimizer](https://doi.org/10.1038/s41598-022-25031-6) | `hco` | human | Yes | Yes | No | Yes |
| [Human Evolutionary Optimization Algorithm](https://doi.org/10.1016/j.eswa.2023.122638) | `heoa` | human | Yes | Yes | No | Yes |
| [Hunger Games Search](https://doi.org/10.1016/j.eswa.2021.114864) | `hgs` | swarm | Yes | Yes | No | Yes |
| [Hunting Search Algorithm](https://doi.org/10.1109/ICSCCW.2009.5379451) | `hus` | swarm | Yes | Yes | No | Yes |
| [Hybrid Bat Algorithm](https://doi.org/10.48550/arXiv.1303.6310) | `hba` | swarm | Yes | Yes | No | Yes |
| [Hybrid Self-Adaptive Bat Algorithm](https://doi.org/10.1155/2014/709738) | `hsaba` | swarm | Yes | Yes | No | Yes |
| [Imperialist Competitive Algorithm](https://doi.org/10.1109/CEC.2007.4425083) | `ica` | human | Yes | Yes | No | Yes |
| [Improved Adaptive Grey Wolf Optimization](https://doi.org/10.1007/s10462-024-10821-3) | `iagwo` | swarm  | Yes | No | No | No|
| [Improved Grey Wolf Optimizer](https://doi.org/10.1016/j.eswa.2020.113917) | `i_gwo` | swarm | Yes | Yes | No | Yes |
| [Improved Kepler Optimization Algorithm](https://doi.org/10.1016/j.eswa.2025.128216) | `ikoa` | physics | Yes | Yes | No | Yes |
| [Improved L-SHADE](https://doi.org/10.1109/CEC.2016.7743922) | `ilshade` | evolutionary | Yes | Yes | No | Yes |
| [Improved Multi-Operator Differential Evolution](https://doi.org/10.1109/CEC48606.2020.9185577) | `imode` | evolutionary | Yes | Yes | No | Yes |
| [Improved Whale Optimization Algorithm](https://doi.org/10.1016/j.jcde.2019.02.002) | `i_woa` | swarm | Yes | Yes | No | Yes |
| [Invasive Weed Optimization](https://doi.org/10.1016/j.ecoinf.2006.07.003) | `iwo` | nature | Yes | Yes | No | Yes |
| [Ivy Algorithm](https://doi.org/10.1016/j.knosys.2024.111850) | `ivya` | nature | Yes | Yes | No | Yes |
| [Jaya Algorithm](https://doi.org/10.5267/j.ijiec.2015.8.004) | `jy` | swarm | Yes | Yes | No | Yes |
| [Jellyfish Search Optimizer](https://doi.org/10.1016/j.amc.2020.125535) | `jso` | swarm | Yes | Yes | No | Yes |
| [Komodo Mlipir Algorithm](https://doi.org/10.1016/j.asoc.2021.108043) | `kma` | swarm | Yes | Yes | No | Yes |
| [Krill Herd Algorithm](https://doi.org/10.1016/j.asoc.2016.08.041) | `kha` | swarm | Yes | No | No | Yes |
| [Leaf in Wind Optimization](https://doi.org/10.1109/ACCESS.2024.3390670) | `liwo` | physics | Yes | Yes | No | Yes |
| [Lévy Flight Distribution](https://doi.org/10.1016/j.engappai.2020.103731) | `lfd` | swarm | Yes | Yes | No | Yes |
| [LSHADE-cnEpSin](https://doi.org/10.1109/CEC.2016.7744173) | `lshade_cnepsin` | evolutionary | Yes | Yes | No | Yes |
| [Life Choice-Based Optimizer](https://doi.org/10.1007/s00500-019-04443-z) | `lco` | human | Yes | Yes | No | Yes |
| [Light Spectrum Optimizer](https://doi.org/10.1016/j.asoc.2024.112318) | `lso_spectrum` | physics | Yes | Yes | No | Yes |
| [Linear Subspace Surrogate Modeling Evolutionary Algorithm](https://doi.org/10.1109/TEVC.2023.3319640) | `l2smea` | evolutionary | Yes | Yes | No | Yes |
| [Lion Optimization Algorithm](https://doi.org/10.1016/j.jcde.2015.06.003) | `loa` | swarm | Yes | Yes | No | Yes |
| [Liver Cancer Algorithm](https://doi.org/10.1016/j.compbiomed.2023.107389) | `lca` | nature | Yes | Yes | No | Yes |
| [Lungs Performance-Based Optimization](https://doi.org/10.1016/j.cma.2023.116582) | `lpo` | nature | Yes | Yes | No | Yes |
| [Lyrebird Optimization Algorithm](https://doi.org/10.1016/j.cma.2023.116436) | `loa_lyrebird` | swarm | Yes | Yes | No | Yes |
| [Manta Ray Foraging Optimization](https://doi.org/10.1016/j.engappai.2019.103300) | `mrfo` | swarm | Yes | Yes | No | Yes |
| [Mantis Shrimp Optimization Algorithm](https://doi.org/10.3390/math13091500) | `mshoa` | swarm | Yes | Yes | No | Yes |
| [Marine Predators Algorithm](https://doi.org/10.1016/j.eswa.2020.113377) | `mpa` | swarm | Yes | Yes | No | Yes |
| [Market Game Optimization Algorithm](https://doi.org/10.1016/j.asoc.2024.112466) | `mgoa_market` | human | Yes | Yes | No | Yes |
| [Memetic Algorithm](https://doi.org/10.1007/978-3-540-92910-9_29) | `memetic_a` | evolutionary | Yes | Yes | No | Yes |
| [Mirage-Search Optimizer](https://doi.org/10.1016/j.advengsoft.2025.103883) | `mso` | physics | Yes | Yes | No | Yes |
| [Monarch Butterfly Optimization](https://doi.org/10.1007/s00521-015-1923-y) | `mbo` | swarm | Yes | Yes | No | Yes |
| [Monkey King Evolution V1](https://doi.org/10.1016/j.knosys.2016.01.009) | `mke` | evolutionary | Yes | Yes | No | Yes |
| [Moss Growth Optimization](https://doi.org/10.1093/jcde/qwae080) | `moss_go` | nature | Yes | Yes | No | Yes |
| [Most Valuable Player Algorithm](https://doi.org/10.1007/s12351-017-0320-y) | `mvpa` | human | Yes | Yes | No | Yes |
| [Moth Flame Algorithm](https://doi.org/10.1016/j.knosys.2015.07.006) | `mfa` | swarm | Yes | Yes | No | Yes |
| [Moth Search Algorithm](https://doi.org/10.1007/s12293-016-0212-3) | `msa_e` | swarm | Yes | Yes | No | Yes |
| [Mountain Gazelle Optimizer](https://doi.org/10.1016/j.advengsoft.2022.103282) | `mgo` | swarm | Yes | Yes | No | Yes |
| [Multi-Surrogate-Assisted Ant Colony Optimization](https://doi.org/10.1109/TCYB.2021.3064676) | `misaco` | swarm | Yes | Yes | No | Yes |
| [Multi-Verse Optimizer](https://doi.org/10.1007/s00521-015-1870-7) | `mvo` | swarm | Yes | Yes | No | Yes |
| [Multifactorial Evolutionary Algorithm](https://doi.org/10.1109/TEVC.2015.2458037) | `mfea` | evolutionary | Yes | Yes | No | Yes |
| [Multifactorial Evolutionary Algorithm II](https://doi.org/10.1109/TEVC.2019.2906927) | `mfea2` | evolutionary | Yes | Yes | No | Yes |
| [Multiple Trajectory Search](https://doi.org/10.1109/CEC.2008.4631210) | `mts` | trajectory | Yes | Yes | No | Yes |
| [Multiswarm-Assisted Expensive Optimization](https://doi.org/10.1109/TCYB.2020.2967553) | `samso` | swarm | Yes | Yes | No | Yes |
| [Naked Mole-Rat Algorithm](https://doi.org/10.1007/s00521-019-04464-7) | `nmra` | swarm | Yes | Yes | No | Yes |
| [Narwhal Optimizer](https://doi.org/10.1038/s41598-024-61278-8) | `nwoa` | swarm | Yes | Yes | No | Yes |
| [Nelder-Mead Method](https://doi.org/10.1093/comjnl/7.4.308) | `nmm` | trajectory | Yes | Yes | No | Yes |
| [Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO)](https://doi.org/10.1109/TEVC.2024.3400398) | `nndrea_so` | evolutionary | Yes | Yes | No | Yes |
| [Nizar Optimization Algorithm](https://doi.org/10.1007/s11227-023-05579-4) | `noa` | math | Yes | Yes | No | Yes |
| [Northern Goshawk Optimization](https://doi.org/10.1109/ACCESS.2021.3133286) | `ngo` | swarm | Yes | Yes | No | Yes |
| [Nuclear Reaction Optimization](https://doi.org/10.1109/ACCESS.2019.2918406) | `nro` | physics | Yes | Yes | No | Yes |
| [Numeric Crunch Algorithm](https://doi.org/10.1007/s00500-023-08925-z) | `nca` | math | Yes | Yes | No | Yes |
| [Optimal Foraging Algorithm](https://doi.org/10.1016/j.eswa.2022.117735) | `ofa` | swarm | Yes | Yes | No | Yes |
| [Osprey Optimization Algorithm](https://doi.org/10.3389/fmech.2022.1126450) | `ooa` | swarm | Yes | Yes | No | Yes |
| [Parameter-Free Bat Algorithm](https://www.iztok-jr-fister.eu/static/publications/124.pdf) | `plba` | swarm | Yes | Yes | No | Yes |
| [Parent-Centric Crossover (G3-PCX style)](https://doi.org/10.1109/CEC.2004.1331141) | `pcx` | evolutionary | Yes | Yes | No | Yes |
| [Pareto Sequential Sampling](https://doi.org/10.1007/s00500-021-05853-8) | `pss` | math | Yes | Yes | No | Yes |
| [Parrot Optimizer](https://doi.org/10.1016/j.compbiomed.2024.108064) | `parrot_o` | swarm | Yes | Yes | No | Yes |
| [Particle Swarm Optimization](https://doi.org/10.1109/ICNN.1995.488968) | `pso` | swarm | Yes | Yes | No | Yes |
| [Pathfinder Algorithm](https://doi.org/10.1016/j.asoc.2019.03.012) | `pfa` | swarm | Yes | Yes | No | Yes |
| [Pelican Optimization Algorithm](https://doi.org/10.3390/s22030855) | `poa` | swarm | Yes | Yes | No | Yes |
| [Physical Education Teacher Inspired Optimization](https://doi.org/10.13140/RG.2.2.12097.06245)| `petio` | human  | Yes | No | No | Yes |
| [Pied Kingfisher Optimizer](https://doi.org/10.1007/s00521-024-09879-5) | `pko` | swarm | Yes | Yes | No | Yes |
| [Polar Fox Optimization](https://doi.org/10.1007/s00521-024-10346-4)| `pfa_polar_fox` | swarm | Yes | No | No | No |
| [Polar Lights Optimizer](https://doi.org/10.1016/j.neucom.2024.128427) | `plo` | physics | Yes | Yes | No | Yes |
| [Political Optimizer](https://doi.org/10.1016/j.knosys.2020.105709) | `political_o` | human | Yes | Yes | No | Yes |
| [Poor and Rich Optimization Algorithm](https://doi.org/10.1016/j.engappai.2019.08.025) | `pro` | human | Yes | Yes | No | Yes |
| [Population-Based Incremental Learning](https://doi.org/10.1109/SSE62657.2024.00022) | `pbil` | distribution | No | No | No | No |
| [Prairie Dog Optimization Algorithm](https://doi.org/10.1007/s00521-022-07530-9) | `pdo` | swarm | Yes | Yes | No | Yes |
| [Puma Optimizer](https://doi.org/10.1007/s10586-023-04221-5) | `puma_o` | swarm | Yes | Yes | No | Yes |
| [Quadratic Interpolation Optimization](https://doi.org/10.1016/j.cma.2023.116446) | `qio` | math | Yes | Yes | No | Yes |
| [Queuing Search Algorithm](https://doi.org/10.1007/s12652-020-02849-4) | `qsa` | human | Yes | Yes | No | Yes |
| [Random Search](https://doi.org/10.1016/j.advengsoft.2022.103141) | `random_s` | trajectory | Yes | Yes | No | Yes |
| [Rat Swarm Optimizer](https://doi.org/10.1007/s12652-020-02580-0) | `rso` | swarm | Yes | Yes | No | Yes |
| [Red-billed Blue Magpie Optimizer](https://doi.org/10.1007/s10462-024-10716-3) | `rbmo` | swarm | Yes | Yes | No | Yes |
| [Remora Optimization Algorithm](https://doi.org/10.1016/j.eswa.2021.115665) | `roa` | swarm | Yes | Yes | No | Yes |
| [Reptile Search Algorithm](https://doi.org/10.1016/j.eswa.2021.116158) | `rsa` | swarm | Yes | Yes | No | Yes |
| [RIME-ice Algorithm](https://doi.org/10.1016/j.neucom.2023.02.010) | `rime` | physics | Yes | Yes | No | Yes |
| [RMSProp](https://www.youtube.com/watch?v=defQQqkXEfE) | `rmsprop` | math | No | No | No | No |
| [Rock Hyraxes Swarm Optimization](https://doi.org/10.32604/cmc.2021.013648)   | `rhso`  | swarm  | Yes | No | No | Yes |
| [RUNge Kutta Optimizer](https://doi.org/10.1016/j.eswa.2021.115079) | `run` | math | Yes | Yes | No | Yes |
| [Rüppell's Fox Optimizer](https://doi.org/10.1007/s10586-024-04950-1) | `rfo` | swarm | Yes | Yes | No | Yes |
| [Sailfish Optimizer](https://doi.org/10.1016/j.engappai.2019.01.001) | `sfo` | swarm | Yes | Yes | No | Yes |
| [Salp Swarm Algorithm](https://doi.org/10.1016/j.advengsoft.2017.07.002) | `ssa` | swarm | Yes | Yes | No | Yes |
| [Sammon Mapping Assisted Differential Evolution](https://doi.org/10.1016/j.petrol.2019.106633) | `sade_sammon` | evolutionary | Yes | Yes | No | Yes |
| [Sand Cat Swarm Optimization](https://doi.org/10.1007/s00366-022-01604-x) | `scso` | swarm | Yes | Yes | No | Yes |
| [Satin Bowerbird Optimizer](https://doi.org/10.1016/j.engappai.2017.01.006) | `sbo` | swarm | Yes | Yes | No | Yes |
| [Sea Lion Optimization](https://doi.org/10.14569/IJACSA.2019.0100548) | `slo` | swarm | Yes | Yes | No | Yes |
| [Seagull Optimization Algorithm](https://doi.org/10.1016/j.knosys.2018.11.024) | `soa` | swarm | Yes | Yes | No | Yes |
| [Seahorse Optimizer](https://doi.org/10.1007/s10489-022-03994-3) | `seaho` | swarm | Yes | Yes | No | Yes |
| [Search And Rescue Optimization](https://doi.org/10.1155/2019/2482543) | `saro` | human | Yes | Yes | No | Yes |
| [Search Space Independent Operator Based Deep Reinforcement Learning](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125444) | `ssio_rl` | evolutionary | Yes | Yes | No | Yes |
| [Secretary Bird Optimization Algorithm](https://doi.org/10.1007/s10462-024-10729-y) | `sboa` | swarm | Yes | Yes | No | Yes |
| [Self-Adaptive Bat Algorithm](https://doi.org/10.1155/2014/709738) | `saba` | swarm | Yes | Yes | No | Yes |
| [Self-Adaptive Differential Evolution](https://doi.org/10.1109/TEVC.2006.872133) | `jde` | evolutionary | Yes | Yes | No | Yes |
| [Sequential Quadratic Programming](https://doi.org/10.1017/S0962492900002518) | `sqp` | math | No | No | No | No |
| [Serval Optimization Algorithm](https://doi.org/10.3390/biomimetics7040204) | `serval_oa` | swarm | Yes | Yes | No | Yes |
| [Shuffle-based Runner-Root Algorithm](https://doi.org/10.1007/978-3-319-70139-4_16) | `srsr` | swarm | Yes | Yes | No | Yes |
| [Siberian Tiger Optimization](https://doi.org/10.1109/ACCESS.2022.3229964) | `sto` | swarm | Yes | Yes | No | Yes |
| [Simulated Annealing](https://doi.org/10.1126/science.220.4598.671) | `sa` | trajectory | No | Yes | Yes | No |
| [Sine Cosine Algorithm](https://doi.org/10.1016/j.knosys.2015.12.022) | `sine_cosine_a` | swarm | Yes | Yes | No | Yes |
| [Sinh Cosh Optimizer](https://doi.org/10.1016/j.knosys.2023.111081) | `scho` | math | Yes | Yes | No | Yes |
| [Slime Mould Algorithm](https://doi.org/10.1016/j.future.2020.03.055) | `sma` | nature | Yes | Yes | No | Yes |
| [Snake Optimizer](https://doi.org/10.1016/j.knosys.2022.108320) | `so_snake` | swarm | Yes | Yes | No | Yes |
| [Snow Ablation Optimizer](https://doi.org/10.1016/j.eswa.2023.120069) | `snow_oa` | physics | Yes | Yes | No | Yes |
| [Social Ski-Driver Optimization](https://doi.org/10.1007/s00521-019-04159-z) | `ssdo` | human | Yes | Yes | No | Yes |
| [Social Spider Algorithm](https://doi.org/10.1016/j.asoc.2015.02.014) | `sspider_a` | swarm | Yes | Yes | No | Yes |
| [Social Spider Swarm Optimizer](https://doi.org/10.1016/j.eswa.2013.05.041) | `sso` | swarm | Yes | Yes | No | Yes |
| [Sparrow Search Algorithm](https://doi.org/10.1080/21642583.2019.1708830) | `sparrow_sa` | swarm | Yes | Yes | No | Yes |
| [Spider Monkey Optimization](https://doi.org/10.1007/s12293-013-0128-0) | `smo` | swarm | Yes | Yes | No | Yes |
| [Spotted Hyena Inspired Optimizer](https://doi.org/10.1016/j.advengsoft.2017.05.014) | `shio` | swarm | Yes | Yes | No | Yes |
| [Spotted Hyena Optimizer](https://doi.org/10.1016/j.advengsoft.2017.05.014) | `sho` | swarm | Yes | Yes | No | Yes |
| [Squirrel Search Algorithm](https://doi.org/10.1016/j.swevo.2018.02.013) | `squirrel_sa` | swarm | Yes | Yes | No | Yes |
| [Starfish Optimization Algorithm](https://doi.org/10.1007/s00521-024-10694-1) | `sfoa` | swarm | Yes | Yes | No | Yes |
| [Steepest Descent](https://doi.org/10.1006/hmat.1996.2146) | `sd` | math | No | No | No | No |
| [Stellar Oscillator Optimization](https://doi.org/10.1007/s10586-024-04976-5) | `soo` | physics | Yes | Yes | No | Yes |
| [Student Psychology Based Optimization](https://doi.org/10.1016/j.advengsoft.2020.102804) | `spbo` | swarm | Yes | Yes | No | Yes |
| [Success-History Adaptive Differential Evolution](https://doi.org/10.1109/CEC.2014.6900380) | `shade` | evolutionary | Yes | Yes | No | Yes |
| [Success-History Intelligent Optimizer](https://doi.org/10.1016/j.cma.2024.117272) | `shio_success` | swarm | Yes | Yes | No | Yes |
| [Superb Fairy-wren Optimization Algorithm](https://doi.org/10.1007/s10586-024-04901-w) | `superb_foa` | swarm | Yes | Yes | No | Yes |
| [Supply-Demand-Based Optimization](https://doi.org/10.1109/ACCESS.2019.2919408) | `supply_do` | human | Yes | Yes | No | Yes |
| [Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II](https://doi.org/10.1007/978-3-319-97773-7_4) | `sacc_eam2` | evolutionary | Yes | Yes | No | Yes |
| [Surrogate-Assisted Cooperative Swarm Optimization](https://doi.org/10.1109/TEVC.2017.2675628) | `sacoso` | swarm | Yes | Yes | No | Yes |
| [Surrogate-Assisted DE with Adaptive Multi-Subspace Search](https://doi.org/10.1109/TEVC.2022.3226837) | `sade_amss` | evolutionary | Yes | Yes | No | Yes |
| [Surrogate-Assisted DE with Adaptive Training Data Selection Criterion](https://doi.org/10.1109/SSCI51031.2022.10022105) | `sade_atdsc` | evolutionary | Yes | Yes | No | Yes |
| [Surrogate-Assisted Partial Optimization](https://doi.org/10.1007/978-3-031-70068-2_24) | `sapo` | evolutionary | Yes | Yes | No | Yes |
| [Symbiotic Organisms Search](https://doi.org/10.1016/j.compstruc.2014.03.007) | `sos` | swarm | Yes | Yes | No | Yes |
| [Tabu Search](https://doi.org/10.1287/ijoc.1.3.190) | `ts` | trajectory | No | No | No | No |
| [Tasmanian Devil Optimization](https://doi.org/10.1109/ACCESS.2022.3151641) | `tdo` | swarm | Yes | Yes | No | Yes |
| [Teaching Learning Based Optimization](https://doi.org/10.1016/j.cad.2010.12.015) | `tlbo` | swarm | Yes | Yes | No | Yes |
| [Teamwork Optimization Algorithm](https://doi.org/10.3390/s21134567) | `toa` | human | Yes | Yes | No | Yes |
| [Termite Life Cycle Optimizer](https://doi.org/10.1016/j.eswa.2022.119211) | `tlco` | swarm | Yes | Yes | No | Yes |
| [Tianji Horse Racing Optimizer](https://doi.org/10.1007/s10462-025-11269-9) | `thro` | human | Yes | Yes | No | Yes |
| [Tornado Optimizer with Coriolis Force](https://doi.org/10.1007/s10462-025-11118-9) | `toc` | physics | Yes | Yes | No | Yes |
| [Tree Physiology Optimization](https://doi.org/10.1515/jisys-2017-0156) | `tpo` | nature | Yes | Yes | No | Yes |
| [Triangulation Topology Aggregation Optimizer](https://doi.org/10.1016/j.eswa.2023.121744) | `ttao` | math | Yes | Yes | No | Yes |
| [Tug of War Optimization](https://doi.org/10.1007/978-3-030-04067-3_11) | `two` | physics | Yes | Yes | No | Yes |
| [Tuna Swarm Optimization](https://doi.org/10.1155/2021/9210050) | `tso` | swarm | Yes | Yes | No | Yes |
| [Tunicate Swarm Algorithm](https://doi.org/10.1016/j.engappai.2020.103541) | `tsa` | swarm | Yes | Yes | No | Yes |
| [Virus Colony Search](https://doi.org/10.1016/j.advengsoft.2015.11.004) | `vcs` | swarm | Yes | Yes | No | Yes |
| [Walrus Optimization Algorithm](https://doi.org/10.1038/s41598-023-35863-5) | `waoa` | swarm | Yes | Yes | No | Yes |
| [War Strategy Optimization](https://doi.org/10.1109/ACCESS.2022.3153493) | `warso` | human | Yes | Yes | No | Yes |
| [Water Cycle Algorithm](https://doi.org/10.1016/j.compstruc.2012.07.010) | `wca` | nature | Yes | Yes | No | Yes |
| [Water Uptake and Transport in Plants](https://doi.org/10.1007/s00521-025-11228-z) | `wutp` | nature | Yes | Yes | No | Yes |
| [Wave Optimization Algorithm](https://doi.org/10.1016/j.cor.2014.10.008) | `wo_wave` | physics | Yes | Yes | No | Yes |
| [Weighting and Inertia Random Walk Optimizer](https://doi.org/10.1016/j.eswa.2022.116516) | `info` | math | Yes | Yes | No | Yes |
| [Whale Optimization Algorithm](https://doi.org/10.1016/j.advengsoft.2016.01.008) | `woa` | swarm | Yes | Yes | No | Yes |
| [White Shark Optimizer](https://doi.org/10.1016/j.knosys.2022.108457) | `wso` | swarm | Yes | Yes | No | Yes |
| [Wildebeest Herd Optimization](https://doi.org/10.3233/JIFS-190495) | `who` | swarm | Yes | Yes | No | Yes |
| [Wind Driven Optimization](https://doi.org/10.1109/APS.2010.5562213) | `wdo` | physics | Yes | Yes | No | Yes |
| [Young's Double-Slit Experiment Optimizer](https://doi.org/10.1016/j.cma.2022.115652) | `ydse` | physics | Yes | Yes | No | Yes |
| [Zebra Optimization Algorithm](https://doi.org/10.1109/ACCESS.2022.3172789) | `zoa` | swarm | Yes | Yes | No | Yes |

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

The table below summarizes the benchmark functions currently available in the library. The **Function** column reports the conventional function name, **ID** gives the callable identifier used in the codebase (when importing from `pymetaheuristic.src.test_functions`), **Domain** and  **Global Minimum** describes, when applicable, the corresponding decision vector, and the known global optimum in terms of objective value.

### Benchmark Function Optima

All functions below use the **minimization** convention.
**Notation**

| Symbol | Meaning |
|---|---|
| *D* | Number of decision variables. |
| *x*<sup>*</sup> | Global minimizer. |
| *f*<sup>*</sup> | Global minimum value. |
| 0<sub>D</sub> | *D*-dimensional vector of zeros. |
| 1<sub>D</sub> | *D*-dimensional vector of ones. |

### 2-Dimensional Functions

| Function | ID | Domain | Global Minimum |
|---|---|---|---|
| Ackley | `ackley` | [-32.768, 32.768]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Beale | `beale` | [-4.5, 4.5]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (3, 0.5) |
| Bohachevsky F1 | `bohachevsky_1` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Bohachevsky F2 | `bohachevsky_2` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Bohachevsky F3 | `bohachevsky_3` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Booth | `booth` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (1, 3) |
| Branin RCOS | `branin_rcos` | *x<sub>1</sub>* ∈ [-5, 10], *x<sub>2</sub>* ∈ [0, 15] | *f*<sup>*</sup> = 0.3978873577 at (-π, 12.275), (π, 2.275), (3π, 2.475) |
| Bukin F6 | `bukin_6` | *x<sub>1</sub>* ∈ [-15, -5], *x<sub>2</sub>* ∈ [-3, 3] | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (-10, 1) |
| Cross-in-Tray | `cross_in_tray` | [-10, 10]<sup>2</sup> | *f*<sup>*</sup> ≈ -2.0626118708 at *(x<sub>1</sub>, x<sub>2</sub>)* = (±1.349406609, ±1.349406609) |
| Drop-Wave | `drop_wave` | [-5.12, 5.12]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = -1; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Easom | `easom` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = -1; *(x<sub>1</sub>, x<sub>2</sub>)* = (π, π) |
| Eggholder | `eggholder` | [-512, 512]<sup>2</sup> | *f*<sup>*</sup> ≈ -959.6407; *(x<sub>1</sub>, x<sub>2</sub>)* ≈ (512, 404.2319) |
| Goldstein-Price | `goldstein_price` | [-2, 2]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 3; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, -1) |
| Himmelblau | `himmelblau` | [-5, 5]<sup>2</sup> | *f*<sup>*</sup> = 0 at (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126) |
| Hölder Table | `holder_table` | [-10, 10]<sup>2</sup> | *f*<sup>*</sup> ≈ -19.208502568 at *(x<sub>1</sub>, x<sub>2</sub>)* = (±8.055023472, ±9.664590029) |
| Levi F13 | `levi_13` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (1, 1) |
| Matyas | `matyas` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| McCormick | `mccormick` | *x<sub>1</sub>* ∈ [-1.5, 4], *x<sub>2</sub>* ∈ [-3, 4] | *f*<sup>*</sup> ≈ -1.913222955; *(x<sub>1</sub>, x<sub>2</sub>)* ≈ (-0.54719756, -1.54719756) |
| Schaffer F2 | `schaffer_2` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Schaffer F4 | `schaffer_4` | [-100, 100]<sup>2</sup> | *f*<sup>*</sup> ≈ 0.292578632 at (0, ±1.25313), (±1.25313, 0) |
| Schaffer F6 | `schaffer_6` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Six-Hump Camel Back | `six_hump_camel_back` | *x<sub>1</sub>* ∈ [-3, 3], *x<sub>2</sub>* ∈ [-2, 2] | *f*<sup>*</sup> ≈ -1.031628453 at (0.089842, -0.712656), (-0.089842, 0.712656) |
| Three-Hump Camel Back | `three_hump_camel_back` | [-5, 5]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |

### D-Dimensional Functions

| Function | ID | Domain | Global Minimum |
|---|---|---|---|
| Alpine 1 | `alpine_1` | [-10, 10]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = 0, *i* = 1, ..., *D* |
| Alpine 2 | `alpine_2` | [0, 10]<sup>D</sup> | *f*<sup>*</sup> ≈ -(2.808131180)<sup>D</sup>; *x<sub>i</sub>* ≈ 7.917052698 [N1] |
| Axis Parallel Hyper-Ellipsoid | `axis_parallel_hyper_ellipsoid` | [-5.12, 5.12]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = 0, *i* = 1, ..., *D* |
| Bent Cigar | `bent_cigar` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Chung-Reynolds | `chung_reynolds` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Cosine Mixture | `cosine_mixture` | [-1, 1]<sup>D</sup> | *f(x)* = -0.1*D*; *x* = 0<sub>D</sub> [N1] |
| Csendes | `csendes` | [-1, 1]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| De Jong F1 / Sphere | `de_jong_1` | [-5.12, 5.12]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Discus | `discus` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Dixon-Price | `dixon_price` | [-10, 10]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = 2<sup>-((2<sup>i</sup> - 2) / 2<sup>i</sup>)</sup>, *i* = 1, ..., *D* |
| Elliptic | `elliptic` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Expanded Griewank plus Rosenbrock | `expanded_griewank_plus_rosenbrock` | [-5, 5]<sup>D</sup> | *f(x)* = 0; *x* = 1<sub>D</sub> |
| Griewank | `griewangk_8` | [-600, 600]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Happy Cat | `happy_cat` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = -1<sub>D</sub> |
| HGBat | `hgbat` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = -1<sub>D</sub> |
| Katsuura | `katsuura` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> [N2] |
| Levy | `levy` | [-10, 10]<sup>D</sup> | *f(x)* = 0; *x* = 1<sub>D</sub> |
| Michalewicz | `michalewicz` | [0, π]<sup>D</sup> | Dimension- and *m*-dependent [N3] |
| Modified Schwefel | `modified_schwefel` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> [N4] |
| Perm 0,d,beta | `perm` | [-D, D]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = 1 / *i*, *i* = 1, ..., *D* |
| Pinter | `pinter` | [-10, 10]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Powell | `powell` | [-4, 5]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> [N5] |
| Qing | `qing` | [-500, 500]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = ±√*i*, *i* = 1, ..., *D* |
| Quintic | `quintic` | [-10, 10]<sup>D</sup> | *f(x)* = 0; each *x<sub>i</sub>* ∈ {-1, 2} |
| Rastrigin | `rastrigin` | [-5.12, 5.12]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Ridge | `ridge` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> [N6] |
| Rosenbrock Valley | `rosenbrocks_valley` | [-5, 10]<sup>D</sup> | *f(x)* = 0; *x* = 1<sub>D</sub> |
| Salomon | `salomon` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Schumer-Steiglitz | `schumer_steiglitz` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Schwefel | `schwefel` | [-500, 500]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* ≈ 420.968746228, *i* = 1, ..., *D* |
| Schwefel 2.21 | `schwefel_221` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Schwefel 2.22 | `schwefel_222` | [-100, 100]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Sphere 2 / Sum of Different Powers | `sphere_2` | [-1, 1]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Sphere 3 / Rotated Hyper-Ellipsoid | `sphere_3` | [-65.536, 65.536]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Step | `step` | [-100, 100]<sup>D</sup> | *f(x)* = 0; abs(*x<sub>i</sub>*) < 1 [N7] |
| Step 2 | `step_2` | [-100, 100]<sup>D</sup> | *f(x)* = 0; -0.5 ≤ *x<sub>i</sub>* < 0.5 [N7] |
| Step 3 | `step_3` | [-100, 100]<sup>D</sup> | *f(x)* = 0; abs(*x<sub>i</sub>*) < 1 [N7] |
| Stepint | `stepint` | [-5.12, 5.12]<sup>D</sup> | *f*<sup>*</sup> = 25 - 6*D*; *x<sub>i</sub>* ∈ [-5.12, -5) [N8] |
| Styblinski-Tang | `styblinski_tang` | [-5, 5]<sup>D</sup> | *f*<sup>*</sup> ≈ -39.166165704*D*; *x<sub>i</sub>* ≈ -2.903534028 |
| Trid | `trid` | [-D<sup>2</sup>, D<sup>2</sup>]<sup>D</sup> | *f*<sup>*</sup> = -*D*(*D* + 4)(*D* - 1) / 6; *x<sub>i</sub>* = *i*(*D* + 1 - *i*) |
| Weierstrass | `weierstrass` | [-0.5, 0.5]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Whitley | `whitley` | [-10.24, 10.24]<sup>D</sup> | *f(x)* = 0; *x* = 1<sub>D</sub> |
| Zakharov | `zakharov` | [-5, 10]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |

### CEC 2022 Functions

| Function | ID | Domain | Global Minimum |
|---|---|---:|---:|
| CEC 2022 F1 | `cec_2022_f01` | 2, 10, 20 | *f*<sup>*</sup> = 300 |
| CEC 2022 F2 | `cec_2022_f02` | 2, 10, 20 | *f*<sup>*</sup> = 400 |
| CEC 2022 F3 | `cec_2022_f03` | 2, 10, 20 | *f*<sup>*</sup> = 600 |
| CEC 2022 F4 | `cec_2022_f04` | 2, 10, 20 | *f*<sup>*</sup> = 800 |
| CEC 2022 F5 | `cec_2022_f05` | 2, 10, 20 | *f*<sup>*</sup> = 900 |
| CEC 2022 F6 | `cec_2022_f06` | 10, 20 | *f*<sup>*</sup> = 1800 |
| CEC 2022 F7 | `cec_2022_f07` | 10, 20 | *f*<sup>*</sup> = 2000 |
| CEC 2022 F8 | `cec_2022_f08` | 10, 20 | *f*<sup>*</sup> = 2200 |
| CEC 2022 F9 | `cec_2022_f09` | 2, 10, 20 | *f*<sup>*</sup> = 2300 |
| CEC 2022 F10 | `cec_2022_f10` | 2, 10, 20 | *f*<sup>*</sup> = 2400 |
| CEC 2022 F11 | `cec_2022_f11` | 2, 10, 20 | *f*<sup>*</sup> = 2600 |
| CEC 2022 F12 | `cec_2022_f12` | 2, 10, 20 | *f*<sup>*</sup> = 2700 |

### Engineering Design Benchmarks

Engineering benchmarks expose an objective function, along with bounds and constraints. Use `get_engineering_benchmark("<id>")` to retrieve `objective`, `constraints`, `min_values`, `max_values`, and best-known metadata. Constraint functions follow the package convention *g(x)* ≤ 0.

| Function | ID | Domain | Global Minimum | Constraints |
|---|---|---|---|---|
| Tension/compression spring design | `tension_spring` | *d* ∈ [0.05, 2], *D* ∈ [0.25, 1.30], *N* ∈ [2, 15] | *f*<sup>*</sup> ≈ 0.012665; *(d, D, N)* ≈ (0.05169, 0.35675, 11.2871) [N9] | 4 inequalities |
| Welded beam design | `welded_beam` | *h* ∈ [0.1, 2], *l* ∈ [0.1, 10], *t* ∈ [0.1, 10], *b* ∈ [0.1, 2] | *f*<sup>*</sup> ≈ 1.724852; *(h, l, t, b)* ≈ (0.20573, 3.47049, 9.03662, 0.20573) | 7 inequalities |
| Pressure vessel design, continuous relaxation | `pressure_vessel` | *T<sub>s</sub>*, *T<sub>h</sub>* ∈ [0, 99], *R* ∈ [10, 200], *L* ∈ [10, 240] | *f*<sup>*</sup> ≈ 5804.376217; *(T<sub>s</sub>, T<sub>h</sub>, R, L)* ≈ (0.727591, 0.359649, 37.699012, 240) [N10] | 4 inequalities |
| Pressure vessel design, discrete thickness | `pressure_vessel_discrete` | *T<sub>s</sub>*, *T<sub>h</sub>* rounded upward to multiples of 1/16; *R* ∈ [10, 200], *L* ∈ [10, 240] | *f*<sup>*</sup> ≈ 6059.714335; *(T<sub>s</sub>, T<sub>h</sub>, R, L)* ≈ (0.8125, 0.4375, 42.098446, 176.636596) [N10] | 4 inequalities |
| Speed reducer design | `speed_reducer` | 7 bounded design variables | *f*<sup>*</sup> ≈ 2994.471066; *x* ≈ (3.5, 0.7, 17, 7.3, 7.71532, 3.35021, 5.28665) | 11 inequalities |
| Three-bar truss design | `three_bar_truss` | *A<sub>1</sub>*, *A<sub>2</sub>* ∈ [0, 1] | *f*<sup>*</sup> ≈ 263.895843; *(A<sub>1</sub>, A<sub>2</sub>)* ≈ (0.788675, 0.408248) | 3 inequalities |
| Cantilever beam design | `cantilever_beam` | *x<sub>i</sub>* ∈ [0.01, 100], *i* = 1, ..., 5 | *f*<sup>*</sup> ≈ 1.339956; *x* ≈ (6.016016, 5.309173, 4.494330, 3.501475, 2.152665) | 1 inequality |
| Gear train design | `gear_train` | integer *x<sub>i</sub>* ∈ [12, 60], *i* = 1, ..., 4 | *f*<sup>*</sup> ≈ 2.700857 × 10<sup>-12</sup>; *x* = (16, 19, 43, 49) [N11] | box + integrality |

### Notes

| Note | Meaning |
|---|---|
| N1 | Alpine 2 and Cosine Mixture have sign-convention traps in the literature. This package uses minimization-compatible signs. |
| N2 | Katsuura is implemented as the product expression minus 1, so the exposed minimum is 0 at the origin. |
| N3 | Michalewicz has no single dimension-free closed-form optimum. For *m* = 10, common reference values are approximately: *D* = 2, *f*<sup>*</sup> = -1.8013; *D* = 5, *f*<sup>*</sup> = -4.6877; *D* = 10, *f*<sup>*</sup> = -9.6602. |
| N4 | Modified Schwefel is exposed in shifted CEC-style coordinates, so the visible optimizer is 0<sub>D</sub>. |
| N5 | Powell requires *D* to be a multiple of 4. |
| N6 | This is the cumulative ridge implementation, not the BBOB sharp-ridge function. |
| N7 | Step functions have optimizer intervals, not isolated optimizer points. |
| N8 | Stepint is bound-dependent. With bounds [-5.12, 5.12]<sup>D</sup>, *f*<sup>*</sup> = 25 - 6*D*; without bounds, it is unbounded below. |
| N9 | Engineering-design rows are constrained benchmarks. The Python module exposes `get_engineering_benchmark(id)` so users can pass the returned objective, bounds, and constraints directly to `pymetaheuristic.optimize`. |
| N10 | Pressure vessel has two common variants. `pressure_vessel` is the continuous relaxation; `pressure_vessel_discrete` rounds shell/head thickness upward to multiples of 1/16 before objective and constraint evaluation. |
| N11 | Gear train is a discrete integer benchmark. The implementation rounds variables to the nearest integer tooth counts by default. |

## 5. **Other Libraries**

[Back to Summary](#b-summary)

* For Multiobjective Optimization or Many Objectives Optimization, try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)
* For Traveling Salesman Problems (TSP), try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

### **Acknowledgement**

This section is dedicated to everyone who helped improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)


