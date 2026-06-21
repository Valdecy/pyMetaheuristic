
<p align="center">
  <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/logo_pmh_.png" alt="Logo" width="300" height="300"/>
</p>



# pymetaheuristic

A Python library for metaheuristic optimization and collaborative search, bringing together **394 optimization algorithms** across swarm, evolutionary, trajectory, physics-inspired, nature-inspired, human-inspired, and mathematical families. **pymetaheuristic** makes metaheuristics observable, comparable, cooperative, and benchmarkable through single optimizers, island systems, adaptive orchestration, diagnostics, and scientific benchmark studies.

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
pymetaheuristic.web_stop()
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
   - [2.8 Island System Unified Interface](#28-island-system-unified-interface) --- [[Colab Demo] ](https://colab.research.google.com/drive/15tidtz3PuVvBVUpO11RDpnRS6sjoVSD1?usp=sharing) --- 
   - [2.9 Adaptive Orchestration Policies](#29-adaptive-orchestration-policies)
   - [2.10 Chaotic Maps, Initialisation Presets, and Transfer Functions](#210-chaotic-maps-initialisation-presets-and-transfer-functions) --- [[Colab Demo]](https://colab.research.google.com/drive/1cvrahJ5Bp4E4vU7I-O6Uqru9SK2hxMXX?usp=sharing) ---
   - [2.11 Hyperparameter Tuner](#211-hyperparameter-tuner) --- [[Colab Demo] ](https://colab.research.google.com/drive/13pZQyrMDyegRAcYUJRO6cSwvQ7pZvDKs?usp=sharing) ---
   - [2.12 Save, Load, and Checkpoint](#212-save-load-and-checkpoint) --- [[Colab Demo] ](https://colab.research.google.com/drive/1detpXqDFMO-rNUpCSiN0RnuljUt5xD-E?usp=sharing) ---
   - [2.13 Benchmark Runner](#213-benchmark-runner) --- [[Colab Demo] ](https://colab.research.google.com/drive/1ZMw5RLFIU-EBPJoNp3kNyXg1KCU1KlFA?usp=sharing) ---
   - [2.14 Benchmark Study](#214-benchmark-study) --- [[Colab Demo] ](https://colab.research.google.com/drive/1yEDSdtUaiAhzpZX9KgVUsjhz0Q08w-B8?usp=sharing) ---
   - [2.15 EvoMapX Explainability](#215-evomapx-explainability) --- [[Colab Demo] ](https://colab.research.google.com/drive/1qhv8gPJci0VfIcr4N2etgc19A29wOPLl?usp=sharing) ---
3. [Algorithm Details](#3-algorithm-details)
4. [Test Functions](#4-test-functions) --- [[Colab Demo]](https://colab.research.google.com/drive/132-yqoaJKkJ4gf6yqjrV1siXVvZ3ZgE7?usp=sharing) ---
5. [Other Libraries](#5-other-libraries)

---
## 1. **Introduction** 

[Back to Summary](#b-summary)

**pymetaheuristic** is a Python optimization library built around metaheuristics, benchmark functions, stepwise execution, telemetry, cooperation, rule-based orchestration, constraint-aware evaluation, composable termination criteria, typed variable spaces, chaotic initialization, transfer functions, hyperparameter tuning, and benchmark sweeps. The package provides:

- a broad collection of metaheuristic algorithms
- benchmark functions for testing and visualization
- a stepwise engine API for controlled execution
- telemetry, export helpers, evaluation-indexed convergence data, and save/load for experiments
- EvoMapX explainability with Levels 1--4, explicit internal probe labels, OAM/PEG/CDS diagnostics, population snapshots, lineage metadata, and non-intrusive operator 
- cooperative multi-island optimization through `cooperative_optimize`
- clean object-based island systems through `IslandSystem`, `Island`, `TopologyConfig`, and `MigrationConfig`
- adaptive orchestration through fixed, rule-based, bandit, and portfolio-adaptive controllers
- island diagnostics, including migration matrices, contribution tables, island roles, action effectiveness, and topology summaries
- built-in constrained optimization support plus named repair strategies (`clip`, `wang`, `reflect`, `rand`, `limit_inverse`)
- composable `Termination` object with four independent stopping conditions
- automatic per-step diversity and exploration/exploitation tracking in history
- plotly-based diversity, convergence, runtime, and explore/exploit charts, including evaluation-indexed convergence plots
- typed variable space (`FloatVar`, `IntegerVar`, `CategoricalVar`, `PermutationVar`, `BinaryVar`)
- ten chaotic maps plus `lhs`, `obl`, and `sobol` population initialization presets
- eight transfer functions and `BinaryAdapter` for binary/discrete optimization
- `HyperparameterTuner` for grid/random hyperparameter search
- `BenchmarkRunner` for lightweight multi-algorithm × multi-problem sweeps
- `BenchmarkStudy` for scientific benchmarking of algorithms, island systems, and orchestration controllers, with rank tables, statistical tests, convergence plots, ECDFs, performance profiles, and result persistence
- `save_result`, `load_result`, `save_checkpoint`, `load_checkpoint` for persistence
- callback system with lifecycle hooks and callback-driven early stopping
- object-based `Problem` API with parametrized bounds, `latex_code()`, and curated test-problem wrappers
- human-readable `algorithm.info()` metadata

---
## 2. **Installation and Package Overview**

---
### 2.1 **Installation**

Standard installation:

```bash
pip install pymetaheuristic
```
---
### 2.2 **Package Overview**

[Back to Summary](#b-summary)

| Area | Main objects / functions | What it covers |
|---|---|---|
| Core Optimization | `optimize`, `list_algorithms`, `get_algorithm_info`, `create_optimizer` | Single-algorithm optimization, algorithm discovery, and inspection of default parameters |
| Termination | `Termination`, `EarlyStopping`, callbacks | Composable stopping criteria: max_steps, max_evaluations, max_time, max_early_stop, target_fitness, and callback-driven stops |
| Constraints and Feasibility | `optimize(..., constraints=..., constraint_handler=...)` | Constrained optimization with inequality/equality constraints, feasibility-aware evaluation |
| Benchmarks and Plots (Plotly) | `FUNCTIONS`, `get_test_function`, `plot_function`, `plot_convergence`, `compare_convergence`, `plot_benchmark_summary`, `plot_island_dynamics`, `plot_collaboration_network`, `plot_population_snapshot` | Built-in benchmark functions and plotly-based landscape, convergence, and cooperation visualizations |
| History Charts (Plotly) | `plot_global_best_chart`, `plot_diversity_chart`, `plot_explore_exploit_chart`, `plot_runtime_chart`, `plot_run_dashboard`, `plot_diversity_comparison` | Per-step diversity, exploration/exploitation, runtime, and convergence charts using plotly |
| Telemetry and Export | `summarize_result`, `export_history_csv`, `export_population_snapshots_json`, `convergence_data` | Experiment summarization, evaluation-indexed convergence extraction, and export of history and snapshots |
| EvoMapX Explainability | `get_evomapx_profile`, `get_evomapx_operators`, `result.evomapx_analysis`, `result.explain_evomapx`, `result.plot_evomapx_attribution`, `result.plot_evomapx_cds`, `result.plot_evomapx_peg`, `result.export_evomapx_json`, `result.export_evomapx_csv` | Passive explainability layer with explicit internal probe labels, OAM/IAM attribution matrices, CDS rankings, PEG lineage graphs, population snapshots, and Levels 1--4 diagnostics |
| IO (Persistence) | `save_result`, `load_result`, `save_checkpoint`, `load_checkpoint`, `result_to_json`, `result_from_json` | Save and restore results; checkpoint-and-resume for long runs |
| Typed Variable Space | `FloatVar`, `IntegerVar`, `BinaryVar`, `CategoricalVar`, `PermutationVar`, `build_problem_spec`, `decode_position`, `encode_position` | Define mixed-type search spaces; automatic encode/decode to/from continuous representation |
| Problem Objects | `Problem`, `FunctionalProblem`, `SphereProblem`, `RastriginProblem`, `AckleyProblem`, `RosenbrockProblem`, `ZakharovProblem`, `get_test_problem` | N-dimensional object-based problem definitions with parametrized bounds and `latex_code()` |
| Chaotic Maps | `ChaoticMap`, `chaotic_sequence`, `chaotic_population`, `AVAILABLE_CHAOTIC_MAPS` | Ten chaotic maps for diversity-preserving population initialisation and perturbation |
| Initialisation Presets | `uniform_population`, `lhs_population`, `obl_population`, `sobol_population`, `get_init_function`, `AVAILABLE_INIT_STRATEGIES` | Composable initialisation strategies for any algorithm through `init_function=` or `init_name=` |
| Transfer Functions | `apply_transfer`, `binarize`, `BinaryAdapter`, `vstf_01`–`vstf_04`, `sstf_01`–`sstf_04`, `AVAILABLE_TRANSFER_FUNCTIONS` | Eight transfer functions mapping continuous positions to binary probabilities for binary optimization |
| Repair and Random Utilities | `limit`, `limit_inverse`, `wang`, `rand`, `reflect`, `get_repair_function`, `levy_flight` | Named bound-repair policies and a reusable Lévy-flight sampler |
| Hyperparameter Tuner | `HyperparameterTuner` | Grid or random search over algorithm hyperparameters across multiple trials |
| Benchmark Runner | `BenchmarkRunner` | Lightweight multi-algorithm × multi-problem sweeps with summary aggregation and plotly-based benchmark charts |
| Benchmark Study | `BenchmarkStudy`, `BenchmarkResult`, `BenchmarkProblem`, `ProblemSuite`, `ExperimentRecord`, `load_benchmark` | Scientific benchmarking of algorithms, island systems, and orchestration controllers with long-format records, ranks, statistical tests, convergence plots, ECDFs, performance profiles, rank heatmaps, and JSON persistence |
| Cooperation | `cooperative_optimize`, `replay_cooperative_result`, `IslandSystem`, `Island`, `TopologyConfig`, `MigrationConfig`, `ExecutionConfig` | Direct and object-based multi-island cooperative optimization with configurable topology, migration interval, migration size, and migration policy |
| Orchestration | `orchestrated_optimize`, `OrchestrationSpec`, `CollaborativeConfig`, `RulesConfig`, `BanditConfig`, `PortfolioConfig`, `OrchestrationConfig` | Checkpoint-driven cooperation with fixed, rule-based, bandit, and portfolio-adaptive orchestration |
| Island Diagnostics | `migration_matrix`, `topology_summary`, `island_contribution`, `island_roles`, `action_effectiveness`, `diagnostics_summary` | Post-run interpretation of island systems, including communication patterns, donor/receiver behavior, island roles, and controller action effectiveness |
| Reference | `print_root_exports`, `print_reference`, `search_reference` | Programmatic argument reference for all callables |

To quickly inspect parameters:

```python
import pymetaheuristic

# List
pymetaheuristic.print_root_exports()

# Detail
pymetaheuristic.print_reference("optimize")
```
---
### 2.3 **Optimization, Telemetry, Export, and Plotting Example**

[Back to Summary](#b-summary)

`optimize` is the main high-level entry point for running a single metaheuristic on a user-defined objective function. The user specifies the algorithm, search bounds, and computational budget, while optional keyword arguments configure the selected optimizer and control diagnostics, such as history storage and population snapshots. The function returns a structured result object containing the best solution found, its objective value, and optional run traces that can later be summarized, exported, or plotted. In the example below, `optimize` applies Particle Swarm Optimization (PSO) to the Easom function over a bounded two-dimensional domain, stores the optimization trajectory, and then summarizes the run with `summarize_result`.

When `store_history` and `store_population_snapshots` are enabled, the returned result object contains enough information to support post-run analysis, reproducibility, and visualization. The history can be exported as a tabular CSV file, population states can be saved as JSON snapshots for later inspection, and convergence can be visualized directly with the built-in plotting utilities. In the example below, PSO is applied to the Easom function, the optimization trace is exported to disk, and the convergence behavior is plotted for immediate inspection.

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
```
---
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
---
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
# Example 1
constraint  = [lambda x: x[0] + x[1] - 1.0]                    # x0 + x1 <= 1

# Example 2
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

# Example 3
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
---
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
### 2.8 **Island System Unified Interface**

[Back to Summary](#b-summary)

`IslandSystem` is the object-based interface for defining collaborative optimization systems. It wraps the direct APIs `cooperative_optimize` and `orchestrated_optimize` into a cleaner architecture where islands, topology, migration, orchestration, and execution settings are declared as reusable configuration objects. This interface is recommended when the same island portfolio must be reused across cooperative, rule-based, bandit, portfolio-adaptive, or benchmarked runs.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/15tidtz3PuVvBVUpO11RDpnRS6sjoVSD1?usp=sharing)

```python
import numpy as np
import pymetaheuristic

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

system      = pymetaheuristic.IslandSystem(
    islands = [
        pymetaheuristic.Island(
            label     = "pso_explorer",
            algorithm = "pso",
            role      = "explorer",
            config    = {"swarm_size": 25},
        ),
        pymetaheuristic.Island(
            label     = "ga_diversity",
            algorithm = "ga",
            role      = "diversity_keeper",
            config    = {"population_size": 30},
        ),
        pymetaheuristic.Island(
            label     = "sa_refiner",
            algorithm = "sa",
            role      = "local_refiner",
            config    = {"temperature": 10.0},
        ),
        pymetaheuristic.Island(
            label     = "abco_explorer",
            algorithm = "abco",
            role      = "swarm_explorer",
            config    = {},
        ),
    ],
    topology     = pymetaheuristic.TopologyConfig(name = "ring",),
    migration    = pymetaheuristic.MigrationConfig(
        interval = 5,
        size     = 2,
        mode     = "elite",
        policy   = "push",
    ),
    orchestration = pymetaheuristic.OrchestrationConfig(
        checkpoint_interval        = 5,
        warmup_checkpoints         = 1,
        max_actions_per_checkpoint = 2,
    ),
    rules = pymetaheuristic.RulesConfig(
        stagnation_threshold     = 4,
        low_diversity_threshold  = 0.05,
        high_diversity_threshold = 0.25,
        perturbation_sigma       = 0.05,
    ),
    objective = "min",
    max_steps = 250,
    seed      = 42,
)

result = system.optimize(
    target_function = easom,
    min_values      = (-5, -5),
    max_values      = ( 5,  5),
    mode            = "cooperative",
)

print(result.best_fitness)
print(result.best_position)
print(len(result.events))

```
Island diagnostics transform cooperative and orchestrated runs into interpretable collaborative-search reports. After a run, the result object can summarize migration flows, topology structure, island contributions, island roles, and the effectiveness of orchestration actions. These diagnostics are useful for understanding whether cooperation helped, which island acted as the best refiner, which island donated useful candidates, and whether adaptive interventions were beneficial.

```python
import pandas as pd

# Migration matrix: how many candidates moved between islands.
migration_df = pd.DataFrame(result.migration_matrix(value = "migrants")).fillna(0)
print(migration_df)

# Contribution table: final fitness, improvement, donor/receiver behavior.
contribution_df = pd.DataFrame(result.island_contribution()).T
print(contribution_df)

# Interpretable island roles.
roles_df = pd.DataFrame(result.island_roles()).T
print(roles_df)

# Topology and communication summary.
topology = result.topology_summary()
print(topology)

# Action effectiveness for cooperative migration or orchestrated decisions.
actions = result.action_effectiveness()
print(actions)
```

Diagnostic plots are also available:

```python
result.plot_migration_network(value = "migrants", show = True, renderer = "colab")
result.plot_island_fitness(show = True, renderer = "colab")
```

---
### 2.9 **Adaptive Orchestration Policies**

[Back to Summary](#b-summary)

The orchestration layer supports multiple coordination policies. The `"cooperative"` mode uses fixed migration, `"rules"` applies checkpoint-based rules, `"bandit"` uses a multi-armed bandit controller to select actions based on previous rewards, and `"portfolio_adaptive"` changes behavior according to the optimization phase and island-state indicators.

```python
import numpy as np
import pymetaheuristic

def easom(x = [0, 0]):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


system = pymetaheuristic.IslandSystem(
    islands=[
        {"label": "pso", "algorithm": "pso", "config": {"swarm_size": 25}},
        {"label": "ga",  "algorithm": "ga",  "config": {"population_size": 30}},
        {"label": "sa",  "algorithm": "sa",  "config": {"temperature": 10.0}},
    ],
    max_steps = 250,
    seed      = 42,
)

modes   = ["cooperative", "rules", "bandit", "portfolio_adaptive"]
results = {}

for mode in modes:
    results[mode] = system.optimize(
        target_function = easom,
        min_values      = (-5, -5),
        max_values      = ( 5,  5),
        mode            = mode,
    )

for mode, res in results.items():
    print(mode,
         "best_fitness = ", res.best_fitness,
         "events       = ", len(getattr(res, "events", []) or []),
         "checkpoints  = ", len(getattr(res, "checkpoints", []) or []),
         "decisions    = ", len(getattr(res, "decisions", []) or []),
    )
```

Bandit orchestration can be configured explicitly:

```python
config = pymetaheuristic.CollaborativeConfig(
    orchestration = pymetaheuristic.OrchestrationSpec(
        mode                       = "bandit",
        checkpoint_interval        = 5,
        max_actions_per_checkpoint = 2,
        warmup_checkpoints         = 1,
    ),
    rules = pymetaheuristic.RulesConfig(
        stagnation_threshold       = 3,
        low_diversity_threshold    = 0.10,
        perturbation_sigma         = 0.05,
    ),
    bandit = pymetaheuristic.BanditConfig(
        policy                     = "ucb",
        exploration                = 0.5,
        action_cost_penalty        = 0.05,
    ),
)

result = pymetaheuristic.orchestrated_optimize(
    islands = [
        {"label": "pso", "algorithm": "pso", "config": {"swarm_size": 25}},
        {"label": "ga",  "algorithm": "ga",  "config": {"population_size": 30}},
        {"label": "sa",  "algorithm": "sa",  "config": {"temperature": 10.0}},
    ],
    target_function = easom,
    min_values      = (-5, -5),
    max_values      = ( 5,  5),
    max_steps       = 25,
    seed            = 42,
    config          = config,
)

print(result.best_fitness)
print(result.action_effectiveness())
```

---
### 2.10 **Chaotic Maps, Initialisation Presets, and Transfer Functions**

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

In addition to ordinary uniform random initialization, **pymetaheuristic** supports composable population initialization presets. These presets can be selected by name through `init_name=` or passed directly as a callable through `init_function=`. Available initialization strategies include:

| Strategy | Name / alias | Description |
|---|---|---|
| Uniform random | `"uniform"` | Standard independent uniform sampling inside the search bounds. |
| Latin Hypercube Sampling | `"lhs"`, `"latin_hypercube"`, `"lhs_population"` | Stratified sampling that spreads the initial population more evenly across each dimension. |
| Opposition-Based Learning | `"obl"` | Generates opposition-aware candidates to increase initial search coverage. |
| Sobol sequence | `"sobol"` | Low-discrepancy quasi-random sampling for space-filling initialization. |
| Chaotic initialization | `"chaotic:<map>"` | Uses one of the available chaotic maps, e.g. `"chaotic:tent"` or `"chaotic:logistic"`. |

```python
import pymetaheuristic
print(pymetaheuristic.AVAILABLE_INIT_STRATEGIES)

def sphere(x):
    return sum(v * v for v in x)

result = pymetaheuristic.optimize(
    algorithm       = "pso",
    target_function = sphere,
    min_values      = [-5.0] * 10,
    max_values      = [ 5.0] * 10,
    max_steps       = 100,
    seed            = 42,
    init_name       = "lhs",
)

print(result.best_fitness)
print(result.best_position)

```


---
### 2.11 **Hyperparameter Tuner**

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
### 2.12 **Save, Load, and Checkpoint**

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
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)

# Optimize - Run
result = pymetaheuristic.optimize(
                                  algorithm                  = 'pso',
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
### 2.13 **Benchmark Runner**

[Back to Summary](#b-summary)

`BenchmarkRunner` is the lightweight benchmark interface for multi-algorithm × multi-problem comparative sweeps. It executes every algorithm on every problem for a configurable number of independent trials, records the best fitness and wall-clock time for each run, and captures failed trials without interrupting the sweep. The raw results are returned as a tidy DataFrame that can be aggregated into summary statistics, rank tables, and publication-quality compact tables. For a more complete scientific benchmarking workflow involving algorithms, island systems, orchestration controllers, statistical tests, convergence plots, ECDFs, performance profiles, and persistence, use `BenchmarkStudy`.

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
### 2.14 **Benchmark Study**

[Back to Summary](#b-summary)

`BenchmarkStudy` is the scientific benchmarking interface. Unlike `BenchmarkRunner`, which focuses on lightweight algorithm sweeps, `BenchmarkStudy` can compare ordinary algorithms, island systems, and orchestration controllers under the same experimental protocol. It stores long-format experiment records, supports repeated trials, computes rank tables and statistical tests, and provides benchmark plots such as convergence curves, ECDFs, performance profiles, and rank heatmaps.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1yEDSdtUaiAhzpZX9KgVUsjhz0Q08w-B8?usp=sharing)

```python
import pymetaheuristic

# Benchmark problems.
problems = pymetaheuristic.ProblemSuite.from_names(["sphere", "rastrigin", "ackley", "rosenbrock"], dimensions = 2)

system      = pymetaheuristic.IslandSystem(
    islands = [
        pymetaheuristic.Island(
            label     = "pso_explorer",
            algorithm = "pso",
            role      = "explorer",
            config    = {"swarm_size": 25},
        ),
        pymetaheuristic.Island(
            label     = "ga_diversity",
            algorithm = "ga",
            role      = "diversity_keeper",
            config    = {"population_size": 30},
        ),
        pymetaheuristic.Island(
            label     = "sa_refiner",
            algorithm = "sa",
            role      = "local_refiner",
            config    = {"temperature": 10.0},
        ),
        pymetaheuristic.Island(
            label     = "abco_explorer",
            algorithm = "abco",
            role      = "swarm_explorer",
            config    = {},
        ),
    ],
    topology     = pymetaheuristic.TopologyConfig(name = "ring",),
    migration    = pymetaheuristic.MigrationConfig(
        interval = 5,
        size     = 2,
        mode     = "elite",
        policy   = "push",
    ),
    orchestration = pymetaheuristic.OrchestrationConfig(
        checkpoint_interval        = 5,
        warmup_checkpoints         = 1,
        max_actions_per_checkpoint = 2,
    ),
    rules = pymetaheuristic.RulesConfig(
        stagnation_threshold     = 4,
        low_diversity_threshold  = 0.05,
        high_diversity_threshold = 0.25,
        perturbation_sigma       = 0.05,
    ),
    objective = "min",
    max_steps = 250,
    seed      = 42,
)

benchmark_system_rules     = {"type": "island_system", "name": "islands_rules",              "system": system, "mode": "rules",}
benchmark_system_bandit    = {"type": "island_system", "name": "islands_bandit",             "system": system, "mode": "bandit",}
benchmark_system_portfolio = {"type": "island_system", "name": "islands_portfolio_adaptive", "system": system, "mode": "portfolio_adaptive",}

study = pymetaheuristic.BenchmarkStudy(
    candidates = [
        {
            "name":      "pso",
            "type":      "algorithm",
            "algorithm": "pso",
            "config":    {"swarm_size": 30},
        },
        {
            "name":      "ga",
            "type":      "algorithm",
            "algorithm": "ga",
            "config":    {"population_size": 40},
        },
        {
            "name":      "de",
            "type":      "algorithm",
            "algorithm": "de",
            "config":    {"population_size": 40},
        },
		  benchmark_system_rules,
		  benchmark_system_bandit,
		  benchmark_system_portfolio,
    ],
    problems        = problems,
    n_trials        = 5,
    max_evaluations = 5000,
    seed            = 42,
)

benchmark_result = study.run()

# Long-format experiment table.
df = benchmark_result.to_dataframe()
print(df.head())

# Summary and ranking.
print(benchmark_result.summary())
print(benchmark_result.rank_table())
print(benchmark_result.scientific_summary())

# Statistical tests.
print(benchmark_result.friedman_test())
print(benchmark_result.wilcoxon_pairwise())

# Save and reload.
benchmark_result.save("benchmark_demo.json")
loaded = pymetaheuristic.load_benchmark("benchmark_demo.json")
print(loaded.summary())
```

Benchmark plots:

```python
benchmark_result.plot_convergence(show = True, renderer = "colab")
benchmark_result.plot_ecdf(show = True, renderer = "colab")
benchmark_result.plot_performance_profile(show = True, renderer = "colab")
benchmark_result.plot_rank_heatmap(show = True, renderer = "colab")
```


Use `BenchmarkRunner` when you want a quick multi-algorithm × multi-problem sweep and a compact DataFrame summary. Use `BenchmarkStudy` when you need a scientific experimental protocol with repeated trials, fixed budgets, algorithm and island-system candidates, rank tables, statistical tests, convergence plots, ECDFs, performance profiles, rank heatmaps, and save/load support.

---
### 2.15 **EvoMapX Explainability**

[Back to Summary](#b-summary)

**pymetaheuristic** includes a package-wide **EvoMapX Explainability** layer for ordinary optimizers, cooperative island systems, and orchestrated island systems. It helps answer a question that convergence curves alone cannot answer: **which algorithm, island, migration event, or operator mechanism drove the improvement?** The current implementation uses a **probe architecture**. The probes observe optimizer execution but do not replace the original engine logic. They do not call the objective function independently, consume random numbers, reorder candidates, alter stopping criteria, change the evaluation budget, or modify the optimization trajectory. EvoMapX currently provides three complementary diagnostics:

- **Operator / Island Attribution Matrix (OAM/IAM):** A time-indexed contribution matrix. Rows are attribution units and columns are optimization steps. The attribution unit can be an explicit internal operator label, an algorithm, an island, a migration event, or an agent.
- **Convergence Driver Score (CDS):** An aggregate score derived from the attribution matrix. It ranks the units that contributed most to convergence.
- **Population Evolution Graph (PEG):** A graph representation of population continuity, parent-child relationships when available, inferred lineage, and migration links.

Per-operator attribution is computed from population lineage: the signed parent->child fitness change of each candidate is grouped by the operator that produced it, which requires no extra evaluations. The fidelity building blocks are:

| Support level | Meaning |
| --- | --- |
| Lineage Δf telemetry | Signed, operator-level Δf computed passively from parent->child fitness changes. |
| Operator counts | Per-step counts showing how many times each operator was applied. |
| Population lineage | Parent -> child metadata used to build PEG ancestry edges instead of nearest-neighbour fallback edges. |
| Profile metadata | Declared operator taxonomy used for documentation, web-app summaries, and support tables. |

The explainability layer supports four levels:

| EvoMapX level | Main purpose | What is recorded |
|---:|---|---|
| 1 | Population snapshots | Copied population states over time, avoiding reference aliasing to the final population |
| 2 | Operator attribution | Explicit probe labels, signed fitness deltas, operator counts, and OAM/CDS-ready contribution records |
| 3 | Lineage tracing | Level 2 plus candidate IDs, parent IDs, and PEG-ready ancestry metadata |
| 4 | Full activity diagnostics | Level 3 plus diversity change, displacement, changed-count, inferred acceptance rate, dominant operator, and candidate-evaluation summaries |

EvoMapX uses signed objective-consistent fitness changes. For minimization, a positive contribution means improvement; a negative contribution means deterioration. This preserves the true contribution pattern of each operator instead of clipping losses to zero.

* [Click Here for the Full Google Colab Example](https://colab.research.google.com/drive/1qhv8gPJci0VfIcr4N2etgc19A29wOPLl?usp=sharing)

Inspecting EvoMapX:

```python
import pymetaheuristic

profile = pymetaheuristic.get_evomapx_profile("wca")
labels  = pymetaheuristic.get_evomapx_operators("wca")
print(profile.to_dict())
print(labels)
```

Single Algorithm EvoMapX:

```python
import pymetaheuristic

def sphere(x):
    return sum(v * v for v in x)

result = pymetaheuristic.optimize(
    "woa",
    target_function = sphere,
    min_values      = [-5.0] * 10,
    max_values      = [ 5.0] * 10,
    objective       = "min",
    max_steps       = 40,
    seed            = 42,
    store_history   = True,
    evomapx         = True,
    evomapx_level   = 4,
)

print("Best fitness:",  result.best_fitness)
print("Best position:", result.best_position)
report = result.evomapx_analysis(level = "operator")
print(result.explain_evomapx(level = "operator"))

# Interactive Plot
result.plot_evomapx_attribution(level = "operator", filepath = "woa_oam.html")
result.plot_evomapx_cds(level = "operator", filepath = "woa_cds.html")
result.plot_evomapx_peg(filepath = "woa_peg.html")

# Exports
result.export_evomapx_json("woa_evomapx.json", level = "operator")
result.export_evomapx_csv("woa_oam.csv",       level = "operator")
```

Island EvoMapX:

```python
import pymetaheuristic

def sphere(x):
    return sum(v * v for v in x)

result = pymetaheuristic.cooperative_optimize(
    islands = [
        {"algorithm": "de",  "label": "DE explorer", "config": {"size": 25}},
        {"algorithm": "pso", "label": "PSO swarm",   "config": {"size": 25}},
        {"algorithm": "cem", "label": "CEM modeler", "config": {"size": 30, "k_samples": 6}},
    ],
    target_function            = sphere,
    min_values                 = [-5.0] * 10,
    max_values                 = [ 5.0] * 10,
    objective                  = "min",
    max_steps                  = 50,
    migration_interval         = 5,
    migration_size             = 3,
    topology                   = "ring",
    seed                       = 42,
    store_history              = True,
    store_population_snapshots = True,
    evomapx                    = True,
    evomapx_level              = 4,
)

# Which Island drove convergence?
print(result.explain_evomapx(level = "island"))

# Which explicit internal operators drove convergence?
print(result.explain_evomapx(level = "operator"))
result.plot_evomapx_attribution(level = "island", filepath = "island_oam.html")
result.plot_evomapx_cds(level = "island", filepath = "island_cds.html")
result.plot_evomapx_peg(filepath = "population_evolution_graph.html")
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
algo_info    = pymetaheuristic.get_algorithm_info(algorithm_id)

# Results
print("Algorithm ID:",   algo_info["algorithm_id"])
print("Algorithm Name:", algo_info["algorithm_name"])
print("")
print("Default Parameters:")
pprint(algo_info["defaults"])
```

Output:

```text
Algorithm ID:   pso
Algorithm Name: Particle Swarm Optimization

Default Parameters:
{'c1': 2.0, 'c2': 2.0, 'decay': 0, 'swarm_size': 30, 'w': 0.9}
```

The table below summarizes the optimization engines currently available in the library. Click the algorithm name to open its primary reference or original source, and all algorithms support checkpointing through the library framework, and all constraint handling is available through the framework-level constraint machinery.

- **Algorithm** reports the conventional algorithm name, 
- **ID** gives the identifier used in the codebase, 
- **Family** provides a coarse methodological grouping, 
- **Population** indicates whether the algorithm maintains an explicit candidate population and can also show population snapshots, 
- **Candidate Injection** indicates whether the algorithm is currently marked as able to absorb external candidates during cooperative or orchestrated workflows,
- **Restart** shows whether native restart support is declared,
- **EvoMapX** lists the semantic operator labels used by the passive EvoMapX resolver. Each label names an interpretable operator region of the algorithm (for example, `gwo.alpha_guidance`, `woa.spiral_bubble_net`, or `wca.evaporation_raining`), and the per-operator Convergence Driver Score is computed from the signed parent->child fitness change of the candidates that operator actually produced. The engines evaluate each operator output separately and therefore expose a multi-operator decomposition whose attribution responds to the seed and the objective. 

---

| Algorithm | ID | Family | Population | Candidate Injection | Restart | EvoMapX |
| --- | --- | --- | --- | --- | --- | --- |
| [Adam (Adaptive Moment Estimation)](https://doi.org/10.48550/arXiv.1412.6980) | `adam` | math | No | No | No | `adam.candidate_generation`<br>`adam.selection`<br>`adam.search_direction`<br>`adam.step_acceptance`<br>`adam.initialization` |
| [Adam Gradient Descent Optimizer](https://doi.org/10.1038/s41598-025-01678-9) | `agdo` | math | Yes | No | No | `agdo.progressive_gradient_momentum_dynamic_interaction`<br>`agdo.system_optimization_operator` |
<details>
<summary><b>🔍 View complete Metaheuristic reference table</b></summary>
<br/>

| Algorithm | ID | Family | Population | Candidate Injection | Restart | EvoMapX |
| --- | --- | --- | --- | --- | --- | --- |
| [Adam (Adaptive Moment Estimation)](https://doi.org/10.48550/arXiv.1412.6980) | `adam` | math | No | No | No | `adam.candidate_generation`<br>`adam.selection`<br>`adam.search_direction`<br>`adam.step_acceptance`<br>`adam.initialization` |
| [Adam Gradient Descent Optimizer](https://doi.org/10.1038/s41598-025-01678-9) | `agdo` | math | Yes | No | No | `agdo.progressive_gradient_momentum_dynamic_interaction`<br>`agdo.system_optimization_operator` |
| [Adaptive Aquila Optimizer](https://doi.org/10.1016/j.rineng.2024.103261) | `aao` | swarm | Yes | No | No | `aao.adaptive_aquila_guidance`<br>`aao.position_update`<br>`aao.elite_local_refinement`<br>`aao.selection` |
| [Adaptive Chaotic Grey Wolf Optimizer](https://doi.org/10.1007/s42835-023-01621-w) | `acgwo` | swarm | Yes | Yes | No | `acgwo.selection`<br>`acgwo.adaptive_weighted_pack_update`<br>`acgwo.alpha_guidance_trial`<br>`acgwo.beta_guidance_trial`<br>`acgwo.delta_guidance_trial` |
| [Adaptive Equilibrium Optimization](https://doi.org/10.1016/j.engappai.2020.103836) | `adaptive_eo` | physics | Yes | No | No | `adaptive_eo.selection`<br>`adaptive_eo.adaptive_local_refinement`<br>`adaptive_eo.equilibrium_pool_guided_update` |
| [Adaptive Exploration State-Space Particle Swarm Optimization](https://doi.org/10.1016/j.swevo.2025.101868) | `aesspso` | swarm | Yes | Yes | No | `aesspso.adaptive_velocity_position_update` |
| [Adaptive Inertia Weight Particle Swarm Optimization](https://doi.org/10.1007/11785231_48) | `aiw_pso` | swarm | Yes | No | No | `aiw_pso.position_update`<br>`aiw_pso.selection`<br>`aiw_pso.velocity_update`<br>`aiw_pso.elite_local_refinement` |
| [Adaptive Random Search](https://doi.org/10.1002/nav.20422) | `ars` | trajectory | Yes | Yes | No | `ars.small_step`<br>`ars.large_step` |
| [African Vultures Optimization Algorithm](https://doi.org/10.1016/j.cie.2021.107408) | `avoa` | swarm | Yes | Yes | No | `avoa.exploration_vulture_soaring`<br>`avoa.random_roost_exploration`<br>`avoa.convergent_competition_exploitation`<br>`avoa.levy_food_exploitation`<br>`avoa.aggressive_siege_exploitation`<br>`avoa.spiral_siege_exploitation` |
| [Ali Baba and the Forty Thieves](https://doi.org/10.1007/s00521-021-06392-x) | `aft` | human | Yes | Yes | No | `aft.best_guided_tracking`<br>`aft.random_treasure_search`<br>`aft.opposition_tracking` |
| [Anarchic Society Optimization](https://doi.org/10.1109/CEC.2011.5949940) | `aso` | swarm | Yes | Yes | No | `aso.anarchic_social_position_update` |
| [Animated Oat Optimization Algorithm](https://doi.org/10.1016/j.knosys.2025.113589) | `aoo` | swarm | Yes | No | No | `aoo.mean_wind_animation_update`<br>`aoo.best_wind_animation_update`<br>`aoo.self_wind_animation_update`<br>`aoo.rolling_levy_animation_update`<br>`aoo.projectile_jump_animation_update` |
| [Ant Colony Optimization (Continuous)](https://doi.org/10.1016/j.ejor.2006.06.046) | `acor` | swarm | Yes | Yes | No | `acor.archive_kernel_sampling_update` |
| [Ant Colony Optimization](https://doi.org/10.1109/3477.484436) | `aco` | swarm | Yes | No | No | `aco.pheromone_weighted_perturbation_in_each_dimension` |
| [Ant Lion Optimizer](https://doi.org/10.1016/j.advengsoft.2015.01.010) | `alo` | swarm | Yes | Yes | No | `alo.random_walk`<br>`alo.state_update`<br>`alo.candidate_generation`<br>`alo.selection`<br>`alo.combine` |
| [Aquila Optimizer](https://doi.org/10.1016/j.cie.2021.107250) | `ao` | swarm | Yes | Yes | No | `ao.high_soar_vertical_stoop`<br>`ao.contour_flight_exploration`<br>`ao.low_flight_attack`<br>`ao.walk_and_grab_prey` |
| [Archerfish Hunting Optimizer](https://doi.org/10.1016/j.engappai.2024.108081) | `aho` | swarm | Yes | Yes | No | `aho.single_shot_prey_projection`<br>`aho.double_shot_prey_projection`<br>`aho.levy_stagnation_rescue` |
| [Archimedes Optimization Algorithm](https://doi.org/10.1007/s10489-020-01893-z) | `arch_oa` | physics | Yes | Yes | No | `arch_oa.archimedes_density_volume_acceleration_update` |
| [Arithmetic Optimization Algorithm](https://doi.org/10.1016/j.cma.2020.113609) | `aoa` | swarm | Yes | Yes | No | `aoa.arithmetic_operator_position_update` |
| [Artemisinin Optimization](https://doi.org/10.1016/j.displa.2024.102740) | `artemisinin_o` | nature | Yes | Yes | No | `artemisinin_o.self_growth_update`<br>`artemisinin_o.best_growth_update`<br>`artemisinin_o.differential_mutation_update`<br>`artemisinin_o.self_reset_mutation`<br>`artemisinin_o.best_reset_mutation`<br>`artemisinin_o.boundary_best_repair` |
| [Artificial Algae Algorithm](https://doi.org/10.1016/j.asoc.2015.03.003) | `aaa` | swarm | Yes | No | Yes | `aaa.recombination`<br>`aaa.selection`<br>`aaa.adaptation_most_starving_colony_moves_toward`<br>`aaa.is_replaced_by_corresponding_cell_biggest` |
| [Artificial Bee Colony Optimization](https://doi.org/10.1007/s10898-007-9149-x) | `abco` | swarm | Yes | Yes | No | `abco.employed`<br>`abco.onlooker`<br>`abco.scout` |
| [Artificial Ecosystem Optimization](https://doi.org/10.1007/s00521-019-04452-x) | `aeo` | nature | Yes | Yes | No | `aeo.selection`<br>`aeo.consumer_decomposer_update`<br>`aeo.production_worst_agent` |
| [Artificial Electric Field Algorithm](https://doi.org/10.1016/j.swevo.2019.03.013) | `aefa` | physics | Yes | Yes | No | `aefa.electric_field_force_update` |
| [Artificial Fish Swarm Algorithm](https://doi.org/10.1007/s10462-012-9342-2) | `afsa` | swarm | Yes | Yes | No | `afsa.leap` |
| [Artificial Gorilla Troops Optimizer](https://doi.org/10.1002/int.22535) | `agto` | swarm | Yes | Yes | No | `agto.migration`<br>`agto.exploration`<br>`agto.state_update`<br>`agto.exploitation` |
| [Artificial Hummingbird Algorithm](https://doi.org/10.1016/j.cma.2021.114194) | `aha` | swarm | Yes | Yes | No | `aha.guided_foraging`<br>`aha.territorial_foraging`<br>`aha.migration` |
| [Artificial Lemming Algorithm](https://doi.org/10.1007/s10462-024-11023-7) | `ala` | swarm | Yes | Yes | No | `ala.high_energy_digging_walk`<br>`ala.high_energy_lemming_migration`<br>`ala.low_energy_spiral_foraging`<br>`ala.low_energy_levy_escape` |
| [Artificial Protozoa Optimizer](https://doi.org/10.1016/j.knosys.2024.111737) | `apo` | swarm | Yes | Yes | No | `apo.dormancy_random_restart`<br>`apo.dormancy_local_perturbation`<br>`apo.foraging_reproduction_update`<br>`apo.autotrophic_foraging_update` |
| [Artificial Rabbits Optimization](https://doi.org/10.1016/j.engappai.2022.105082) | `aro` | swarm | Yes | Yes | No | `aro.detour_foraging`<br>`aro.random_hiding` |
| [Atom Search Optimization](https://doi.org/10.1016/j.knosys.2018.08.030) | `aso_atom` | physics | Yes | Yes | No | `aso_atom.do_not_move_current_elites_unless` |
| [Automated Design of Variation Operators](https://doi.org/10.1145/3712256.3726456) | `autov` | evolutionary | Yes | Yes | No | `autov.learned_variation_operator_update` |
| [Bacterial Chemotaxis Optimizer](https://doi.org/10.1007/s13369-025-10749-y) | `bco` | nature | Yes | Yes | No | `bco.swim_refinement_update` |
| [Bacterial Colony Optimization](https://doi.org/10.1155/2012/698057) | `bacterial_colony_o` | nature | Yes | No | No | `bacterial_colony_o.migration`<br>`bacterial_colony_o.position_update`<br>`bacterial_colony_o.recombination`<br>`bacterial_colony_o.selection`<br>`bacterial_colony_o.current_colony_best_accept_only_it`<br>`bacterial_colony_o.implementation_but_only_as_bounded_macro` |
| [Bacterial Foraging Optimization](https://doi.org/10.1109/MCS.2002.1004010) | `bfo` | swarm | Yes | Yes | No | `bfo.chemotaxis_tumble_update`<br>`bfo.selection` |
| [Bald Eagle Search](https://doi.org/10.1007/s10462-019-09732-5) | `bes` | swarm | Yes | Yes | No | `bes.candidate_generation`<br>`bes.selection`<br>`bes.candidate_search` |
| [Barnacles Mating Optimizer](https://doi.org/10.1016/j.engappai.2019.103330) | `bmo` | swarm | Yes | Yes | No | `bmo.barnacle_recombination`<br>`bmo.random_barnacle_drift` |
| [Basin Hopping](https://doi.org/10.1021/jp970984n) | `basin_hopping` | trajectory | No | No | Yes | `basin_hopping.update` |
| [Basketball Team Optimization Algorithm](https://doi.org/10.1038/s41598-025-05477-0) | `btoa` | human | Yes | No | No | `btoa.position_update`<br>`btoa.selection`<br>`btoa.defensive_play_refinement`<br>`btoa.dynamic_position_candidate`<br>`btoa.offensive_play_update` |
| [Bat Algorithm](https://doi.org/10.1007/978-3-642-12538-6_6) | `bat_a` | swarm | Yes | Yes | No | `bat_a.candidate_generation`<br>`bat_a.selection`<br>`bat_a.force_or_velocity_update`<br>`bat_a.position_update`<br>`bat_a.acceptance`<br>`bat_a.state_update`<br>`bat_a.initialization` |
| [Battle Royale Optimization](https://doi.org/10.1007/s00521-020-05004-4) | `bro` | human | Yes | Yes | No | `bro.find_nearest_neighbour`<br>`bro.battle_damage_relocation_update`<br>`bro.selection` |
| [Bees Algorithm](https://doi.org/10.1016/B978-008045157-2/50081-X) | `bea` | swarm | Yes | Yes | No | `bea.elite_site_neighbourhood_search`<br>`bea.selected_site_neighbourhood_search`<br>`bea.scout_site_global_search` |
| [BFGS Quasi-Newton Method](https://doi.org/10.1090/S0025-5718-1970-0274029-X) | `bfgs` | math | No | No | No | `bfgs.update` |
| [Binary Space Partition Tree Genetic Algorithm](https://doi.org/10.1016/j.ins.2019.10.016) | `bspga` | evolutionary | Yes | Yes | No | `bspga.binary_partition_tree_variation_update` |
| [Biogeography-Based Optimization](https://doi.org/10.1109/TEVC.2008.919004) | `bbo` | evolutionary | Yes | Yes | No | `bbo.migration_mutation_selection_update` |
| [BIPOP-CMA-ES](https://doi.org/10.1145/1570256.1570333) | `bipop_cmaes` | evolutionary | Yes | Yes | Yes | `bipop_cmaes.cmaes_sampling`<br>`bipop_cmaes.elite_recombination`<br>`bipop_cmaes.distribution_update`<br>`bipop_cmaes.step_size_adaptation`<br>`bipop_cmaes.large_population_restart`<br>`bipop_cmaes.small_population_restart`<br>`bipop_cmaes.budget_regime_selection`<br>`bipop_cmaes.termination_check`<br>`bipop_cmaes.boundary_repair`<br>`bipop_cmaes.candidate_injection` |
| [Bird Swarm Algorithm](https://doi.org/10.1080/0952813X.2015.1042530) | `bsa` | swarm | Yes | Yes | No | `bsa.foraging_flight_update`<br>`bsa.vigilance_flight_update`<br>`bsa.producer_guided_flight_update`<br>`bsa.scrounger_random_flight_update` |
| [Birds-of-Paradise Search](https://doi.org/10.1007/s00521-026-11887-6) | `bps` | swarm | Yes | No | No | `bps.long_distance_flight`<br>`bps.local_tree_movement`<br>`bps.best_tree_attraction` |
| [Black Widow Optimization](https://doi.org/10.1016/j.engappai.2019.103249) | `bwo` | evolutionary | Yes | Yes | No | `bwo.crossover`<br>`bwo.mutation`<br>`bwo.procreation`<br>`bwo.candidate_generation`<br>`bwo.selection` |
| [Black-winged Kite Algorithm](https://doi.org/10.1007/s10462-024-10723-4) | `bka` | swarm | Yes | Yes | No | `bka.sine_soaring_update`<br>`bka.random_soaring_update`<br>`bka.peer_repulsion_cauchy_update`<br>`bka.leader_attraction_cauchy_update` |
| [Bonobo Optimizer](https://doi.org/10.1007/s10489-021-02444-w) | `bono` | swarm | Yes | Yes | No | `bono.social_guidance_phase`<br>`bono.exploratory_directional_move` |
| [Boxelder Bug Search Optimization](https://doi.org/10.1007/s00521-025-11637-0) | `bbso` | swarm | Yes | No | No | `bbso.coordinated_following_trial`<br>`bbso.self_following_trial` |
| [Brain Storm Optimization](https://doi.org/10.1007/978-3-642-21515-5_36) | `bso` | human | Yes | Yes | No | `bso.single_cluster_center_idea`<br>`bso.single_cluster_member_idea`<br>`bso.empty_cluster_center_idea`<br>`bso.two_cluster_center_blend`<br>`bso.two_cluster_member_blend` |
| [Brown-Bear Optimization Algorithm](https://doi.org/10.1201/9781003337003-6) | `bboa` | swarm | Yes | Yes | No | `bboa.selection`<br>`bboa.2_sniffing`<br>`bboa.pedal_marking_update` |
| [Butterfly Optimization Algorithm](https://doi.org/10.1007/s00500-018-3102-4) | `boa` | swarm | Yes | Yes | No | `boa.global_fragrance_attraction`<br>`boa.local_fragrance_random_walk` |
| [Camel Algorithm](https://doi.org/10.13140/RG.2.2.21814.56649) | `camel` | swarm | Yes | Yes | No | `camel.endurance_temperature_update`<br>`camel.selection` |
| [Capuchin Search Algorithm](https://doi.org/10.1007/s00521-020-05145-6) | `capsa` | swarm | Yes | Yes | No | `capsa.jumping_global_motion`<br>`capsa.long_jump_global_motion`<br>`capsa.velocity_swing_update`<br>`capsa.best_swing_update`<br>`capsa.velocity_memory_update`<br>`capsa.random_tree_leap`<br>`capsa.group_following_update` |
| [Cat Swarm Optimization](https://doi.org/10.1007/978-3-540-36668-3_94) | `cat_so` | swarm | Yes | Yes | No | `cat_so.seeking_mode_expansive_copy_update`<br>`cat_so.seeking_mode_contracting_copy_update`<br>`cat_so.tracing_mode_velocity_update` |
| [Catch Fish Optimization Algorithm](https://doi.org/10.1007/s10586-024-04618-w) | `cfoa` | swarm | Yes | No | No | `cfoa.individual_foraging_update`<br>`cfoa.group_foraging_update`<br>`cfoa.late_gaussian_capture_update` |
| [Cauchy-Gaussian mutation and improved search strategy GWO](https://doi.org/10.1038/s41598-022-23713-9) | `cg_gwo` | swarm | Yes | No | No | `cg_gwo.selection`<br>`cg_gwo.elite_local_refinement`<br>`cg_gwo.leader_guided_population_update` |
| [Chameleon Swarm Algorithm](https://doi.org/10.1016/j.eswa.2021.114685) | `chameleon_sa` | swarm | Yes | Yes | No | `chameleon_sa.social_pbest_gbest_update`<br>`chameleon_sa.random_global_exploration` |
| [Chaos Game Optimization](https://doi.org/10.1007/s10462-020-09867-w) | `cgo` | math | Yes | Yes | No | `cgo.current_seed_attractor`<br>`cgo.best_seed_attractor`<br>`cgo.mean_group_seed_attractor`<br>`cgo.dimension_mutation_seed` |
| [Chaotic-based Grey Wolf Optimizer](https://doi.org/10.1016/j.jcde.2017.02.005) | `chaotic_gwo` | swarm | Yes | No | No | `chaotic_gwo.selection`<br>`chaotic_gwo.elite_local_refinement`<br>`chaotic_gwo.leader_guided_population_update` |
| [Cheetah Based Optimization](https://doi.org/10.1038/s41598-022-14338-z) | `cddo` | swarm | Yes | Yes | No | `cddo.cheetah_chase_position_update` |
| [Cheetah Optimizer](https://doi.org/10.1038/s41598-022-14338-z) | `cdo` | swarm | Yes | Yes | No | `cdo.alpha_cheetah_attack_component`<br>`cdo.beta_cheetah_attack_component`<br>`cdo.gamma_cheetah_attack_component` |
| [Chernobyl Disaster Optimizer](https://doi.org/10.1016/j.compstruc.2023.107488) | `cdo_chernobyl` | physics | Yes | Yes | No | `cdo_chernobyl.alpha_beta_gamma_radiation_update`<br>`cdo_chernobyl.cdo_chernobyl_position_update`<br>`cdo_chernobyl.selection` |
| [Chicken Swarm Optimization](https://doi.org/10.1007/978-3-319-11857-4_10) | `chicken_so` | swarm | Yes | No | No | `chicken_so.selection`<br>`chicken_so.chicken_so_semantic_update` |
| [Child Drawing Development Optimization Algorithm](https://doi.org/10.1016/j.knosys.2024.111558) | `cddo_child` | human | Yes | Yes | No | `cddo_child.child_drawing_development_update` |
| [Chimp Optimization Algorithm](https://doi.org/10.1016/j.eswa.2020.113338) | `choa` | swarm | Yes | Yes | No | `choa.chimp_hunting_position_update` |
| [Chinese Pangolin Optimizer](https://doi.org/10.1007/s11227-025-07004-4) | `cpo` | swarm | Yes | No | No | `cpo.aroma_luring_trial`<br>`cpo.predation_feeding_trial` |
| [Circle-Based Search Algorithm](https://doi.org/10.3390/math10101626) | `circle_sa` | math | Yes | Yes | No | `circle_sa.circle_position_update` |
| [Circulatory System Based Optimization](https://doi.org/10.1016/j.egyr.2025.04.007) | `csbo` | nature | Yes | Yes | No | `csbo.systolic`<br>`csbo.diastolic` |
| [Clonal Selection Algorithm](https://doi.org/10.1109/TEVC.2002.1011539) | `clonalg` | evolutionary | Yes | Yes | No | `clonalg.candidate_generation`<br>`clonalg.selection`<br>`clonalg.cloning`<br>`clonalg.hypermutation` |
| [Coati Optimization Algorithm](https://doi.org/10.1016/j.knosys.2022.110011) | `coati_oa` | swarm | Yes | Yes | No | `coati_oa.candidate_generation`<br>`coati_oa.selection`<br>`coati_oa.behavioral_move` |
| [Cockroach Swarm Optimization](https://doi.org/10.1109/ICCET.2010.5485993) | `cockroach_so` | swarm | Yes | Yes | No | `cockroach_so.dispersal`<br>`cockroach_so.replacement`<br>`cockroach_so.state_update` |
| [Compact Genetic Algorithm](https://doi.org/10.1109/4235.797971) | `compact_ga` | distribution | No | No | No | `compact_ga.model_update`<br>`compact_ga.sampling`<br>`compact_ga.selection`<br>`compact_ga.state_update`<br>`compact_ga.compact_genetic_algorithm_semantic_update` |
| [Competitive Swarm Optimizer](https://doi.org/10.1016/j.swevo.2024.101543) | `cso` | swarm | Yes | Yes | No | `cso.mean_all_positions` |
| [Coot Bird Optimization](https://doi.org/10.1016/j.eswa.2021.115352) | `coot` | swarm | Yes | Yes | No | `coot.chain_movement_update` |
| [Coral Reefs Optimization](https://doi.org/10.1155/2014/739768) | `cro` | evolutionary | Yes | Yes | No | `cro.broadcast_spawning_recombination`<br>`cro.brooding_clone_mutation`<br>`cro.depredation_random_reseeding` |
| [Coronavirus Herd Immunity Optimization](https://doi.org/10.1007/s00521-020-05296-6) | `chio` | human | Yes | Yes | No | `chio.infected_contact_update`<br>`chio.susceptible_contact_update`<br>`chio.immune_contact_update` |
| [Cosmic Evolution Optimization](https://doi.org/10.1007/s00521-025-11234-6) | `ceo_cosmic` | physics | Yes | Yes | No | `ceo_cosmic.exploration_attraction_alignment`<br>`ceo_cosmic.global_collision_update`<br>`ceo_cosmic.resonance_refinement_update` |
| [Covariance Matrix Adaptation Evolution Strategy](https://doi.org/10.1109/ICEC.1996.542381) | `cmaes` | evolutionary | Yes | Yes | No | `cmaes.covariance_sampling_recombination_update` |
| [Coyote Optimization Algorithm](https://doi.org/10.1109/CEC.2018.8477769) | `coa` | swarm | Yes | Yes | No | `coa.alpha_social_condition_update`<br>`coa.tendency_social_condition_update`<br>`coa.pup_birth_replacement`<br>`coa.migration_exchange` |
| [Crayfish Optimization Algorithm](https://doi.org/10.1007/s10462-023-10567-4) | `crayfish_oa` | swarm | Yes | Yes | No | `crayfish_oa.high_temperature_shelter_update`<br>`crayfish_oa.high_temperature_competition_update`<br>`crayfish_oa.food_competition_update`<br>`crayfish_oa.food_intake_update` |
| [Cross Entropy Method](https://doi.org/10.1007/978-1-4757-4321-0) | `cem` | distribution | Yes | Yes | No | `cem.model_sampling_elite_distribution_update` |
| [Crow Search Algorithm](https://doi.org/10.1016/j.compstruc.2016.03.001) | `csa` | swarm | Yes | Yes | No | `csa.memory_following_update`<br>`csa.awareness_random_relocation`<br>`csa.mixed_memory_random_update` |
| [Cuckoo Catfish Optimizer](https://doi.org/10.1007/s10462-025-11291-x) | `cco` | swarm | Yes | Yes | No | `cco.candidate_search`<br>`cco.selection`<br>`cco.candidate_generation` |
| [Cuckoo Search](https://doi.org/10.1109/NABIC.2009.5393690) | `cuckoo_s` | swarm | Yes | Yes | No | `cuckoo_s.levy_flight`<br>`cuckoo_s.replacement`<br>`cuckoo_s.candidate_generation`<br>`cuckoo_s.selection` |
| [Cultural Algorithm](https://doi.org/10.1080/00207160.2015.1067309) | `ca` | evolutionary | Yes | Yes | No | `ca.cultural_belief_guided_update` |
| [Dandelion Optimizer](https://doi.org/10.1016/j.engappai.2022.105075) | `do_dandelion` | swarm | Yes | Yes | No | `do_dandelion.rising_seed_phase`<br>`do_dandelion.descent_diffusion_phase`<br>`do_dandelion.elite_landing_phase`<br>`do_dandelion.candidate_generation`<br>`do_dandelion.selection` |
| [Deep Sleep Optimiser](https://doi.org/10.1109/ACCESS.2023.3298105) | `dso` | human | Yes | Yes | No | `dso.deep_sleep_decay_update`<br>`dso.slow_wave_recovery_update` |
| [Deer Hunting Optimization Algorithm](https://doi.org/10.1093/comjnl/bxy133) | `doa` | human | Yes | Yes | No | `doa.hunting`<br>`doa.search`<br>`doa.state_update`<br>`doa.exploitation_move`<br>`doa.replacement` |
| [Delta Plus](https://doi.org/10.1007/s10586-024-05094-y) | `dp` | math | Yes | No | No | `dp.delta_operation` |
| [Dhole Optimization Algorithm](https://doi.org/10.1007/s10586-024-05005-1) | `dhole_oa` | swarm | Yes | No | No | `dhole_oa.searching_stage`<br>`dhole_oa.encircling_stage`<br>`dhole_oa.large_prey_attack`<br>`dhole_oa.small_prey_kill` |
| [Differential Evolution JADE](https://doi.org/10.1109/TEVC.2009.2014613) | `jade` | evolutionary | Yes | No | No | `jade.candidate_generation`<br>`jade.selection`<br>`jade.mutation`<br>`jade.crossover`<br>`jade.initialization` |
| [Differential Evolution MTS](https://doi.org/10.1109/CEC.2009.4983179) | `hde` | evolutionary | Yes | Yes | No | `hde.candidate_search`<br>`hde.selection`<br>`hde.differential_evolution_update` |
| [Differential Evolution with Self-Adaptive Populations](https://doi.org/10.1007/s00500-005-0537-1) | `sap_de` | evolutionary | Yes | No | No | `sap_de.selection`<br>`sap_de.elite_local_refinement`<br>`sap_de.self_adaptive_parameter_de_update` |
| [Differential Evolution](https://doi.org/10.1023/A:1008202821328) | `de` | evolutionary | Yes | Yes | No | `de.differential_mutation_crossover_selection` |
| [Dispersive Fly Optimization](https://doi.org/10.15439/2014F142) | `dfo` | swarm | Yes | Yes | No | `dfo.dispersive_fly_neighbour_update`<br>`dfo.elite_disturbance_update`<br>`dfo.selection` |
| [Diversity enhanced Strategy based Grey Wolf Optimizer](https://doi.org/10.1016/j.knosys.2022.109100) | `ds_gwo` | swarm | Yes | No | No | `ds_gwo.selection`<br>`ds_gwo.elite_local_refinement`<br>`ds_gwo.leader_guided_population_update` |
| [Divine Religions Algorithm](https://doi.org/10.1007/s10586-024-04954-x) | `dra` | human | Yes | No | No | `dra.selection`<br>`dra.dialectic_interaction_update` |
| [Dolphin Echolocation Optimization](https://doi.org/10.1016/j.advengsoft.2016.05.002) | `deo_dolphin` | swarm | Yes | Yes | No | `deo_dolphin.elite_reference_echo_guidance`<br>`deo_dolphin.elite_jitter_echo_guidance`<br>`deo_dolphin.peer_reference_echo_guidance`<br>`deo_dolphin.peer_jitter_echo_guidance` |
| [Dragonfly Algorithm](https://doi.org/10.1007/s00521-015-1920-1) | `da` | swarm | Yes | Yes | No | `da.neighbour_alignment_update`<br>`da.levy_flight_exploration`<br>`da.food_enemy_swarm_update` |
| [Dream Optimization Algorithm](https://doi.org/10.1016/j.cma.2024.117718) | `dream_oa` | human | Yes | No | No | `dream_oa.dream_generation_refinement_update` |
| [Dung Beetle Optimizer](https://doi.org/10.1007/s11227-022-04959-6) | `dbo` | swarm | Yes | Yes | No | `dbo.foraging`<br>`dbo.selection`<br>`dbo.state_update`<br>`dbo.ball_rolling_dance_update` |
| [Dwarf Mongoose Optimization Algorithm](https://doi.org/10.1016/j.cma.2022.114570) | `dmoa` | swarm | Yes | Yes | No | `dmoa.selection`<br>`dmoa.3_baby_sitter_eviction`<br>`dmoa.scalar_broadcast`<br>`dmoa.scout_phase` |
| [Dynamic Differential Annealed Optimization](https://doi.org/10.1016/j.asoc.2020.106392) | `ddao` | physics | Yes | Yes | No | `ddao.exploration`<br>`ddao.selection`<br>`ddao.state_update`<br>`ddao.dynamic_annealed_refinement_update` |
| [Dynamic Virtual Bats Algorithm](https://doi.org/10.1109/INCoS.2014.40) | `dvba` | swarm | Yes | Yes | No | `dvba.force_or_velocity_update`<br>`dvba.position_update`<br>`dvba.random_walk`<br>`dvba.state_update`<br>`dvba.candidate_generation`<br>`dvba.selection` |
| [Earthworm Optimization Algorithm](https://doi.org/10.1504/IJBIC.2015.10004283) | `eoa` | swarm | Yes | Yes | No | `eoa.crossover`<br>`eoa.state_update`<br>`eoa.mutation`<br>`eoa.candidate_generation`<br>`eoa.selection`<br>`eoa.reproduction` |
| [Ecological Cycle Optimizer](https://doi.org/10.48550/arXiv.2508.20458) | `ecological_cycle_o` | swarm | Yes | Yes | No | `ecological_cycle_o.selection`<br>`ecological_cycle_o.ecological_cycle_transition_update`<br>`ecological_cycle_o.eval_accept_group` |
| [Educational Competition Optimizer](https://doi.org/10.3390/biomimetics10030176) | `eco` | human | Yes | Yes | No | `eco.primary_competition_update`<br>`eco.sine_cosine_learning_update`<br>`eco.best_weighted_learning_update`<br>`eco.levy_exam_update` |
| [Eel and Grouper Optimizer](https://doi.org/10.1007/s10586-024-04545-w) | `eel_grouper_o` | swarm | Yes | No | No | `eel_grouper_o.eel_weighted_hunting_update`<br>`eel_grouper_o.grouper_weighted_hunting_update` |
| [Efficient and Robust Grey Wolf Optimizer](https://doi.org/10.1007/s00500-019-03939-y) | `er_gwo` | swarm | Yes | No | No | `er_gwo.selection`<br>`er_gwo.elite_local_refinement`<br>`er_gwo.leader_guided_population_update` |
| [Efficient Global Optimization](https://doi.org/10.1023/A:1008306431147) | `ego` | distribution | Yes | Yes | No | `ego.expected_improvement_candidate_generation` |
| [Egret Swarm Optimization Algorithm](https://doi.org/10.3390/biomimetics7040144) | `esoa` | swarm | Yes | Yes | No | `esoa.behavioral_move`<br>`esoa.selection`<br>`esoa.egret_sit_and_wait_update` |
| [Electric Charged Particles Optimization](https://doi.org/10.1007/s10462-020-09890-x) | `ecpo` | physics | Yes | Yes | No | `ecpo.electric_charge_random_perturbation` |
| [Electric Eel Foraging Optimization](https://doi.org/10.1016/j.eswa.2023.122200) | `eefo` | swarm | Yes | No | No | `eefo.interaction_migration`<br>`eefo.resting_area_update`<br>`eefo.levy_hunting_update`<br>`eefo.prey_capture_update` |
| [Electrical Storm Optimization](https://doi.org/10.3390/make7010024) | `eso` | physics | Yes | Yes | No | `eso.electric_storm_field_update` |
| [Electromagnetic Field Optimization](https://doi.org/10.1016/j.swevo.2015.07.002) | `efo` | physics | Yes | Yes | No | `efo.electromagnetic_field_update`<br>`efo.random_field_reinitialization`<br>`efo.dimension_reset_mutation` |
| [Elephant Herding Optimization](https://doi.org/10.1109/ISCBI.2015.8) | `eho` | swarm | Yes | Yes | No | `eho.long_range_clan_best_guided_update`<br>`eho.short_range_clan_best_guided_update`<br>`eho.matriarch_center_update`<br>`eho.separating_random_relocation` |
| [Elk Herd Optimizer](https://doi.org/10.1007/s10462-023-10680-4) | `elk_ho` | swarm | Yes | Yes | No | `elk_ho.selection`<br>`elk_ho.family_mating_position_update` |
| [Emperor Penguin Colony](https://doi.org/10.1016/j.knosys.2018.06.001) | `epc` | swarm | Yes | Yes | No | `epc.spiral_attraction_update`<br>`epc.thermal_mutation_update` |
| [Energy Valley Optimizer](https://doi.org/10.1038/s41598-022-27344-y) | `evo` | physics | Yes | Yes | No | `evo.exploration`<br>`evo.state_update`<br>`evo.exploitation` |
| [Enhanced Artificial Ecosystem-Based Optimization](https://doi.org/10.1109/ACCESS.2020.3027654) | `enhanced_aeo` | nature | Yes | No | No | `enhanced_aeo.selection`<br>`enhanced_aeo.ecosystem_producer_consumer_update`<br>`enhanced_aeo.enhanced_decomposition_refinement` |
| [Enhanced Tug of War Optimization](https://doi.org/10.1016/j.procs.2020.03.063) | `enhanced_two` | physics | Yes | No | No | `enhanced_two.candidate_generation`<br>`enhanced_two.selection`<br>`enhanced_two.force_update`<br>`enhanced_two.state_update`<br>`enhanced_two.initialization` |
| [Enzyme Activity Optimizer](https://doi.org/10.1007/s11227-025-07052-w) | `eao` | nature | Yes | Yes | No | `eao.sinusoidal_best_substrate_update`<br>`eao.vector_scaled_differential_substrate_update`<br>`eao.scalar_scaled_differential_substrate_update` |
| [Equilibrium Optimizer](https://doi.org/10.1016/j.knosys.2019.105190) | `eo` | physics | Yes | Yes | No | `eo.equilibrium_position_update` |
| [Escape Algorithm](https://doi.org/10.1007/s10462-024-11008-6) | `esc` | human | Yes | Yes | No | `esc.escape_from_worst_update`<br>`esc.move_toward_best_update`<br>`esc.random_exploration_update` |
| [Evolution Strategy (Mu + Lambda)](https://doi.org/10.1023/A:1015059928466) | `es` | evolutionary | Yes | Yes | No | `es.parent_survivor`<br>`es.large_step_mutation_offspring`<br>`es.small_step_mutation_offspring` |
| [Evolutionary Programming](https://doi.org/10.1007/BF00175356) | `ep` | evolutionary | Yes | Yes | No | `ep.parent_survivor`<br>`ep.large_strategy_mutation_offspring`<br>`ep.small_strategy_mutation_offspring` |
| [Expanded Grey Wolf Optimizer](https://doi.org/10.1007/s00366-019-00837-7) | `ex_gwo` | swarm | Yes | No | No | `ex_gwo.selection`<br>`ex_gwo.elite_local_refinement`<br>`ex_gwo.leader_guided_population_update` |
| [Exponential Distribution Optimizer](https://doi.org/10.1007/s10462-023-10403-9) | `edo` | math | Yes | Yes | No | `edo.distribution_update`<br>`edo.candidate_generation`<br>`edo.state_update` |
| [Exponential-Trigonometric Optimization](https://doi.org/10.1016/j.cma.2024.117411) | `eto` | math | Yes | Yes | No | `eto.exponential_orbit_update`<br>`eto.trigonometric_orbit_update` |
| [Extra-Trees Bayesian Optimization](https://doi.org/10.1007/s10994-006-6226-1) | `et_bo` | surrogate | No | No | No | `et_bo.update` |
| [Fast Evolutionary Programming](https://doi.org/10.1109/4235.771163) | `fep` | evolutionary | Yes | Yes | No | `fep.fast_mutation_tournament_selection_update` |
| [Fata Geophysics Optimizer](https://doi.org/10.1016/j.neucom.2024.128289) | `fata` | physics | Yes | Yes | No | `fata.random_refraction_update`<br>`fata.best_refraction_update`<br>`fata.peer_refraction_update` |
| [Feasibility Rule with Objective Function Information](https://doi.org/10.1109/TCYB.2015.2493239) | `frofi` | evolutionary | Yes | Yes | No | `frofi.current_to_rand_de`<br>`frofi.rand_to_best_crossover_de`<br>`frofi.no_crossover_de`<br>`frofi.targeted_mutation` |
| [Fennec Fox Optimizer](https://doi.org/10.1109/ACCESS.2022.3197745) | `ffo` | swarm | Yes | Yes | No | `ffo.exploration`<br>`ffo.state_update`<br>`ffo.exploitation` |
| [Fick's Law Algorithm](https://doi.org/10.1016/j.knosys.2022.110146) | `fla` | physics | Yes | Yes | No | `fla.forward_diffusion_transfer`<br>`fla.source_fluid_diffusion`<br>`fla.receiver_fluid_diffusion`<br>`fla.reverse_diffusion_transfer`<br>`fla.equilibrium_exploitation_update` |
| [Firefly Algorithm](https://doi.org/10.1504/IJBIC.2010.032124) | `firefly_a` | swarm | Yes | Yes | No | `firefly_a.attraction_dominant_move`<br>`firefly_a.randomization_dominant_move` |
| [Fireworks Algorithm](https://doi.org/10.1016/j.asoc.2017.10.046) | `fwa` | swarm | Yes | Yes | No | `fwa.selection`<br>`fwa.state_update` |
| [Fish School Search](https://doi.org/10.1109/ICSMC.2008.4811695) | `fss` | swarm | Yes | Yes | No | `fss.collective_volitive_movement`<br>`fss.selection` |
| [Fitness Dependent Optimizer](https://doi.org/10.1109/ACCESS.2019.2907012) | `fdo` | swarm | Yes | Yes | No | `fdo.fitness_weighted_pace_update`<br>`fdo.best_guided_position_update`<br>`fdo.selection` |
| [Fletcher-Reeves Conjugate Gradient](https://doi.org/10.1002/er.8067) | `frcg` | math | No | No | No | `frcg.update` |
| [Flood Algorithm](https://doi.org/10.1007/s11227-024-06291-7) | `flood_a` | physics | Yes | Yes | No | `flood_a.flood_flow_direction_update`<br>`flood_a.flood_recession_refinement_update`<br>`flood_a.selection` |
| [Flow Direction Algorithm](https://doi.org/10.1016/j.cie.2021.107224) | `fda` | swarm | Yes | Yes | No | `fda.downhill_flow_direction_update`<br>`fda.neighbour_flow_direction_update`<br>`fda.elite_flow_direction_update` |
| [Flower Pollination Algorithm](https://doi.org/10.1007/978-3-642-32894-7_27) | `fpa` | swarm | Yes | Yes | No | `fpa.global_levy_pollination`<br>`fpa.local_pollination` |
| [Forensic-Based Investigation Optimization](https://doi.org/10.1016/j.asoc.2020.106339) | `fbio` | human | Yes | Yes | No | `fbio.candidate_generation`<br>`fbio.selection`<br>`fbio.exploration` |
| [Forest Optimization Algorithm](https://doi.org/10.1016/j.eswa.2014.05.009) | `foa` | swarm | Yes | Yes | No | `foa.local_seeding_growth_update`<br>`foa.selection` |
| [Fossa Optimization Algorithm](https://doi.org/10.1007/s10462-024-10953-0) | `foa_fossa` | swarm | Yes | Yes | No | `foa_fossa.prey_pursuit_update`<br>`foa_fossa.defensive_escape_update` |
| [Fox Optimizer](https://doi.org/10.1007/s10489-022-03533-0) | `fox` | swarm | Yes | Yes | No | `fox.prey_jump_exploitation`<br>`fox.current_to_random_walk_update`<br>`fox.best_radius_random_walk` |
| [Frilled Lizard Optimization](https://doi.org/10.32604/cmc.2024.053189) | `flo` | swarm | Yes | Yes | No | `flo.update` |
| [Fruit-Fly Algorithm](https://doi.org/10.1016/j.knosys.2011.07.001) | `ffa` | swarm | Yes | Yes | No | `ffa.fruitfly_smell_search_update` |
| [Fuzzy Hierarchical Operator - Grey Wolf Optimizer](https://doi.org/10.1016/j.asoc.2017.03.048) | `fuzzy_gwo` | swarm | Yes | No | No | `fuzzy_gwo.selection`<br>`fuzzy_gwo.elite_local_refinement`<br>`fuzzy_gwo.leader_guided_population_update` |
| [Gaining-Sharing Knowledge Algorithm](https://doi.org/10.1007/s13042-019-01053-x) | `gska` | human | Yes | Yes | No | `gska.gaining_sharing_knowledge_update` |
| [Gaussian Process Bayesian Optimization](https://doi.org/10.1023/A:1008306431147) | `gp_bo` | surrogate | No | No | No | `gp_bo.update` |
| [Gazelle Optimization Algorithm](https://doi.org/10.1007/s00521-022-07854-6) | `gazelle_oa` | swarm | Yes | Yes | No | `gazelle_oa.brownian_foraging_update`<br>`gazelle_oa.levy_elite_transition_update`<br>`gazelle_oa.levy_foraging_update`<br>`gazelle_oa.random_patch_avoidance_update`<br>`gazelle_oa.peer_difference_escape_update` |
| [Gekko Japonicus Algorithm](https://doi.org/10.1016/j.eswa.2025.127982) | `gja` | swarm | Yes | Yes | No | `gja.levy_wall_search`<br>`gja.gaussian_wall_search` |
| [Generalized Normal Distribution Optimizer](https://doi.org/10.1016/j.enconman.2020.113301) | `gndo` | math | Yes | Yes | No | `gndo.generalized_normal_local_update`<br>`gndo.difference_vector_global_update` |
| [Genetic Algorithm](https://doi.org/10.7551/mitpress/1090.001.0001) | `ga` | evolutionary | Yes | Yes | No | `ga.candidate_generation`<br>`ga.selection`<br>`ga.breed`<br>`ga.mutate` |
| [Genghis Khan Shark Optimizer](https://doi.org/10.1016/j.aei.2023.102210) | `gkso` | swarm | Yes | Yes | No | `gkso.genghis_khan_crossover_exploration`<br>`gkso.shark_hunting_pso_update` |
| [Geometric Mean Optimizer](https://doi.org/10.1007/s00500-023-08202-z) | `gmo` | math | Yes | Yes | No | `gmo.marketing_guidance_update` |
| [Germinal Center Optimization](https://doi.org/10.1016/j.ifacol.2018.07.300) | `gco` | human | Yes | Yes | No | `gco.dark_zone_mutation_update` |
| [Geyser Inspired Algorithm](https://doi.org/10.1007/s42235-023-00437-8) | `gea` | physics | Yes | Yes | No | `gea.neighbour_geyser_eruption_update`<br>`gea.pressure_random_eruption_update` |
| [Giant Pacific Octopus Optimizer](https://doi.org/10.1007/s12065-024-00945-4) | `gpoo` | swarm | Yes | No | No | `gpoo.octopus_tentacle_prey_position_update` |
| [Giant Trevally Optimizer](https://doi.org/10.1109/ACCESS.2022.3223388) | `gto` | swarm | Yes | Yes | No | `gto.candidate_search`<br>`gto.selection`<br>`gto.candidate_generation`<br>`gto.behavioral_move` |
| [Glider Snake Optimization](https://doi.org/10.1007/s10462-026-11504-x) | `gso_glider_snake` | swarm | Yes | No | No | `gso_glider_snake.glider_snake_position_update` |
| [Glowworm Swarm Optimization](https://doi.org/10.1007/978-3-319-51595-3) | `gso` | swarm | Yes | Yes | No | `gso.glowworm_luciferin_movement_update` |
| [Golden Jackal Optimizer](https://doi.org/10.1016/j.eswa.2022.116924) | `gjo` | swarm | Yes | Yes | No | `gjo.male_female_exploitation`<br>`gjo.male_female_exploration` |
| [Gradient-Based Optimizer](https://doi.org/10.1007/s11831-022-09872-y) | `gbo` | math | Yes | Yes | No | `gbo.gradient_search_rule_update`<br>`gbo.local_escaping_operator_update` |
| [Gradient-Based Particle Swarm Optimization](https://doi.org/10.48550/arXiv.2312.09703) | `gpso` | swarm | Yes | Yes | No | `gpso.velocity_position_update` |
| [Gradient-Boosted Regression Trees Bayesian Optimization](https://doi.org/10.1214/aos/1013203451) | `gbrt_bo` | surrogate | No | No | No | `gbrt_bo.update` |
| [Grasshopper Optimization Algorithm](https://doi.org/10.1016/j.advengsoft.2017.01.004) | `goa` | swarm | Yes | Yes | No | `goa.grasshopper_social_force_update` |
| [Gravitational Search Algorithm](https://doi.org/10.1016/j.ins.2009.03.004) | `gsa` | physics | Yes | Yes | No | `gsa.gravitational_force_acceleration_update` |
| [Greedy Randomized Adaptive Search Procedure](https://doi.org/10.1007/BF01096763) | `grasp` | trajectory | No | No | Yes | `grasp.update` |
| [Grey Wolf Optimizer](https://doi.org/10.1016/j.advengsoft.2013.12.007) | `gwo` | swarm | Yes | Yes | No | `gwo.alpha_guidance`<br>`gwo.beta_guidance`<br>`gwo.delta_guidance`<br>`gwo.position_update` |
| [Greylag Goose Optimization](https://doi.org/10.1016/j.eswa.2023.122147) | `ggo` | swarm | Yes | Yes | No | `ggo.greylag_goose_flock_update` |
| [Growth Optimizer](https://doi.org/10.1016/j.knosys.2022.110206) | `go_growth` | swarm | Yes | Yes | No | `go_growth.growth_phase_update`<br>`go_growth.maturity_phase_update`<br>`go_growth.selection` |
| [Harmony Search Algorithm](https://doi.org/10.1177/003754970107600201) | `hsa` | trajectory | Yes | No | No | `hsa.harmony_memory_improvisation_update` |
| [Harris Hawks Optimization](https://doi.org/10.1016/j.future.2019.02.028) | `hho` | swarm | Yes | Yes | No | `hho.exploration`<br>`hho.soft_besiege`<br>`hho.hard_besiege`<br>`hho.soft_besiege_rapid_dive`<br>`hho.hard_besiege_rapid_dive`<br>`hho.levy_rapid_dive_refinement` |
| [Heap-Based Optimizer](https://doi.org/10.1016/j.eswa.2020.113702) | `hbo` | human | Yes | Yes | No | `hbo.heap_rank_pressure_update` |
| [Henry Gas Solubility Optimization](https://doi.org/10.1016/j.future.2019.07.015) | `hgso` | physics | Yes | Yes | No | `hgso.cluster_best_solubility_update`<br>`hgso.global_best_solubility_update`<br>`hgso.worst_agent_random_reset` |
| [Hiking Optimization Algorithm](https://doi.org/10.1016/j.knosys.2024.111880) | `hiking_oa` | human | Yes | Yes | No | `hiking_oa.hiking_slope_velocity_update` |
| [Hill Climb Algorithm](https://doi.org/10.1007/978-3-540-75256-1_52) | `hc` | trajectory | No | No | No | `hc.update` |
| [Hippopotamus Optimization Algorithm](https://doi.org/10.1038/s41598-024-54910-3) | `ho_hippo` | swarm | Yes | Yes | No | `ho_hippo.exploitation`<br>`ho_hippo.selection`<br>`ho_hippo.state_update`<br>`ho_hippo.group_defense_position_update`<br>`ho_hippo.predator_defense_update`<br>`ho_hippo.river_pond_position_update` |
| [Honey Badger Algorithm](https://doi.org/10.1016/j.matcom.2021.08.013) | `hba_honey` | swarm | Yes | Yes | No | `hba_honey.digging_phase_update`<br>`hba_honey.honey_phase_update` |
| [Horse Herd Optimization Algorithm](https://doi.org/10.1016/j.knosys.2020.106711) | `horse_oa` | swarm | Yes | Yes | No | `horse_oa.dominant_stallion_update`<br>`horse_oa.experienced_horse_social_update`<br>`horse_oa.middle_rank_grazing_update`<br>`horse_oa.foal_exploration_update` |
| [Human Conception Optimizer](https://doi.org/10.1038/s41598-022-25031-6) | `hco` | human | Yes | Yes | No | `hco.conception_growth_update` |
| [Human Evolutionary Optimization Algorithm](https://doi.org/10.1016/j.eswa.2023.122638) | `heoa` | human | Yes | Yes | No | `heoa.elite_local_refinement`<br>`heoa.learner_levy_best_attraction`<br>`heoa.explorer_centroid_escape`<br>`heoa.follower_best_contraction`<br>`heoa.risk_taker_best_sampling` |
| [Hunger Games Search](https://doi.org/10.1016/j.eswa.2021.114864) | `hgs` | nature | Yes | Yes | No | `hgs.random_hunger_exploration`<br>`hgs.hunger_weighted_approach`<br>`hgs.hunger_weighted_retreat` |
| [Hunting Search Algorithm](https://doi.org/10.1109/ICSCCW.2009.5379451) | `hus` | swarm | Yes | Yes | No | `hus.update` |
| [Hybrid Bat Algorithm](https://doi.org/10.48550/arXiv.1303.6310) | `hba` | swarm | Yes | Yes | No | `hba.bat_frequency_movement`<br>`hba.de_local_search` |
| [Hybrid Grey Wolf - Whale Optimization Algorithm](https://doi.org/10.1177/10775463211003402) | `gwo_woa` | swarm | Yes | No | No | `gwo_woa.selection`<br>`gwo_woa.elite_local_refinement`<br>`gwo_woa.leader_guided_population_update` |
| [Hybrid Improved Whale Optimization Algorithm](https://doi.org/10.1109/ICACCS.2019.8728514) | `hi_woa` | swarm | Yes | No | No | `hi_woa.selection`<br>`hi_woa.elite_local_refinement`<br>`hi_woa.whale_position_update` |
| [Hybrid Self-Adaptive Bat Algorithm](https://doi.org/10.1155/2014/709738) | `hsaba` | swarm | Yes | Yes | No | `hsaba.local_bat_random_walk`<br>`hsaba.velocity_bat_update`<br>`hsaba.differential_evolution_refinement` |
| [iLSHADE-RSP](https://doi.org/10.48550/arXiv.2006.02591) | `ilshade_rsp` | evolutionary | Yes | Yes | No | `ilshade_rsp.mutation`<br>`ilshade_rsp.crossover`<br>`ilshade_rsp.selection`<br>`ilshade_rsp.archive_update`<br>`ilshade_rsp.success_history_update`<br>`ilshade_rsp.population_reduction`<br>`ilshade_rsp.rank_selective_pressure`<br>`ilshade_rsp.weighted_pbest_scaling`<br>`ilshade_rsp.cauchy_target_perturbation` |
| [Imperialist Competitive Algorithm](https://doi.org/10.1109/CEC.2007.4425083) | `ica` | human | Yes | Yes | No | `ica.assimilation`<br>`ica.imperialist_revolution`<br>`ica.colony_revolution`<br>`ica.intra_empire_competition` |
| [Improved Adaptive Grey Wolf Optimization](https://doi.org/10.1007/s10462-024-10821-3) | `iagwo` | swarm | Yes | No | No | `iagwo.adaptive_alpha_beta_delta_update` |
| [Improved Artificial Ecosystem-based Optimization](https://doi.org/10.1016/j.ijhydene.2020.06.256) | `improved_aeo` | nature | Yes | No | No | `improved_aeo.selection`<br>`improved_aeo.ecosystem_producer_consumer_update`<br>`improved_aeo.improved_decomposition_refinement` |
| [Improved Artificial Rabbits Optimization](https://doi.org/10.1016/j.engappai.2022.105082) | `iaro` | swarm | Yes | No | No | `iaro.improved_rabbit_global_update`<br>`iaro.elite_local_refinement`<br>`iaro.selection` |
| [Improved Grey Wolf Optimizer](https://doi.org/10.1016/j.eswa.2020.113917) | `i_gwo` | swarm | Yes | Yes | No | `i_gwo.selection`<br>`i_gwo.alpha_guidance_trial`<br>`i_gwo.beta_guidance_trial`<br>`i_gwo.delta_guidance_trial`<br>`i_gwo.mean_leader_position_update` |
| [Improved Kepler Optimization Algorithm](https://doi.org/10.1016/j.eswa.2025.128216) | `ikoa` | physics | Yes | Yes | No | `ikoa.selection`<br>`ikoa.assignment_matching_position_update`<br>`ikoa.improved_matching_refinement_update` |
| [Improved L-SHADE](https://doi.org/10.1109/CEC.2016.7743922) | `ilshade` | evolutionary | Yes | Yes | No | `ilshade.linear_population_reduction_mutation_selection` |
| [Improved Multi-Operator Differential Evolution](https://doi.org/10.1109/CEC48606.2020.9185577) | `imode` | evolutionary | Yes | Yes | No | `imode.candidate_generation`<br>`imode.selection`<br>`imode.state_update`<br>`imode.initialization`<br>`imode.mutation`<br>`imode.crossover` |
| [Improved Opposite-based Learning Grey Wolf Optimizer](https://doi.org/10.1007/s12652-020-02153-1) | `iobl_gwo` | swarm | Yes | No | No | `iobl_gwo.selection`<br>`iobl_gwo.elite_local_refinement`<br>`iobl_gwo.leader_guided_population_update` |
| [Improved Queuing Search Algorithm](https://doi.org/10.1007/s12652-020-02849-4) | `improved_qsa` | human | Yes | No | No | `improved_qsa.selection`<br>`improved_qsa.queue_business_one_update`<br>`improved_qsa.queue_business_two_refinement` |
| [Improved Teaching-Learning-based Optimization](https://doi.org/10.1016/j.scient.2012.12.005) | `improved_tlo` | human | Yes | No | No | `improved_tlo.selection`<br>`improved_tlo.elite_local_refinement`<br>`improved_tlo.teacher_learner_population_update` |
| [Improved Whale Optimization Algorithm](https://doi.org/10.1016/j.jcde.2019.02.002) | `i_woa` | swarm | Yes | Yes | No | `i_woa.polynomial_breeding_refinement` |
| [Incremental model-based Grey Wolf Optimizer](https://doi.org/10.1007/s00366-019-00837-7) | `incremental_gwo` | swarm | Yes | No | No | `incremental_gwo.selection`<br>`incremental_gwo.elite_local_refinement`<br>`incremental_gwo.leader_guided_population_update` |
| [Invasive Weed Optimization](https://doi.org/10.1016/j.ecoinf.2006.07.003) | `iwo` | nature | Yes | Yes | No | `iwo.seed_dispersal_colonization_update` |
| [IPOP-CMA-ES](https://doi.org/10.1109/CEC.2005.1554902) | `ipop_cmaes` | evolutionary | Yes | Yes | Yes | `ipop_cmaes.initialization`<br>`ipop_cmaes.cmaes_sampling`<br>`ipop_cmaes.elite_recombination`<br>`ipop_cmaes.distribution_update`<br>`ipop_cmaes.population_restart`<br>`ipop_cmaes.boundary_penalty`<br>`ipop_cmaes.candidate_injection` |
| [Iterated Local Search](https://doi.org/10.1007/0-306-48056-5_11) | `ils` | trajectory | No | No | Yes | `ils.update` |
| [Ivy Algorithm](https://doi.org/10.1016/j.knosys.2024.111850) | `ivya` | nature | Yes | Yes | No | `ivya.neighbor_growth_update`<br>`ivya.best_growth_update` |
| [Jaya Algorithm](https://doi.org/10.5267/j.ijiec.2015.8.004) | `jy` | math | Yes | Yes | No | `jy.best_away_from_worst_update` |
| [Jellyfish Search Optimizer](https://doi.org/10.1016/j.amc.2020.125535) | `jso` | swarm | Yes | Yes | No | `jso.ocean_current_swarm_motion_update` |
| [jSO Differential Evolution](https://doi.org/10.1109/CEC.2017.7969362) | `jso_de` | evolutionary | Yes | Yes | No | `jso_de.mutation`<br>`jso_de.weighted_pbest_scaling`<br>`jso_de.crossover`<br>`jso_de.selection`<br>`jso_de.archive_update`<br>`jso_de.success_history_update`<br>`jso_de.population_reduction`<br>`jso_de.bound_resampling` |
| [Komodo Mlipir Algorithm](https://doi.org/10.1016/j.asoc.2021.108043) | `kma` | swarm | Yes | Yes | No | `kma.update` |
| [Krill Herd Algorithm](https://doi.org/10.1016/j.asoc.2016.08.041) | `kha` | swarm | Yes | No | No | `kha.crossover`<br>`kha.diffusion`<br>`kha.mutation`<br>`kha.selection`<br>`kha.state_update`<br>`kha.induced_movement_update` |
| [L-SHADE (SHADE with Linear Population Size Reduction)](https://doi.org/10.1109/CEC.2014.6900380) | `lshade` | evolutionary | Yes | Yes | No | `lshade.mutation`<br>`lshade.crossover`<br>`lshade.selection`<br>`lshade.archive_update`<br>`lshade.success_history_update`<br>`lshade.population_reduction` |
| [Leaf in Wind Optimization](https://doi.org/10.1109/ACCESS.2024.3390670) | `liwo` | physics | Yes | Yes | No | `liwo.breeze_spiral_translation`<br>`liwo.strong_wind_displacement` |
| [Life Choice-Based Optimizer](https://doi.org/10.1007/s00500-019-04443-z) | `lco` | human | Yes | Yes | No | `lco.life_choice_boundary_reflection_update` |
| [Light Spectrum Optimizer](https://doi.org/10.1016/j.asoc.2024.112318) | `lso_spectrum` | physics | Yes | Yes | No | `lso_spectrum.light_spectrum_position_update` |
| [Linear Subspace Surrogate Modeling Evolutionary Algorithm](https://doi.org/10.1109/TEVC.2023.3319640) | `l2smea` | evolutionary | Yes | Yes | No | `l2smea.update` |
| [Lion Optimization Algorithm](https://doi.org/10.1016/j.jcde.2015.06.003) | `loa` | swarm | Yes | Yes | No | `loa.nomad_roaming_update`<br>`loa.pride_mating_recombination`<br>`loa.pride_leader_roaming_update`<br>`loa.nomad_roaming_update.mutation`<br>`loa.pride_mating_recombination.mutation`<br>`loa.pride_leader_roaming_update.mutation`<br>`loa.territorial_takeover_exchange` |
| [Liver Cancer Algorithm](https://doi.org/10.1016/j.compbiomed.2023.107389) | `lca` | nature | Yes | Yes | No | `lca.best_cell_replication`<br>`lca.peer_lateral_invasion`<br>`lca.angiogenesis_mutation` |
| [Love Evolution Algorithm](https://doi.org/10.1007/s11227-024-05905-4) | `lea` | human | Yes | No | No | `lea.reflection_operation`<br>`lea.value_phase_reflection_operation`<br>`lea.value_phase_role_phase` |
| [LSHADE-cnEpSin](https://doi.org/10.1109/CEC.2016.7744173) | `lshade_cnepsin` | evolutionary | Yes | Yes | No | `lshade_cnepsin.cn_epsin_mutation_crossover_selection` |
| [LSHADE-EpSin](https://doi.org/10.1109/CEC.2016.7744313) | `lshade_epsin` | evolutionary | Yes | Yes | No | `lshade_epsin.mutation`<br>`lshade_epsin.crossover`<br>`lshade_epsin.selection`<br>`lshade_epsin.archive_update`<br>`lshade_epsin.success_history_update`<br>`lshade_epsin.population_reduction`<br>`lshade_epsin.sinusoidal_decreasing_f`<br>`lshade_epsin.sinusoidal_increasing_f`<br>`lshade_epsin.adaptive_frequency_update`<br>`lshade_epsin.lshade_second_phase_adaptation`<br>`lshade_epsin.gaussian_walk_local_search`<br>`lshade_epsin.bound_repair` |
| [LSHADE-RSP](https://doi.org/10.1109/CEC.2018.8477957) | `lshade_rsp` | evolutionary | Yes | Yes | No | `lshade_rsp.mutation`<br>`lshade_rsp.weighted_pbest_scaling`<br>`lshade_rsp.crossover`<br>`lshade_rsp.selection`<br>`lshade_rsp.archive_update`<br>`lshade_rsp.success_history_update`<br>`lshade_rsp.population_reduction`<br>`lshade_rsp.rank_selective_pressure`<br>`lshade_rsp.bound_resampling` |
| [LSHADE-SPACMA](https://doi.org/10.1109/CEC.2017.7969307) | `lshade_spacma` | evolutionary | Yes | Yes | No | `lshade_spacma.mutation`<br>`lshade_spacma.crossover`<br>`lshade_spacma.selection`<br>`lshade_spacma.archive_update`<br>`lshade_spacma.success_history_update`<br>`lshade_spacma.population_reduction`<br>`lshade_spacma.cma_es_sampling`<br>`lshade_spacma.cma_es_update`<br>`lshade_spacma.semi_parameter_adaptation`<br>`lshade_spacma.fcp_assignment`<br>`lshade_spacma.fcp_memory_update`<br>`lshade_spacma.lshade_branch`<br>`lshade_spacma.cma_branch`<br>`lshade_spacma.bound_repair` |
| [Lungs Performance-Based Optimization](https://doi.org/10.1016/j.cma.2023.116582) | `lpo` | nature | Yes | Yes | No | `lpo.lichen_growth_propagation_update` |
| [Lyrebird Optimization Algorithm](https://doi.org/10.1016/j.cma.2023.116436) | `loa_lyrebird` | swarm | Yes | Yes | No | `loa_lyrebird.better_bird_imitation_update`<br>`loa_lyrebird.escape_step_update` |
| [Lévy Flight and Selective Opposition Artificial Rabbit Algorithm](https://doi.org/10.3390/sym14112282) | `laro` | swarm | Yes | No | No | `laro.candidate_search`<br>`laro.selection`<br>`laro.candidate_generation`<br>`laro.initialization` |
| [Lévy Flight Distribution](https://doi.org/10.1016/j.engappai.2020.103731) | `lfd` | distribution | Yes | Yes | No | `lfd.levy_flight_search` |
| [Lévy Flight Jaya Algorithm](https://doi.org/10.1016/j.eswa.2020.113902) | `levy_ja` | distribution | Yes | No | No | `levy_ja.candidate_search`<br>`levy_ja.selection`<br>`levy_ja.candidate_generation`<br>`levy_ja.initialization` |
| [Magnificent Frigatebird Optimization](https://doi.org/10.32604/cmc.2024.054317) | `mfo` | swarm | Yes | No | No | `mfo.exploration_move`<br>`mfo.exploitation_move`<br>`mfo.replacement` |
| [Manta Ray Foraging Optimization](https://doi.org/10.1016/j.engappai.2019.103300) | `mrfo` | swarm | Yes | Yes | No | `mrfo.chain_foraging`<br>`mrfo.cyclone_random_foraging`<br>`mrfo.cyclone_best_foraging`<br>`mrfo.somersault_foraging` |
| [Mantis Shrimp Optimization Algorithm](https://doi.org/10.3390/math13091500) | `mshoa` | swarm | Yes | Yes | No | `mshoa.smasher_attack_update`<br>`mshoa.spearer_circular_attack_update`<br>`mshoa.defense_position_update` |
| [Marine Predators Algorithm](https://doi.org/10.1016/j.eswa.2020.113377) | `mpa` | swarm | Yes | Yes | No | `mpa.brownian_exploration`<br>`mpa.brownian_transition`<br>`mpa.levy_transition`<br>`mpa.levy_exploitation`<br>`mpa.fads` |
| [Market Game Optimization Algorithm](https://doi.org/10.1016/j.asoc.2024.112466) | `mgoa_market` | human | Yes | Yes | No | `mgoa_market.market_gradient_position_update` |
| [Memetic Algorithm](https://doi.org/10.1007/978-3-540-92910-9_29) | `memetic_a` | evolutionary | Yes | Yes | No | `memetic_a.candidate_generation`<br>`memetic_a.selection`<br>`memetic_a.recombination`<br>`memetic_a.mutation`<br>`memetic_a.mutate`<br>`memetic_a.xhc` |
| [Mirage-Search Optimizer](https://doi.org/10.1016/j.advengsoft.2025.103883) | `mso` | physics | Yes | Yes | No | `mso.superior_mirage_search_update`<br>`mso.inferior_mirage_search_update` |
| [mLSHADE-RL (Multi-operator Ensemble LSHADE with Restart and Local Search)](https://doi.org/10.48550/arXiv.2409.15994) | `mlshade_rl` | evolutionary | Yes | Yes | Yes | `mlshade_rl.ms1_current_to_pbest_weight_archive`<br>`mlshade_rl.ms2_current_to_pbest_no_archive`<br>`mlshade_rl.ms3_current_to_ordpbest_weight`<br>`mlshade_rl.crossover`<br>`mlshade_rl.selection`<br>`mlshade_rl.strategy_probability_update`<br>`mlshade_rl.parameter_adaptation`<br>`mlshade_rl.archive_update`<br>`mlshade_rl.population_reduction`<br>`mlshade_rl.restart`<br>`mlshade_rl.local_search` |
| [Modified Artificial Ecosystem-Based Optimization](https://doi.org/10.1109/ACCESS.2020.2973351) | `modified_aeo` | nature | Yes | No | No | `modified_aeo.selection`<br>`modified_aeo.ecosystem_producer_consumer_update`<br>`modified_aeo.modified_decomposition_refinement` |
| [Modified Equilibrium Optimizer](https://doi.org/10.1016/j.asoc.2020.106542) | `modified_eo` | physics | Yes | No | No | `modified_eo.selection`<br>`modified_eo.modified_equilibrium_pool_update`<br>`modified_eo.modified_local_refinement` |
| [Monarch Butterfly Optimization](https://doi.org/10.1007/s00521-015-1923-y) | `mbo` | swarm | Yes | Yes | No | `mbo.monarch_migration_adjusting_update` |
| [Monkey King Evolution V1](https://doi.org/10.1016/j.knosys.2016.01.009) | `mke` | evolutionary | Yes | Yes | No | `mke.king_learning_fluctuation_update`<br>`mke.peer_knowledge_difference_update` |
| [Moss Growth Optimization](https://doi.org/10.1093/jcde/qwae080) | `moss_go` | nature | Yes | Yes | No | `moss_go.water_dispersal_growth_update` |
| [Most Valuable Player Algorithm](https://doi.org/10.1007/s12351-017-0320-y) | `mvpa` | human | Yes | Yes | No | `mvpa.mvp_guided_player_update` |
| [Moth Flame Algorithm](https://doi.org/10.1016/j.knosys.2015.07.006) | `mfa` | swarm | Yes | Yes | No | `mfa.moth_flame_spiral_update` |
| [Moth Search Algorithm](https://doi.org/10.1007/s12293-016-0212-3) | `msa_e` | swarm | Yes | Yes | No | `msa_e.golden_ratio_exploitation_update` |
| [Mountain Gazelle Optimizer](https://doi.org/10.1016/j.advengsoft.2022.103282) | `mgo` | swarm | Yes | Yes | No | `mgo.territory_mountain_herding_update` |
| [Mountaineering Team-Based Optimization](https://doi.org/10.3390/math11051273) | `mtbo` | human | Yes | No | No | `mtbo.team_leader_coordinated_movement`<br>`mtbo.avalanche_worst_avoidance`<br>`mtbo.team_mean_movement`<br>`mtbo.random_relocation_phase`<br>`mtbo.candidate_generation`<br>`mtbo.selection` |
| [Multi-Start Local Search](https://doi.org/10.1007/0-306-48056-5_12) | `msls` | trajectory | No | No | Yes | `msls.update` |
| [Multi-Surrogate-Assisted Ant Colony Optimization](https://doi.org/10.1109/TCYB.2021.3064676) | `misaco` | swarm | Yes | Yes | No | `misaco.update` |
| [Multi-Verse Optimizer](https://doi.org/10.1007/s00521-015-1870-7) | `mvo` | swarm | Yes | Yes | No | `mvo.candidate_generation`<br>`mvo.selection`<br>`mvo.exploitation_move`<br>`mvo.replacement` |
| [Multifactorial Evolutionary Algorithm II](https://doi.org/10.1109/TEVC.2019.2906927) | `mfea2` | evolutionary | Yes | Yes | No | `mfea2.update` |
| [Multifactorial Evolutionary Algorithm](https://doi.org/10.1109/TEVC.2015.2458037) | `mfea` | evolutionary | Yes | Yes | No | `mfea.assortative_mating_mutation_transfer_update` |
| [Multiple Trajectory Search](https://doi.org/10.1109/CEC.2008.4631210) | `mts` | trajectory | Yes | Yes | No | `mts.multiple_trajectory_local_search_update` |
| [Multiswarm-Assisted Expensive Optimization](https://doi.org/10.1109/TCYB.2020.2967553) | `samso` | swarm | Yes | Yes | No | `samso.self_adaptive_migratory_swarm_update` |
| [Naked Mole-Rat Algorithm](https://doi.org/10.1007/s00521-019-04464-7) | `nmra` | swarm | Yes | Yes | No | `nmra.breeder_exploitation_update`<br>`nmra.worker_exploration_update` |
| [Narwhal Optimizer](https://doi.org/10.1038/s41598-024-61278-8) | `nwoa` | swarm | Yes | Yes | No | `nwoa.exploration_move`<br>`nwoa.exploitation_move`<br>`nwoa.replacement` |
| [Nelder-Mead Method](https://doi.org/10.1093/comjnl/7.4.308) | `nmm` | trajectory | Yes | Yes | No | `nmm.reflection_update`<br>`nmm.expansion_update`<br>`nmm.contraction_update`<br>`nmm.shrink_update` |
| [Neural Network-Based Dimensionality Reduction Evolutionary Algorithm](https://doi.org/10.1109/TEVC.2024.3400398) | `nndrea_so` | evolutionary | Yes | Yes | No | `nndrea_so.nn_weight_de_stage`<br>`nndrea_so.solution_de_stage` |
| [New Caledonian Crow Learning Algorithm](https://doi.org/10.1016/j.asoc.2020.106325) | `nccla` | swarm | Yes | No | No | `nccla.vertical_social_learning`<br>`nccla.horizontal_social_learning`<br>`nccla.individual_learning`<br>`nccla.juvenile_reinforcement`<br>`nccla.parent_reinforcement`<br>`nccla.parent_selection` |
| [Nizar Optimization Algorithm](https://doi.org/10.1007/s11227-023-05579-4) | `noa` | math | Yes | Yes | No | `noa.newton_position_update` |
| [NL-SHADE-LBC](https://doi.org/10.1109/CEC55065.2022.9870295) | `nlshade_lbc` | evolutionary | Yes | Yes | No | `nlshade_lbc.mutation`<br>`nlshade_lbc.crossover`<br>`nlshade_lbc.selection`<br>`nlshade_lbc.archive_update`<br>`nlshade_lbc.success_history_update`<br>`nlshade_lbc.population_reduction`<br>`nlshade_lbc.rank_selective_pressure`<br>`nlshade_lbc.linear_bias_change`<br>`nlshade_lbc.bound_resampling`<br>`nlshade_lbc.crossover_rate_sorting` |
| [NL-SHADE-RSP-Midpoint](https://doi.org/10.1109/CEC55065.2022.9870220) | `nlshade_rsp_midpoint` | evolutionary | Yes | Yes | Yes | `nlshade_rsp_midpoint.mutation`<br>`nlshade_rsp_midpoint.crossover_binomial`<br>`nlshade_rsp_midpoint.crossover_exponential`<br>`nlshade_rsp_midpoint.crossover_rate_sorting`<br>`nlshade_rsp_midpoint.selection`<br>`nlshade_rsp_midpoint.archive_update`<br>`nlshade_rsp_midpoint.adaptive_archive_probability`<br>`nlshade_rsp_midpoint.success_history_update`<br>`nlshade_rsp_midpoint.nonlinear_population_reduction`<br>`nlshade_rsp_midpoint.rank_selective_pressure`<br>`nlshade_rsp_midpoint.bound_resampling`<br>`nlshade_rsp_midpoint.bound_random_repair_fallback`<br>`nlshade_rsp_midpoint.midpoint_evaluation`<br>`nlshade_rsp_midpoint.midpoint_replacement`<br>`nlshade_rsp_midpoint.kmeans_midpoint`<br>`nlshade_rsp_midpoint.midpoint_restart`<br>`nlshade_rsp_midpoint.bounds_restart`<br>`nlshade_rsp_midpoint.restart` |
| [NL-SHADE-RSP](https://doi.org/10.1109/CEC45853.2021.9504959) | `nlshade_rsp` | evolutionary | Yes | Yes | No | `nlshade_rsp.mutation`<br>`nlshade_rsp.crossover_binomial`<br>`nlshade_rsp.crossover_exponential`<br>`nlshade_rsp.crossover_rate_sorting`<br>`nlshade_rsp.selection`<br>`nlshade_rsp.archive_update`<br>`nlshade_rsp.adaptive_archive_probability`<br>`nlshade_rsp.success_history_update`<br>`nlshade_rsp.nonlinear_population_reduction`<br>`nlshade_rsp.rank_selective_pressure`<br>`nlshade_rsp.bound_random_repair` |
| [NLAPSMjSO-EDA](https://doi.org/10.3390/sym17020153) | `nlapsmjso_eda` | evolutionary | Yes | No | No | `nlapsmjso_eda.sampling`<br>`nlapsmjso_eda.selection`<br>`nlapsmjso_eda.state_update`<br>`nlapsmjso_eda.non_linear_population_analysis_update` |
| [Northern Goshawk Optimization](https://doi.org/10.1109/ACCESS.2021.3133286) | `ngo` | swarm | Yes | Yes | No | `ngo.phase_one_update`<br>`ngo.pursuit_exploitation_update`<br>`ngo.selection` |
| [Nuclear Reaction Optimization](https://doi.org/10.1109/ACCESS.2019.2918406) | `nro` | physics | Yes | Yes | No | `nro.nuclear_fission_update`<br>`nro.nuclear_fusion_update`<br>`nro.selection` |
| [Numeric Crunch Algorithm](https://doi.org/10.1007/s00500-023-08925-z) | `nca` | math | Yes | Yes | No | `nca.acceleration_hyperbolic_contraction_random_subset_components` |
| [Opposition-based Coral Reefs Optimization](https://doi.org/10.2991/ijcis.d.190930.003) | `ocro` | evolutionary | Yes | No | No | `ocro.candidate_generation`<br>`ocro.selection`<br>`ocro.position_update`<br>`ocro.state_update`<br>`ocro.initialization` |
| [Opposition-based learning Grey Wolf Optimizer](https://doi.org/10.1016/j.knosys.2021.107139) | `ogwo` | swarm | Yes | No | No | `ogwo.selection`<br>`ogwo.elite_local_refinement`<br>`ogwo.leader_guided_population_update` |
| [Optimal Foraging Algorithm](https://doi.org/10.1016/j.eswa.2022.117735) | `ofa` | swarm | Yes | Yes | No | `ofa.owl_neighbour_flight_update` |
| [Osprey Optimization Algorithm](https://doi.org/10.3389/fmech.2022.1126450) | `ooa` | swarm | Yes | Yes | No | `ooa.hunting`<br>`ooa.search`<br>`ooa.selection`<br>`ooa.state_update`<br>`ooa.fish_carrying_local_update` |
| [Parameter-Free Bat Algorithm](https://www.iztok-jr-fister.eu/static/publications/124.pdf) | `plba` | swarm | Yes | Yes | No | `plba.path_looping_bat_update` |
| [Parent-Centric Crossover (G3-PCX style)](https://doi.org/10.1109/CEC.2004.1331141) | `pcx` | evolutionary | Yes | Yes | No | `pcx.parent_centric_crossover_update` |
| [Pareto Sequential Sampling](https://doi.org/10.1007/s00500-021-05853-8) | `pss` | math | Yes | Yes | No | `pss.prominent_domain_sampling_update`<br>`pss.full_domain_sampling_update`<br>`pss.mixed_domain_sampling_update` |
| [Parrot Optimizer](https://doi.org/10.1016/j.compbiomed.2024.108064) | `parrot_o` | swarm | Yes | Yes | No | `parrot_o.flight_area_search_update` |
| [Particle Swarm Optimization](https://doi.org/10.1109/ICNN.1995.488968) | `pso` | swarm | Yes | Yes | No | `pso.inertia_velocity_update`<br>`pso.cognitive_memory_update`<br>`pso.social_global_update` |
| [Pathfinder Algorithm](https://doi.org/10.1016/j.asoc.2019.03.012) | `pfa` | swarm | Yes | Yes | No | `pfa.pathfinder_position_update` |
| [Pelican Optimization Algorithm](https://doi.org/10.3390/s22030855) | `poa` | swarm | Yes | Yes | No | `poa.prey_pursuit_update`<br>`poa.water_surface_winging_update` |
| [Philoponella prominens Optimizer](https://doi.org/10.1007/s10586-024-04761-4) | `ppo` | swarm | Yes | No | No | `ppo.escape_sexual_cannibalism_juvenile_generation`<br>`ppo.escape_predation_local_search` |
| [Physical Education Teacher Inspired Optimization](https://doi.org/10.13140/RG.2.2.12097.06245) | `petio` | human | Yes | No | No | `petio.performance_evaluation_teaching_update` |
| [Pied Kingfisher Optimizer](https://doi.org/10.1007/s00521-024-09879-5) | `pko` | swarm | Yes | Yes | No | `pko.diving_beating_rate_update`<br>`pko.crest_angle_foraging_update`<br>`pko.hovering_attack_update`<br>`pko.population_escape_update` |
| [Polar Fox Optimization](https://doi.org/10.1007/s00521-024-10346-4) | `pfa_polar_fox` | swarm | Yes | No | No | `pfa_polar_fox.exploitation`<br>`pfa_polar_fox.selection`<br>`pfa_polar_fox.state_update`<br>`pfa_polar_fox.experience_phase`<br>`pfa_polar_fox.leader_guided_refinement_update`<br>`pfa_polar_fox.leader_phase` |
| [Polar Lights Optimizer](https://doi.org/10.1016/j.neucom.2024.128427) | `plo` | physics | Yes | Yes | No | `plo.aurora_global_local_update`<br>`plo.polar_light_collision_update` |
| [Political Optimizer](https://doi.org/10.1016/j.knosys.2020.105709) | `political_o` | human | Yes | Yes | No | `political_o.candidate_generation`<br>`political_o.selection`<br>`political_o.parliamentary` |
| [Poor and Rich Optimization Algorithm](https://doi.org/10.1016/j.engappai.2019.08.025) | `pro` | human | Yes | Yes | No | `pro.learning`<br>`pro.state_update`<br>`pro.candidate_generation`<br>`pro.selection` |
| [Population-Based Incremental Learning](https://doi.org/10.1109/SSE62657.2024.00022) | `pbil` | distribution | No | No | No | `pbil.update` |
| [Prairie Dog Optimization Algorithm](https://doi.org/10.1007/s00521-022-07530-9) | `pdo` | swarm | Yes | Yes | No | `pdo.prairie_dog_burrow_alarm_update` |
| [Puma Optimizer](https://doi.org/10.1007/s10586-023-04221-5) | `puma_o` | swarm | Yes | Yes | No | `puma_o.update` |
| [QLE Sine Cosine Algorithm](https://doi.org/10.1016/j.eswa.2021.116417) | `qle_sca` | math | Yes | No | No | `qle_sca.candidate_generation`<br>`qle_sca.selection`<br>`qle_sca.learning`<br>`qle_sca.state_update`<br>`qle_sca.initialization` |
| [Quadratic Interpolation Optimization](https://doi.org/10.1016/j.cma.2023.116446) | `qio` | math | Yes | Yes | No | `qio.three_point_quadratic_interpolation`<br>`qio.two_point_reflection_interpolation` |
| [Queuing Search Algorithm](https://doi.org/10.1007/s12652-020-02849-4) | `qsa` | human | Yes | Yes | No | `qsa.business1`<br>`qsa.business2`<br>`qsa.business3` |
| [Rain-Cloud Condensation Optimizer](https://doi.org/10.3390/eng6100281) | `rcco` | physics | Yes | No | No | `rcco.rain_cloud_convection_update`<br>`rcco.cloud_collision_local_update`<br>`rcco.selection` |
| [Random Forest Bayesian Optimization](https://doi.org/10.1023/A:1010933404324) | `rf_bo` | surrogate | No | No | No | `rf_bo.update` |
| [Random Search](https://doi.org/10.1016/j.advengsoft.2022.103141) | `random_s` | trajectory | Yes | Yes | No | `random_s.random_sampling_update` |
| [Rat Swarm Optimizer](https://doi.org/10.1007/s12652-020-02580-0) | `rso` | swarm | Yes | Yes | No | `rso.long_chasing_update`<br>`rso.short_chasing_update` |
| [RDEx-SOP](https://doi.org/10.48550/arXiv.2603.27089) | `rdex_sop` | evolutionary | Yes | Yes | No | `rdex_sop.standard_branch_mutation`<br>`rdex_sop.exploitation_biased_mutation`<br>`rdex_sop.binomial_crossover`<br>`rdex_sop.cauchy_local_perturbation`<br>`rdex_sop.greedy_selection`<br>`rdex_sop.dynamic_pbest_selection`<br>`rdex_sop.hybrid_rate_update`<br>`rdex_sop.success_history_update`<br>`rdex_sop.linear_population_reduction`<br>`rdex_sop.bound_resampling` |
| [Reconstructed Differential Evolution](https://doi.org/10.48550/arXiv.2404.16280) | `rde` | evolutionary | Yes | Yes | No | `rde.mutation_current_to_pbest`<br>`rde.mutation_current_to_order_pbest`<br>`rde.strategy_resource_allocation`<br>`rde.extended_rank_selective_pressure`<br>`rde.crossover`<br>`rde.cauchy_target_perturbation`<br>`rde.selection`<br>`rde.archive_update`<br>`rde.success_history_update`<br>`rde.linear_population_reduction`<br>`rde.bound_repair` |
| [Red-billed Blue Magpie Optimizer](https://doi.org/10.1007/s10462-024-10716-3) | `rbmo` | swarm | Yes | Yes | No | `rbmo.update` |
| [Remora Optimization Algorithm](https://doi.org/10.1016/j.eswa.2021.115665) | `roa` | swarm | Yes | Yes | No | `roa.remora_attempt_update` |
| [Reptile Search Algorithm](https://doi.org/10.1016/j.eswa.2021.116158) | `rsa` | swarm | Yes | Yes | No | `rsa.reptile_hunting_encircling_update` |
| [RIME-ice Algorithm](https://doi.org/10.1016/j.neucom.2023.02.010) | `rime` | physics | Yes | Yes | No | `rime.hard_rime_puncture_update` |
| [RMSProp](https://www.youtube.com/watch?v=defQQqkXEfE) | `rmsprop` | math | No | No | No | `rmsprop.candidate_generation`<br>`rmsprop.selection`<br>`rmsprop.search_direction`<br>`rmsprop.step_acceptance`<br>`rmsprop.initialization` |
| [Rock Hyraxes Swarm Optimization](https://doi.org/10.32604/cmc.2021.013648) | `rhso` | swarm | Yes | No | No | `rhso.rhinoceros_herd_position_update` |
| [RRT-based Optimizer](https://doi.org/10.1109/ACCESS.2025.3547537) | `rrto` | swarm | Yes | No | No | `rrto.adaptive_step_size_wandering`<br>`rrto.absolute_difference_step`<br>`rrto.boundary_based_step` |
| [RUNge Kutta Optimizer](https://doi.org/10.1016/j.eswa.2021.115079) | `run` | math | Yes | Yes | No | `run.selection`<br>`run.enhanced_solution_quality_update`<br>`run.runge_kutta_position_update` |
| [Rüppell's Fox Optimizer](https://doi.org/10.1007/s10586-024-04950-1) | `rfo` | swarm | Yes | Yes | No | `rfo.red_fox_smell_search_update` |
| [Sailfish Optimizer](https://doi.org/10.1016/j.engappai.2019.01.001) | `sfo` | swarm | Yes | Yes | No | `sfo.behavioral_move`<br>`sfo.selection` |
| [Salp Swarm Algorithm](https://doi.org/10.1016/j.advengsoft.2017.07.002) | `ssa` | swarm | Yes | Yes | No | `ssa.leader_plus_food_guidance`<br>`ssa.leader_minus_food_guidance`<br>`ssa.follower_front_chain_update`<br>`ssa.follower_rear_chain_update` |
| [Sammon Mapping Assisted Differential Evolution](https://doi.org/10.1016/j.petrol.2019.106633) | `sade_sammon` | evolutionary | Yes | Yes | No | `sade_sammon.sammon_surrogate_de_selection_update` |
| [Sand Cat Swarm Optimization](https://doi.org/10.1007/s00366-022-01604-x) | `scso` | swarm | Yes | Yes | No | `scso.exploration_move`<br>`scso.replacement`<br>`scso.selection`<br>`scso.exploitation_move`<br>`scso.candidate_generation` |
| [Satin Bowerbird Optimizer](https://doi.org/10.1016/j.engappai.2017.01.006) | `sbo` | swarm | Yes | Yes | No | `sbo.bowerbird_mutation_update` |
| [Sea Lion Optimization](https://doi.org/10.14569/IJACSA.2019.0100548) | `slo` | swarm | Yes | Yes | No | `slo.best_encircling_update`<br>`slo.random_peer_encircling_update`<br>`slo.spiral_attack_update` |
| [Seagull Optimization Algorithm](https://doi.org/10.1016/j.knosys.2018.11.024) | `soa` | swarm | Yes | Yes | No | `soa.seagull_spiral_attack_update` |
| [Seahorse Optimizer](https://doi.org/10.1007/s10489-022-03994-3) | `seaho` | swarm | Yes | Yes | No | `seaho.candidate_generation`<br>`seaho.selection`<br>`seaho.recombination` |
| [Search And Rescue Optimization](https://doi.org/10.1155/2019/2482543) | `saro` | human | Yes | Yes | No | `saro.candidate_generation`<br>`saro.selection`<br>`saro.individual`<br>`saro.candidate_search`<br>`saro.social` |
| [Search Space Independent Operator Based Deep Reinforcement Learning](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125444) | `ssio_rl` | evolutionary | Yes | Yes | No | `ssio_rl.update` |
| [Secant Optimization Algorithm](https://doi.org/10.1038/s41598-026-36691-z) | `secant_oa` | math | Yes | Yes | No | `secant_oa.secant_update`<br>`secant_oa.stochastic_exploitation`<br>`secant_oa.mutation_gate`<br>`secant_oa.selection` |
| [Secretary Bird Optimization Algorithm](https://doi.org/10.1007/s10462-024-10729-y) | `sboa` | swarm | Yes | Yes | No | `sboa.update` |
| [Self-Adaptive Bat Algorithm](https://doi.org/10.1155/2014/709738) | `saba` | swarm | Yes | Yes | No | `saba.self_adaptive_bat_update` |
| [Self-Adaptive Differential Evolution](https://doi.org/10.1109/CEC.2005.1554904) | `sade` | evolutionary | Yes | No | No | `sade.selection`<br>`sade.adaptive_strategy_de_update`<br>`sade.elite_local_refinement` |
| [Self-Adaptive Differential Evolution](https://doi.org/10.1109/TEVC.2006.872133) | `jde` | evolutionary | Yes | Yes | No | `jde.de_trial`<br>`jde.f_self_adaptation_trial`<br>`jde.cr_self_adaptation_trial`<br>`jde.f_cr_self_adaptation_trial` |
| [Sequential Quadratic Programming](https://doi.org/10.1017/S0962492900002518) | `sqp` | math | No | No | No | `sqp.update` |
| [Serval Optimization Algorithm](https://doi.org/10.3390/biomimetics7040204) | `serval_oa` | swarm | Yes | Yes | No | `serval_oa.hunting`<br>`serval_oa.selection`<br>`serval_oa.state_update` |
| [Shuffle-based Runner-Root Algorithm](https://doi.org/10.1007/978-3-319-70139-4_16) | `srsr` | swarm | Yes | Yes | No | `srsr.exploration`<br>`srsr.selection`<br>`srsr.1_accumulation_new_positions_via_gaussian` |
| [Siberian Tiger Optimization](https://doi.org/10.1109/ACCESS.2022.3229964) | `sto` | swarm | Yes | Yes | No | `sto.prey_hunting_update`<br>`sto.range_reduction_update` |
| [Simple Optimization Algorithm](https://scispace.com/pdf/an-efficient-metaheuristic-algorithm-for-engineering-2vvsafbir9.pdf) | `sopt` | distribution | Yes | Yes | No | `sopt.statistical_population_selection_update` |
| [Simulated Annealing](https://doi.org/10.1126/science.220.4598.671) | `sa` | trajectory | No | Yes | Yes | `sa.update` |
| [Sine Cosine Algorithm](https://doi.org/10.1016/j.knosys.2015.12.022) | `sine_cosine_a` | math | Yes | Yes | No | `sine_cosine_a.sine_position_update`<br>`sine_cosine_a.cosine_position_update` |
| [Singer Optimization Algorithm](https://doi.org/10.22266/ijies2025.0630.09) | `singer_oa` | human | Yes | Yes | No | `singer_oa.imitation_mimicry_phase`<br>`singer_oa.creation_perturbation_phase` |
| [Sinh Cosh Optimizer](https://doi.org/10.1016/j.knosys.2023.111081) | `scho` | math | Yes | Yes | No | `scho.scholar_chess_position_update` |
| [Slime Mould Algorithm](https://doi.org/10.1016/j.future.2020.03.055) | `sma` | nature | Yes | Yes | No | `sma.random_dispersion_update`<br>`sma.best_weighted_oscillation_update`<br>`sma.contracting_vibration_update` |
| [Snake Optimizer](https://doi.org/10.1016/j.knosys.2022.108320) | `so_snake` | swarm | Yes | Yes | No | `so_snake.male_snake_update`<br>`so_snake.female_snake_update`<br>`so_snake.selection` |
| [Snow Ablation Optimizer](https://doi.org/10.1016/j.eswa.2023.120069) | `snow_oa` | physics | Yes | Yes | No | `snow_oa.exploration_group_update`<br>`snow_oa.development_group_update` |
| [Social Ski-Driver Optimization](https://doi.org/10.1007/s00521-019-04159-z) | `ssdo` | human | Yes | Yes | No | `ssdo.sine_velocity_update`<br>`ssdo.cosine_velocity_update` |
| [Social Spider Algorithm](https://doi.org/10.1016/j.asoc.2015.02.014) | `sspider_a` | swarm | Yes | Yes | No | `sspider_a.social_spider_vibration_update` |
| [Social Spider Swarm Optimizer](https://doi.org/10.1016/j.eswa.2013.05.041) | `sso` | swarm | Yes | Yes | No | `sso.female_spider_position_update`<br>`sso.male_spider_position_update` |
| [Sparrow Search Algorithm](https://doi.org/10.1080/21642583.2019.1708830) | `sparrow_sa` | swarm | Yes | Yes | No | `sparrow_sa.producer_safe_foraging`<br>`sparrow_sa.producer_alarm_random_walk`<br>`sparrow_sa.scrounger_worst_avoidance`<br>`sparrow_sa.scrounger_best_following`<br>`sparrow_sa.awareness_best_escape`<br>`sparrow_sa.awareness_worst_escape` |
| [Spider Monkey Optimization](https://doi.org/10.1007/s12293-013-0128-0) | `smo` | swarm | Yes | Yes | No | `smo.local_leader_phase`<br>`smo.global_leader_phase`<br>`smo.local_leader_decision` |
| [Spotted Hyena Inspired Optimizer](https://doi.org/10.1016/j.advengsoft.2017.05.014) | `shio` | swarm | Yes | Yes | No | `shio.first_iguana_guidance`<br>`shio.second_iguana_guidance`<br>`shio.third_iguana_guidance` |
| [Spotted Hyena Optimizer](https://doi.org/10.1016/j.advengsoft.2017.05.014) | `sho` | swarm | Yes | Yes | No | `sho.spotted_hyena_hunting_update` |
| [Squirrel Search Algorithm](https://doi.org/10.1016/j.swevo.2018.02.013) | `squirrel_sa` | swarm | Yes | Yes | No | `squirrel_sa.acorn_to_hickory_glide`<br>`squirrel_sa.normal_to_acorn_glide`<br>`squirrel_sa.normal_to_hickory_glide`<br>`squirrel_sa.predator_random_relocation` |
| [Starfish Optimization Algorithm](https://doi.org/10.1007/s00521-024-10694-1) | `sfoa` | swarm | Yes | Yes | No | `sfoa.foraging`<br>`sfoa.state_update`<br>`sfoa.exploitation_move`<br>`sfoa.replacement`<br>`sfoa.exploration` |
| [Steepest Descent](https://doi.org/10.1006/hmat.1996.2146) | `sd` | math | No | No | No | `sd.candidate_generation`<br>`sd.selection`<br>`sd.search_direction`<br>`sd.step_acceptance`<br>`sd.initialization` |
| [Stellar Oscillator Optimization](https://doi.org/10.1007/s10586-024-04976-5) | `soo` | physics | Yes | Yes | No | `soo.selection`<br>`soo.1_oscillatory_position`<br>`soo.2_top_3_average_oscillatory` |
| [Student Psychology Based Optimization](https://doi.org/10.1016/j.advengsoft.2020.102804) | `spbo` | human | Yes | Yes | No | `spbo.groups`<br>`spbo.selection`<br>`spbo.average_student_phase_update`<br>`spbo.best_student`<br>`spbo.excellent_student_phase_update` |
| [Success-History Adaptive Differential Evolution](https://doi.org/10.1109/CEC.2014.6900380) | `shade` | evolutionary | Yes | Yes | No | `shade.success_history_mutation_crossover_selection` |
| [Success-History Intelligent Optimizer](https://doi.org/10.1007/s11227-021-04093-9) | `shio_success` | swarm | Yes | Yes | No | `shio_success.best_history_guidance`<br>`shio_success.second_history_guidance`<br>`shio_success.third_history_guidance` |
| [Superb Fairy-wren Optimization Algorithm](https://doi.org/10.1007/s10586-024-04901-w) | `superb_foa` | swarm | Yes | Yes | No | `superb_foa.global_smell_random_update`<br>`superb_foa.levy_food_attraction`<br>`superb_foa.best_food_convergence` |
| [Supply-Demand-Based Optimization](https://doi.org/10.1109/ACCESS.2019.2919408) | `supply_do` | human | Yes | Yes | No | `supply_do.quantity_equilibrium_update`<br>`supply_do.price_equilibrium_update` |
| [Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II](https://doi.org/10.1007/978-3-319-97773-7_4) | `sacc_eam2` | evolutionary | Yes | Yes | No | `sacc_eam2.even_subcomponent_de_update`<br>`sacc_eam2.odd_subcomponent_de_update` |
| [Surrogate-Assisted Cooperative Swarm Optimization](https://doi.org/10.1109/TEVC.2017.2675628) | `sacoso` | swarm | Yes | Yes | No | `sacoso.cognitive_swarm_update`<br>`sacoso.social_swarm_update` |
| [Surrogate-Assisted DE with Adaptive Multi-Subspace Search](https://doi.org/10.1109/TEVC.2022.3226837) | `sade_amss` | evolutionary | Yes | Yes | No | `sade_amss.adaptive_multistrategy_subspace_de_update` |
| [Surrogate-Assisted DE with Adaptive Training Data Selection Criterion](https://doi.org/10.1109/SSCI51031.2022.10022105) | `sade_atdsc` | evolutionary | Yes | Yes | No | `sade_atdsc.adaptive_trial_distribution_selection_update` |
| [Surrogate-Assisted Partial Optimization](https://doi.org/10.1007/978-3-031-70068-2_24) | `sapo` | evolutionary | Yes | Yes | No | `sapo.update` |
| [Swarm Robotics Search And Rescue](https://doi.org/10.1016/j.asoc.2017.02.028) | `srsr_robotics` | swarm | Yes | No | No | `srsr_robotics.candidate_generation`<br>`srsr_robotics.selection`<br>`srsr_robotics.guidance`<br>`srsr_robotics.state_update`<br>`srsr_robotics.exploration` |
| [Symbiotic Organisms Search](https://doi.org/10.1016/j.compstruc.2014.03.007) | `sos` | swarm | Yes | Yes | No | `sos.candidate_generation`<br>`sos.selection`<br>`sos.mutualism`<br>`sos.comensalism`<br>`sos.parasitism` |
| [Tabu Search](https://doi.org/10.1287/ijoc.1.3.190) | `ts` | trajectory | No | No | No | `ts.update` |
| [Tasmanian Devil Optimization](https://doi.org/10.1109/ACCESS.2022.3151641) | `tdo` | swarm | Yes | Yes | No | `tdo.exploration`<br>`tdo.state_update`<br>`tdo.hunting`<br>`tdo.exploitation` |
| [Teaching Learning Based Optimization](https://doi.org/10.1016/j.cad.2010.12.015) | `tlbo` | human | Yes | Yes | No | `tlbo.teacher_phase`<br>`tlbo.learner_phase` |
| [Teamwork Optimization Algorithm](https://doi.org/10.3390/s21134567) | `toa` | human | Yes | Yes | No | `toa.stage_1_supervisor_guidance`<br>`toa.learning`<br>`toa.state_update`<br>`toa.candidate_generation`<br>`toa.selection` |
| [Termite Life Cycle Optimizer](https://doi.org/10.1016/j.eswa.2022.119211) | `tlco` | swarm | Yes | Yes | No | `tlco.teacher_phase_update`<br>`tlco.learner_phase_update`<br>`tlco.selection` |
| [Tianji Horse Racing Optimizer](https://doi.org/10.1007/s10462-025-11269-9) | `thro` | human | Yes | Yes | No | `thro.throwing_race_update` |
| [Tornado Optimizer with Coriolis Force](https://doi.org/10.1007/s10462-025-11118-9) | `toc` | physics | Yes | Yes | No | `toc.fitness_proportional_assignment`<br>`toc.coriolis_velocity_update`<br>`toc.windstorm_to_tornado_evolution`<br>`toc.windstorm_to_thunderstorm_evolution`<br>`toc.thunderstorm_to_tornado_evolution`<br>`toc.random_windstorm_formation`<br>`toc.role_exchange_replacement` |
| [Tree Physiology Optimization](https://doi.org/10.1515/jisys-2017-0156) | `tpo` | nature | Yes | Yes | No | `tpo.carbon_nutrient_leaf_update` |
| [Tree-Seed Algorithm](https://doi.org/10.1016/j.eswa.2015.04.055) | `tree_seed_a` | nature | Yes | No | No | `tree_seed_a.toward_best_seed`<br>`tree_seed_a.away_random_seed` |
| [Triangulation Topology Aggregation Optimizer](https://doi.org/10.1016/j.eswa.2023.121744) | `ttao` | math | Yes | Yes | No | `ttao.crossover`<br>`ttao.selection`<br>`ttao.state_update`<br>`ttao.extra_candidate_diversification_update`<br>`ttao.random_population_refresh_update` |
| [Tug of War Optimization](https://doi.org/10.1007/978-3-030-04067-3_11) | `two` | physics | Yes | Yes | No | `two.tug_of_war_weight_force_update` |
| [Tuna Swarm Optimization](https://doi.org/10.1155/2021/9210050) | `tso` | swarm | Yes | Yes | No | `tso.leader_spiral_update`<br>`tso.random_migration_update`<br>`tso.spiral_following_update`<br>`tso.parabolic_foraging_update` |
| [Tunicate Swarm Algorithm](https://doi.org/10.1016/j.engappai.2020.103541) | `tsa` | swarm | Yes | Yes | No | `tsa.toward_best_tunicate_update`<br>`tsa.away_best_tunicate_update`<br>`tsa.swarm_chain_averaging_update` |
| [Turbulent Flow of Water-based Optimization](https://doi.org/10.1016/j.engappai.2020.103666) | `tfwo` | physics | Yes | No | No | `tfwo.effect_of_objects`<br>`tfwo.random_object_relocation`<br>`tfwo.effect_of_whirlpools`<br>`tfwo.best_whirlpool_preservation`<br>`tfwo.object_whirlpool_exchange`<br>`tfwo.state_structure_update` |
| [Variable Neighborhood Search](https://doi.org/10.1016/S0305-0548(97)00031-2) | `vns` | trajectory | No | No | Yes | `vns.update` |
| [Virus Colony Search](https://doi.org/10.1016/j.advengsoft.2015.11.004) | `vcs` | swarm | Yes | Yes | No | `vcs.virus_diffusion`<br>`vcs.host_cell_infection`<br>`vcs.immune_response` |
| [Walrus Optimization Algorithm](https://doi.org/10.1038/s41598-023-35863-5) | `waoa` | swarm | Yes | Yes | No | `waoa.feeding_exploration_update`<br>`waoa.range_narrowing_exploitation` |
| [War Strategy Optimization](https://doi.org/10.1109/ACCESS.2022.3153493) | `warso` | human | Yes | Yes | No | `warso.attack_strategy_update`<br>`warso.defense_strategy_update` |
| [Water Cycle Algorithm](https://doi.org/10.1016/j.compstruc.2012.07.010) | `wca` | nature | Yes | Yes | No | `wca.stream_toward_river`<br>`wca.stream_river_exchange`<br>`wca.river_toward_sea`<br>`wca.evaporation_raining` |
| [Water Uptake and Transport in Plants](https://doi.org/10.1007/s00521-025-11228-z) | `wutp` | nature | Yes | Yes | No | `wutp.horizontal_water_transport_update` |
| [Wave Optimization Algorithm](https://doi.org/10.1016/j.cor.2014.10.008) | `wo_wave` | physics | Yes | Yes | No | `wo_wave.wave_propagation_position_update` |
| [Wavelet Mutation and Quadratic Interpolation MRFO](https://doi.org/10.1016/j.knosys.2021.108071) | `wmqimrfo` | swarm | Yes | No | No | `wmqimrfo.selection`<br>`wmqimrfo.elite_local_refinement`<br>`wmqimrfo.weighted_multi_quadratic_mrfo_update` |
| [Weighting and Inertia Random Walk Optimizer](https://doi.org/10.1016/j.eswa.2022.116516) | `info` | math | Yes | Yes | No | `info.best_weighted_mean_rule`<br>`info.random_weighted_mean_rule` |
| [Whale Fruit-fly Optimization Algorithm](https://doi.org/10.1016/j.eswa.2020.113502) | `whale_foa` | swarm | Yes | No | No | `whale_foa.selection`<br>`whale_foa.elite_local_refinement`<br>`whale_foa.whale_position_update` |
| [Whale Optimization Algorithm](https://doi.org/10.1016/j.advengsoft.2016.01.008) | `woa` | swarm | Yes | Yes | No | `woa.search_for_prey`<br>`woa.encircling_prey`<br>`woa.spiral_bubble_net` |
| [White Shark Optimizer](https://doi.org/10.1016/j.knosys.2022.108457) | `wso` | swarm | Yes | Yes | No | `wso.white_shark_swarm_position_update` |
| [Wildebeest Herd Optimization](https://doi.org/10.3233/JIFS-190495) | `who` | swarm | Yes | Yes | No | `who.selection`<br>`who.1_local_movement_milling`<br>`who.2_herd_instinct`<br>`who.social_memory` |
| [Wind Driven Optimization](https://doi.org/10.1109/APS.2010.5562213) | `wdo` | physics | Yes | Yes | No | `wdo.wind_velocity_position_update` |
| [Wolverine Optimization Algorithm](https://doi.org/10.32604/cmes.2024.055171) | `wooa` | swarm | Yes | Yes | No | `wooa.scavenging_predator_following`<br>`wooa.prey_attack_update`<br>`wooa.fight_chase_local_update` |
| [Young's Double-Slit Experiment Optimizer](https://doi.org/10.1016/j.cma.2022.115652) | `ydse` | physics | Yes | Yes | No | `ydse.central_bright_fringe_update`<br>`ydse.bright_fringe_interference_update`<br>`ydse.dark_fringe_interference_update` |
| [Yukthi Opus](https://doi.org/10.48550/arXiv.2601.01832) | `yo` | trajectory | Yes | No | No | `yo.mcmc_burn_in`<br>`yo.post_burnin_selection`<br>`yo.mcmc_proposal`<br>`yo.greedy_refinement`<br>`yo.simulated_annealing_acceptance`<br>`yo.blacklist_filter`<br>`yo.adaptive_reheating`<br>`yo.elite_update` |
| [Zebra Optimization Algorithm](https://doi.org/10.1109/ACCESS.2022.3172789) | `zoa` | swarm | Yes | Yes | No | `zoa.behavioral_move`<br>`zoa.selection`<br>`zoa.candidate_generation` |

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
| Branin RCOS | `branin_rcos` | *x<sub>1</sub>* ∈ [-5, 10], *x<sub>2</sub>* ∈ [0, 15] | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0.3978873577; *(x<sub>1</sub>, x<sub>2</sub>)* = (-π, 12.275); (π, 2.275); (3π, 2.475) |
| Bukin F6 | `bukin_6` | *x<sub>1</sub>* ∈ [-15, -5], *x<sub>2</sub>* ∈ [-3, 3] | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (-10, 1) |
| Cross-in-Tray | `cross_in_tray` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ -2.0626118708; *(x<sub>1</sub>, x<sub>2</sub>)* = (±1.349406609, ±1.349406609) |
| Drop-Wave | `drop_wave` | [-5.12, 5.12]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = -1; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Easom | `easom` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = -1; *(x<sub>1</sub>, x<sub>2</sub>)* = (π, π) |
| Eggholder | `eggholder` | [-512, 512]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ -959.6407; *(x<sub>1</sub>, x<sub>2</sub>)* ≈ (512, 404.2319) |
| Goldstein-Price | `goldstein_price` | [-2, 2]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 3; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, -1) |
| Himmelblau | `himmelblau` | [-5, 5]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (3, 2); (-2.805118, 3.131312); (-3.779310, -3.283186); (3.584428, -1.848126) |
| Hölder Table | `holder_table` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ -19.208502568; *(x<sub>1</sub>, x<sub>2</sub>)* = (±8.055023472, ±9.664590029) |
| Levi F13 | `levi_13` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (1, 1) |
| Matyas | `matyas` | [-10, 10]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| McCormick | `mccormick` | *x<sub>1</sub>* ∈ [-1.5, 4], *x<sub>2</sub>* ∈ [-3, 4] | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ -1.913222955; *(x<sub>1</sub>, x<sub>2</sub>)* ≈ (-0.54719756, -1.54719756) |
| Schaffer F2 | `schaffer_2` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Schaffer F4 | `schaffer_4` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ 0.292578632; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, ±1.25313), (±1.25313, 0) |
| Schaffer F6 | `schaffer_6` | [-100, 100]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |
| Six-Hump Camel Back | `six_hump_camel_back` | *x<sub>1</sub>* ∈ [-3, 3], *x<sub>2</sub>* ∈ [-2, 2] | *f(x<sub>1</sub>, x<sub>2</sub>)* ≈ -1.031628453; *(x<sub>1</sub>, x<sub>2</sub>)* = (0.089842, -0.712656); (-0.089842, 0.712656) |
| Three-Hump Camel Back | `three_hump_camel_back` | [-5, 5]<sup>2</sup> | *f(x<sub>1</sub>, x<sub>2</sub>)* = 0; *(x<sub>1</sub>, x<sub>2</sub>)* = (0, 0) |

### D-Dimensional Functions

| Function | ID | Domain | Global Minimum |
|---|---|---|---|
| Alpine 1 | `alpine_1` | [-10, 10]<sup>D</sup> | *f(x)* = 0; *x<sub>i</sub>* = 0, *i* = 1, ..., *D* |
| Alpine 2 | `alpine_2` | [0, 10]<sup>D</sup> | *f(x)* ≈ -(2.808131180)<sup>D</sup>; *x<sub>i</sub>* ≈ 7.917052698 [N1] |
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
| Stepint | `stepint` | [-5.12, 5.12]<sup>D</sup> | *f(x)* = 25 - 6*D*; *x<sub>i</sub>* ∈ [-5.12, -5) [N8] |
| Styblinski-Tang | `styblinski_tang` | [-5, 5]<sup>D</sup> | *f(x)* ≈ -39.166165704*D*; *x<sub>i</sub>* ≈ -2.903534028 |
| Trid | `trid` | [-D<sup>2</sup>, D<sup>2</sup>]<sup>D</sup> | *f(x)* = -*D*(*D* + 4)(*D* - 1) / 6; *x<sub>i</sub>* = *i*(*D* + 1 - *i*) |
| Weierstrass | `weierstrass` | [-0.5, 0.5]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |
| Whitley | `whitley` | [-10.24, 10.24]<sup>D</sup> | *f(x)* = 0; *x* = 1<sub>D</sub> |
| Zakharov | `zakharov` | [-5, 10]<sup>D</sup> | *f(x)* = 0; *x* = 0<sub>D</sub> |

### CEC 2022 Functions

| Function | ID | Domain | Global Minimum |
|---|---|---:|---:|
| CEC 2022 F1 | `cec_2022_f01` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 300 |
| CEC 2022 F2 | `cec_2022_f02` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 400 |
| CEC 2022 F3 | `cec_2022_f03` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 600 |
| CEC 2022 F4 | `cec_2022_f04` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 800 |
| CEC 2022 F5 | `cec_2022_f05` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 900 |
| CEC 2022 F6 | `cec_2022_f06` | [-100, 100]<sup>D</sup>, *D* = 10, 20 | *f(x)* = 1800 |
| CEC 2022 F7 | `cec_2022_f07` | [-100, 100]<sup>D</sup>, *D* = 10, 20 | *f(x)* = 2000 |
| CEC 2022 F8 | `cec_2022_f08` | [-100, 100]<sup>D</sup>, *D* = 10, 20 | *f(x)* = 2200 |
| CEC 2022 F9 | `cec_2022_f09` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 2300 |
| CEC 2022 F10 | `cec_2022_f10` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 2400 |
| CEC 2022 F11 | `cec_2022_f11` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 2600 |
| CEC 2022 F12 | `cec_2022_f12` | [-100, 100]<sup>D</sup>, *D* = 2, 10, 20 | *f(x)* = 2700 |

### BBOB Functions

The 24 noiseless BBOB functions are deterministic instance-based benchmarks. Use `get_bbob_function(function_id, dimension, instance)` to retrieve a callable function and `get_bbob_optimum(function_id, dimension, instance)` to retrieve the corresponding shifted optimizer and optimum value.

| Function | ID | Domain | Global Minimum |
|---|---|---|---|
| BBOB F01: Sphere | `bbob_f01` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(1, dimension, instance)` |
| BBOB F02: Ellipsoidal | `bbob_f02` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(2, dimension, instance)` |
| BBOB F03: Rastrigin | `bbob_f03` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(3, dimension, instance)` |
| BBOB F04: Bueche-Rastrigin | `bbob_f04` | [-5, 5]<sup>D</sup>, *D* ≥ 2 |`get_bbob_optimum(4, dimension, instance)` |
| BBOB F05: Linear Slope | `bbob_f05` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(5, dimension, instance)` |
| BBOB F06: Attractive Sector | `bbob_f06` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(6, dimension, instance)` |
| BBOB F07: Step Ellipsoidal | `bbob_f07` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(7, dimension, instance)` |
| BBOB F08: Rosenbrock | `bbob_f08` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(8, dimension, instance)` |
| BBOB F09: Rosenbrock Rotated | `bbob_f09` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(9, dimension, instance)` |
| BBOB F10: Ellipsoidal Rotated | `bbob_f10` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(10, dimension, instance)` |
| BBOB F11: Discus | `bbob_f11` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(11, dimension, instance)` |
| BBOB F12: Bent Cigar | `bbob_f12` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(12, dimension, instance)` |
| BBOB F13: Sharp Ridge | `bbob_f13` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(13, dimension, instance)` |
| BBOB F14: Different Powers | `bbob_f14` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(14, dimension, instance)` |
| BBOB F15: Rastrigin Rotated | `bbob_f15` | [-5, 5]<sup>D</sup>, *D* ≥ 2 |  `get_bbob_optimum(15, dimension, instance)` |
| BBOB F16: Weierstrass | `bbob_f16` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(16, dimension, instance)` |
| BBOB F17: Schaffers F7, condition 10 | `bbob_f17` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(17, dimension, instance)` |
| BBOB F18: Schaffers F7, condition 1000 | `bbob_f18` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(18, dimension, instance)` |
| BBOB F19: Griewank-Rosenbrock | `bbob_f19` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(19, dimension, instance)` |
| BBOB F20: Schwefel | `bbob_f20` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(20, dimension, instance)` |
| BBOB F21: Gallagher 101 Peaks | `bbob_f21` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(21, dimension, instance)` |
| BBOB F22: Gallagher 21 Peaks | `bbob_f22` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(22, dimension, instance)` |
| BBOB F23: Katsuura | `bbob_f23` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(23, dimension, instance)` |
| BBOB F24: Lunacek bi-Rastrigin | `bbob_f24` | [-5, 5]<sup>D</sup>, *D* ≥ 2 | `get_bbob_optimum(24, dimension, instance)` |

### Engineering Design Benchmarks

Engineering benchmarks expose an objective function, along with bounds and constraints. Use `get_engineering_benchmark("<id>")` to retrieve `objective`, `constraints`, `min_values`, `max_values`, and best-known metadata. Constraint functions follow the package convention *g(x)* ≤ 0. [N9]

| Function | ID | Domain | Global Minimum | Constraints |
|---|---|---|---|---|
| Cantilever beam design | `cantilever_beam` | *x<sub>i</sub>* ∈ [0.01, 100], *i* = 1, ..., 5 | *f(x)* ≈ 1.339956; *x* ≈ (6.016016, 5.309173, 4.494330, 3.501475, 2.152665) | 1 inequality |
| Gear train design | `gear_train` | integer *x<sub>i</sub>* ∈ [12, 60], *i* = 1, ..., 4 | *f(x)* ≈ 2.700857 × 10<sup>-12</sup>; *x* = (16, 19, 43, 49) [N10] | box + integrality |
| Pressure vessel design, continuous relaxation | `pressure_vessel` | *T<sub>s</sub>*, *T<sub>h</sub>* ∈ [0, 99], *R* ∈ [10, 200], *L* ∈ [10, 240] | *f(x)* ≈ 5804.376217; *(T<sub>s</sub>, T<sub>h</sub>, R, L)* ≈ (0.727591, 0.359649, 37.699012, 240) [N11] | 4 inequalities |
| Pressure vessel design, discrete thickness | `pressure_vessel_discrete` | *T<sub>s</sub>*, *T<sub>h</sub>* rounded upward to multiples of 1/16; *R* ∈ [10, 200], *L* ∈ [10, 240] | *f(x)* ≈ 6059.714335; *(T<sub>s</sub>, T<sub>h</sub>, R, L)* ≈ (0.8125, 0.4375, 42.098446, 176.636596) [N11] | 4 inequalities |
| Speed reducer design | `speed_reducer` | 7 bounded design variables | *f(x)* ≈ 2994.471066; *x* ≈ (3.5, 0.7, 17, 7.3, 7.71532, 3.35021, 5.28665) | 11 inequalities |
| Tension/compression spring design | `tension_spring` | *d* ∈ [0.05, 2], *D* ∈ [0.25, 1.30], *N* ∈ [2, 15] | *f(x)* ≈ 0.012665; *(d, D, N)* ≈ (0.05169, 0.35675, 11.2871) | 4 inequalities |
| Three-bar truss design | `three_bar_truss` | *A<sub>1</sub>*, *A<sub>2</sub>* ∈ [0, 1] | *f(x)* ≈ 263.895843; *(A<sub>1</sub>, A<sub>2</sub>)* ≈ (0.788675, 0.408248) | 3 inequalities |
| Welded beam design | `welded_beam` | *h* ∈ [0.1, 2], *l* ∈ [0.1, 10], *t* ∈ [0.1, 10], *b* ∈ [0.1, 2] | *f(x)* ≈ 1.724852; *(h, l, t, b)* ≈ (0.20573, 3.47049, 9.03662, 0.20573) | 7 inequalities |

### Notes

<table>
  <thead>
    <tr>
      <th align="left">Note</th>
      <th align="left">Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>N1</strong></td>
      <td>
        Alpine 2 and Cosine Mixture have sign-convention traps in the literature.
        This package uses minimization-compatible signs.
      </td>
    </tr>
    <tr>
      <td><strong>N2</strong></td>
      <td>
        Katsuura is implemented as the product expression minus 1, so the exposed
        minimum is 0 at the origin.
      </td>
    </tr>
    <tr>
      <td><strong>N3</strong></td>
      <td>
        Michalewicz has no single dimension-free closed-form optimum.
        For <em>m</em> = 10, common reference values are approximately:<br>
        <em>D</em> = 2, <em>f</em><sup>*</sup> = −1.8013;<br>
        <em>D</em> = 5, <em>f</em><sup>*</sup> = −4.6877;<br>
        <em>D</em> = 10, <em>f</em><sup>*</sup> = −9.6602.
      </td>
    </tr>
    <tr>
      <td><strong>N4</strong></td>
      <td>
        Modified Schwefel is exposed in shifted CEC-style coordinates, so the
        visible optimizer is 0<sub>D</sub>.
      </td>
    </tr>
    <tr>
      <td><strong>N5</strong></td>
      <td>
        Powell requires <em>D</em> to be a multiple of 4.
      </td>
    </tr>
    <tr>
      <td><strong>N6</strong></td>
      <td>
        This is the cumulative ridge implementation, not the BBOB sharp-ridge function.
      </td>
    </tr>
    <tr>
      <td><strong>N7</strong></td>
      <td>
        Step functions have optimizer intervals, not isolated optimizer points.
      </td>
    </tr>
    <tr>
      <td><strong>N8</strong></td>
      <td>
        Stepint is bound-dependent. With bounds [−5.12, 5.12]<sup>D</sup>,
        <em>f</em><sup>*</sup> = 25 − 6<em>D</em>; without bounds, it is unbounded below.
      </td>
    </tr>
    <tr>
      <td><strong>N9</strong></td>
      <td>
        Engineering-design rows are constrained benchmarks. The Python module exposes
        <code>get_engineering_benchmark(id)</code>, so users can pass the returned
        objective, bounds, and constraints directly to <code>pymetaheuristic.optimize</code>.
      </td>
    </tr>
    <tr>
      <td><strong>N10</strong></td>
      <td>
        Gear train is a discrete integer benchmark. The implementation rounds variables
        to the nearest integer tooth counts by default.
      </td>
    </tr>
    <tr>
      <td><strong>N11</strong></td>
      <td>
        Pressure vessel has two common variants. <code>pressure_vessel</code> is the
        continuous relaxation; <code>pressure_vessel_discrete</code> rounds shell/head
        thickness upward to multiples of 1/16 before objective and constraint evaluation.
      </td>
    </tr>
  </tbody>
</table>


---
## 5. **Other Libraries**

[Back to Summary](#b-summary)

* For Multiobjective Optimization or Many Objectives Optimization, try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)
* For Traveling Salesman Problems (TSP), try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

### **Acknowledgement**

This section is dedicated to everyone who helped improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)


