from __future__ import annotations

ROOT_EXPORTS = [
    'list_algorithms', 'get_algorithm_info', 'create_optimizer', 'optimize',
    'Callback', 'CallbackList', 'EarlyStopping', 'HistoryRecorder', 'ProgressPrinter',
    'cooperative_optimize', 'replay_cooperative_result',
    'summarize_result', 'summarize_cooperative_result', 'convergence_data',
    'export_history_csv', 'export_island_telemetry_csv',
    'export_population_snapshots_json', 'export_replay_manifest_json',
    'plot_convergence', 'compare_convergence', 'plot_population_snapshot',
    'plot_benchmark_summary', 'plot_function', 'plot_function_1d',
    'plot_function_2d', 'plot_function_3d', 'plot_function_nd',
    'plot_function_contour', 'plot_function_surface',
    'plot_island_dynamics', 'plot_collaboration_network',
    'Termination',
    'plot_global_best_chart', 'plot_diversity_chart',
    'plot_explore_exploit_chart', 'plot_runtime_chart',
    'plot_run_dashboard', 'plot_diversity_comparison',
    'FloatVar', 'IntegerVar', 'BinaryVar', 'CategoricalVar', 'PermutationVar',
    'build_problem_spec', 'encode_position', 'decode_position',
    'Problem', 'FunctionalProblem', 'SphereProblem', 'RastriginProblem',
    'AckleyProblem', 'RosenbrockProblem', 'ZakharovProblem', 'get_test_problem',
    'get_engineering_benchmark', 'list_engineering_benchmarks', 'get_test_function_info',
    'validate_known_optima', 'validate_engineering_benchmarks',
    'get_cec2022_optimum', 'validate_cec2022_optima',
    'ConstrainedFunctionalProblem', 'get_engineering_problem', 'list_engineering_problems',
    'ChaoticMap', 'chaotic_sequence', 'chaotic_population', 'AVAILABLE_CHAOTIC_MAPS',
    'vstf_01', 'vstf_02', 'vstf_03', 'vstf_04',
    'sstf_01', 'sstf_02', 'sstf_03', 'sstf_04',
    'apply_transfer', 'binarize', 'BinaryAdapter', 'AVAILABLE_TRANSFER_FUNCTIONS',
    'limit', 'limit_inverse', 'wang', 'rand', 'reflect',
    'get_repair_function', 'AVAILABLE_REPAIR_STRATEGIES',
    'uniform_population', 'lhs_population', 'obl_population', 'sobol_population',
    'chaotic_init_function', 'get_init_function', 'AVAILABLE_INIT_STRATEGIES',
    'levy_flight',
    'HyperparameterTuner', 'BenchmarkRunner',
    'save_result', 'load_result', 'save_checkpoint', 'load_checkpoint',
    'result_to_json', 'result_from_json',
]

ARGUMENT_REFERENCE = [
    {'callable': 'list_algorithms', 'signature': '()', 'summary': 'Return the algorithm identifiers available in the package.', 'arguments': []},
    {'callable': 'get_algorithm_info', 'signature': '(algorithm: str)', 'summary': 'Return metadata about one algorithm, including defaults, capability flags, and the human-readable info() string.', 'arguments': [{'name': 'algorithm', 'required': True, 'default': None, 'annotation': 'str', 'description': 'Algorithm identifier.', 'accepted_values': 'Use one value returned by list_algorithms().'}]},
    {'callable': 'create_optimizer', 'signature': '(algorithm, target_function=None, min_values=None, max_values=None, objective="min", constraints=None, constraint_handler=None, variable_types=None, repair_function=None, repair_name=None, callbacks=None, init_function=None, init_name=None, problem=None, penalty_coefficient=1000000.0, equality_tolerance=1e-06, resample_attempts=25, max_steps=None, max_evaluations=None, target_fitness=None, seed=None, verbose=False, store_history=True, store_population_snapshots=False, snapshot_interval=1, timeout_seconds=None, config=None, termination=None, **params)', 'summary': 'Create and return an engine instance (does not run it). Supports callbacks, named repair strategies, named initialisation presets, and Problem objects.', 'arguments': [
        {'name': 'algorithm', 'required': True, 'default': None, 'annotation': 'str', 'description': 'Algorithm identifier.', 'accepted_values': 'Use one value returned by list_algorithms().'},
        {'name': 'target_function', 'required': False, 'default': 'None', 'annotation': 'callable | Problem | None', 'description': 'Objective function or a Problem object.', 'accepted_values': None},
        {'name': 'min_values', 'required': False, 'default': 'None', 'annotation': 'sequence | None', 'description': 'Lower bounds. Optional when problem= or a Problem object is supplied.', 'accepted_values': None},
        {'name': 'max_values', 'required': False, 'default': 'None', 'annotation': 'sequence | None', 'description': 'Upper bounds. Optional when problem= or a Problem object is supplied.', 'accepted_values': None},
        {'name': 'repair_name', 'required': False, 'default': 'None', 'annotation': 'str | None', 'description': 'Named repair strategy.', 'accepted_values': 'clip, limit, limit_inverse, wrap, wang, rand, random, reflect'},
        {'name': 'callbacks', 'required': False, 'default': 'None', 'annotation': 'Callback | list[Callback] | None', 'description': 'Lifecycle callbacks fired before/after the run and before/after each iteration.', 'accepted_values': None},
        {'name': 'init_function', 'required': False, 'default': 'None', 'annotation': 'callable | None', 'description': 'Custom population initialiser receiving (problem, pop_size, rng).', 'accepted_values': None},
        {'name': 'init_name', 'required': False, 'default': 'None', 'annotation': 'str | None', 'description': 'Named initialisation preset.', 'accepted_values': 'uniform, lhs, obl, sobol, chaotic:<map>'},
        {'name': 'problem', 'required': False, 'default': 'None', 'annotation': 'Problem | ProblemSpec | None', 'description': 'Alternative object-based problem definition.', 'accepted_values': None},
        {'name': 'termination', 'required': False, 'default': 'None', 'annotation': 'Termination | dict | None', 'description': 'Composable stopping criteria.', 'accepted_values': None},
        {'name': '**params', 'required': False, 'default': None, 'annotation': 'dict', 'description': 'Algorithm-specific keyword arguments.', 'accepted_values': 'Algorithm-specific.'},
    ]},
    {'callable': 'optimize', 'signature': '(*args, **kwargs)', 'summary': 'Shortcut: create_optimizer(*args, **kwargs).run().', 'arguments': [{'name': '*args, **kwargs', 'required': False, 'default': None, 'annotation': 'same as create_optimizer', 'description': 'Supports the same public arguments as create_optimizer(...).', 'accepted_values': 'See create_optimizer.'}]},
    {'callable': 'Callback', 'signature': '()', 'summary': 'Base callback with before_run, after_run, before_iteration, and after_iteration hooks.', 'arguments': []},
    {'callable': 'CallbackList', 'signature': '(callbacks=None)', 'summary': 'Container that chains multiple callbacks.', 'arguments': []},
    {'callable': 'EarlyStopping', 'signature': '(patience=20, min_delta=1e-12, reason="callback_early_stopping")', 'summary': 'Callback-driven early stopping utility.', 'arguments': []},
    {'callable': 'HistoryRecorder', 'signature': '()', 'summary': 'Callback that stores copies of observations after each iteration.', 'arguments': []},
    {'callable': 'ProgressPrinter', 'signature': '(every=1, prefix="")', 'summary': 'Callback that prints progress periodically.', 'arguments': []},
    {'callable': 'cooperative_optimize', 'signature': '(...)', 'summary': 'Run a collaborative / island-based optimization process.', 'arguments': []},
    {'callable': 'replay_cooperative_result', 'signature': '(...)', 'summary': 'Replay a previous collaborative run from its replay manifest.', 'arguments': []},
    {'callable': 'summarize_result', 'signature': '(result)', 'summary': 'Return a flat summary dict for one single-run result.', 'arguments': [{'name': 'result', 'required': True, 'default': None, 'annotation': 'OptimizationResult', 'description': 'Result returned by optimize() or create_optimizer().run().', 'accepted_values': None}]},
    {'callable': 'convergence_data', 'signature': '(result, x_axis="steps")', 'summary': 'Return (x, y) convergence arrays indexed by steps or evaluations.', 'arguments': [{'name': 'x_axis', 'required': False, 'default': '"steps"', 'annotation': 'str', 'description': 'Horizontal axis mode.', 'accepted_values': ['steps', 'iterations', 'evaluations', 'evals']}]},
    {'callable': 'plot_convergence', 'signature': '(result, filepath=None, title=None, show=False, x_axis="steps")', 'summary': 'Plot the convergence curve of one run (matplotlib). Supports step- or evaluation-indexed curves.', 'arguments': []},
    {'callable': 'compare_convergence', 'signature': '(results, labels=None, filepath=None, title="Convergence comparison", show=False, x_axis="steps")', 'summary': 'Overlay multiple convergence curves in one plot (matplotlib).', 'arguments': []},
    {'callable': 'plot_population_snapshot', 'signature': '(...)', 'summary': 'Plot one stored population snapshot.', 'arguments': []},
    {'callable': 'plot_benchmark_summary', 'signature': '(...)', 'summary': 'Bar chart for one scalar metric across runs or algorithms.', 'arguments': []},
    {'callable': 'plot_function', 'signature': '(target_function, min_values=None, max_values=None, ...)', 'summary': 'Auto-select the best landscape plot for the problem dimension. Accepts Problem objects directly.', 'arguments': []},
    {'callable': 'plot_island_dynamics', 'signature': '(...)', 'summary': 'Plot one telemetry metric per island across cooperative global steps.', 'arguments': []},
    {'callable': 'plot_collaboration_network', 'signature': '(...)', 'summary': 'Plot migration traffic between islands as a network.', 'arguments': []},
    {'callable': 'Termination', 'signature': '(max_steps=None, max_evaluations=None, max_time=None, max_early_stop=None, epsilon=1e-10, target_fitness=None)', 'summary': 'Composable stopping-criteria object.', 'arguments': []},
    {'callable': 'plot_global_best_chart', 'signature': '(result, filepath=None, title=None, show=False, color=_ACCENT, x_axis="steps")', 'summary': 'Matplotlib chart of the best-so-far curve over steps or evaluations.', 'arguments': []},
    {'callable': 'plot_diversity_chart', 'signature': '(result, filepath=None, title=None, show=False, color=_GREEN)', 'summary': 'Plot population diversity over steps.', 'arguments': []},
    {'callable': 'plot_explore_exploit_chart', 'signature': '(result, filepath=None, title=None, show=False)', 'summary': 'Stacked area chart of exploration vs exploitation.', 'arguments': []},
    {'callable': 'plot_runtime_chart', 'signature': '(result, filepath=None, title=None, show=False, color=_ORANGE)', 'summary': 'Bar chart of per-step runtime.', 'arguments': []},
    {'callable': 'plot_run_dashboard', 'signature': '(result, filepath=None, title=None, show=False)', 'summary': 'Four-panel run dashboard.', 'arguments': []},
    {'callable': 'FloatVar', 'signature': '(lb, ub, name="float_var")', 'summary': 'Continuous floating-point variable descriptor.', 'arguments': []},
    {'callable': 'IntegerVar', 'signature': '(lb, ub, name="int_var")', 'summary': 'Integer variable descriptor.', 'arguments': []},
    {'callable': 'BinaryVar', 'signature': '(name="binary_var")', 'summary': 'Binary variable descriptor.', 'arguments': []},
    {'callable': 'CategoricalVar', 'signature': '(options, name="cat_var")', 'summary': 'Categorical variable descriptor.', 'arguments': []},
    {'callable': 'PermutationVar', 'signature': '(items, name="perm_var")', 'summary': 'Permutation variable descriptor.', 'arguments': []},
    {'callable': 'build_problem_spec', 'signature': '(target_function, bounds, objective="min", ...)', 'summary': 'Build a ProblemSpec from typed variable descriptors.', 'arguments': []},
    {'callable': 'encode_position', 'signature': '(decoded_values, bounds)', 'summary': 'Encode typed values into the internal continuous representation.', 'arguments': []},
    {'callable': 'decode_position', 'signature': '(position, bounds)', 'summary': 'Decode one continuous position back to typed values.', 'arguments': []},
    {'callable': 'Problem', 'signature': '(dimension, lower, upper, name="problem")', 'summary': 'Abstract N-dimensional problem base class with parametrised bounds and latex_code().', 'arguments': []},
    {'callable': 'FunctionalProblem', 'signature': '(dimension, lower, upper, name="problem", function=..., latex="f(x)")', 'summary': 'Wrap an arbitrary callable as an object-based Problem.', 'arguments': []},
    {'callable': 'SphereProblem', 'signature': '(dimension=2, lower=-5.12, upper=5.12)', 'summary': 'Object-based sphere benchmark with latex_code().', 'arguments': []},
    {'callable': 'RastriginProblem', 'signature': '(dimension=2, lower=-5.12, upper=5.12)', 'summary': 'Object-based Rastrigin benchmark with latex_code().', 'arguments': []},
    {'callable': 'AckleyProblem', 'signature': '(dimension=2, lower=-5.0, upper=5.0)', 'summary': 'Object-based Ackley benchmark with latex_code().', 'arguments': []},
    {'callable': 'RosenbrockProblem', 'signature': '(dimension=2, lower=-5.0, upper=10.0)', 'summary': 'Object-based Rosenbrock benchmark with latex_code().', 'arguments': []},
    {'callable': 'ZakharovProblem', 'signature': '(dimension=2, lower=-5.0, upper=10.0)', 'summary': 'Object-based Zakharov benchmark with latex_code().', 'arguments': []},
    {'callable': 'get_test_problem', 'signature': '(name, dimension=2, lower=None, upper=None)', 'summary': 'Return one curated object-based benchmark problem.', 'arguments': []},
    {'callable': 'list_engineering_benchmarks', 'signature': '()', 'summary': 'Return the constrained engineering benchmark identifiers.', 'arguments': []},
    {'callable': 'get_engineering_benchmark', 'signature': '(name)', 'summary': 'Return objective, constraints, bounds, and best-known metadata for an engineering benchmark.', 'arguments': []},
    {'callable': 'get_test_function_info', 'signature': '(name)', 'summary': 'Return metadata for one test function.', 'arguments': []},
    {'callable': 'validate_known_optima', 'signature': '(tol=1e-7, include_cec=False)', 'summary': 'Validate documented analytic optima, optionally including CEC 2022.', 'arguments': []},
    {'callable': 'validate_engineering_benchmarks', 'signature': '(tol=1e-4, feasibility_tol=1e-4)', 'summary': 'Validate engineering benchmark best-known designs and feasibility.', 'arguments': []},
    {'callable': 'get_cec2022_optimum', 'signature': '(func_num, dimension=10)', 'summary': 'Return the official shifted CEC 2022 optimizer and bias value.', 'arguments': []},
    {'callable': 'validate_cec2022_optima', 'signature': '(tol=1e-7, dimensions=(2,10,20))', 'summary': 'Validate supported CEC 2022 functions at official shifted optima.', 'arguments': []},
    {'callable': 'get_engineering_problem', 'signature': '(name, lower=None, upper=None)', 'summary': 'Return an engineering benchmark as a constrained Problem object.', 'arguments': []},
    {'callable': 'chaotic_sequence', 'signature': '(n, map_name="logistic", seed=0.7, skip=100)', 'summary': 'Generate n values from a named chaotic map. Output is normalised to [0, 1].', 'arguments': []},
    {'callable': 'chaotic_population', 'signature': '(pop_size, dimension, min_values, max_values, map_name="logistic", seed=0.7)', 'summary': 'Generate a population using a chaotic map.', 'arguments': []},
    {'callable': 'ChaoticMap', 'signature': '(static class)', 'summary': 'Ten static chaotic map methods.', 'arguments': []},
    {'callable': 'AVAILABLE_CHAOTIC_MAPS', 'signature': '(list)', 'summary': 'List of all available chaotic map names.', 'arguments': []},
    {'callable': 'apply_transfer', 'signature': '(x, fn_name="v2")', 'summary': 'Apply a named transfer function to a continuous array.', 'arguments': []},
    {'callable': 'binarize', 'signature': '(x, fn_name="v2", rng=None)', 'summary': 'Stochastic binarisation after transfer.', 'arguments': []},
    {'callable': 'BinaryAdapter', 'signature': '(engine, transfer_fn="v2")', 'summary': 'Wrap any continuous engine for binary optimisation.', 'arguments': []},
    {'callable': 'AVAILABLE_TRANSFER_FUNCTIONS', 'signature': '(list)', 'summary': 'List of all transfer function names.', 'arguments': []},
    {'callable': 'limit', 'signature': '(x, lower, upper)', 'summary': 'Clip repair strategy.', 'arguments': []},
    {'callable': 'limit_inverse', 'signature': '(x, lower, upper)', 'summary': 'Wrap-to-opposite-bound repair strategy.', 'arguments': []},
    {'callable': 'wang', 'signature': '(x, lower, upper)', 'summary': 'Reflection-style Wang repair strategy.', 'arguments': []},
    {'callable': 'rand', 'signature': '(x, lower, upper, rng=None)', 'summary': 'Random resampling repair strategy for violated dimensions.', 'arguments': []},
    {'callable': 'reflect', 'signature': '(x, lower, upper)', 'summary': 'Periodic reflection repair strategy.', 'arguments': []},
    {'callable': 'get_repair_function', 'signature': '(name)', 'summary': 'Resolve a named repair strategy to a callable.', 'arguments': []},
    {'callable': 'AVAILABLE_REPAIR_STRATEGIES', 'signature': '(list)', 'summary': 'List of named repair strategies.', 'arguments': []},
    {'callable': 'uniform_population', 'signature': '(pop_size, dimension, min_values, max_values, rng=None)', 'summary': 'Uniform random population initialiser.', 'arguments': []},
    {'callable': 'lhs_population', 'signature': '(pop_size, dimension, min_values, max_values, rng=None)', 'summary': 'Latin hypercube population initialiser.', 'arguments': []},
    {'callable': 'obl_population', 'signature': '(pop_size, dimension, min_values, max_values, rng=None)', 'summary': 'Opposition-based population initialiser.', 'arguments': []},
    {'callable': 'sobol_population', 'signature': '(pop_size, dimension, min_values, max_values, rng=None)', 'summary': 'Sobol quasi-random population initialiser.', 'arguments': []},
    {'callable': 'chaotic_init_function', 'signature': '(map_name="logistic", seed=0.7)', 'summary': 'Return an init_function wrapper backed by chaotic_population().', 'arguments': []},
    {'callable': 'get_init_function', 'signature': '(name)', 'summary': 'Resolve a named init preset to a callable.', 'arguments': []},
    {'callable': 'AVAILABLE_INIT_STRATEGIES', 'signature': '(list)', 'summary': 'List of named initialisation presets.', 'arguments': []},
    {'callable': 'levy_flight', 'signature': '(rng=None, alpha=0.01, beta=1.5, size=None)', 'summary': 'Sample Lévy-flight steps using the Mantegna algorithm.', 'arguments': []},
    {'callable': 'HyperparameterTuner', 'signature': '(algorithm, param_grid, target_function, min_values, max_values, termination=None, n_trials=3, objective="min", seed=42, search="grid", max_configs=20, n_jobs=1)', 'summary': 'Grid or random search over algorithm hyperparameters.', 'arguments': []},
    {'callable': 'BenchmarkRunner', 'signature': '(algorithms, problems, termination=None, n_trials=5, seed=0, n_jobs=1)', 'summary': 'Run multiple algorithms × multiple problems × multiple trials.', 'arguments': []},
    {'callable': 'save_result', 'signature': '(result, path)', 'summary': 'Pickle an OptimizationResult to disk.', 'arguments': []},
    {'callable': 'load_result', 'signature': '(path)', 'summary': 'Restore a pickled OptimizationResult from disk.', 'arguments': []},
    {'callable': 'save_checkpoint', 'signature': '(engine, state, path)', 'summary': 'Pickle a running engine + EngineState for checkpoint-and-resume.', 'arguments': []},
    {'callable': 'load_checkpoint', 'signature': '(path)', 'summary': 'Restore (engine, state) from a checkpoint file.', 'arguments': []},
    {'callable': 'result_to_json', 'signature': '(result, path, indent=2)', 'summary': 'Export a JSON-safe summary of an OptimizationResult.', 'arguments': []},
    {'callable': 'result_from_json', 'signature': '(path)', 'summary': 'Read a JSON file written by result_to_json().', 'arguments': []},
]


def search_reference(name: str):
    key = str(name).strip().lower()
    return [record for record in ARGUMENT_REFERENCE if key in record['callable'].lower()]


def print_root_exports() -> None:
    print('Wrapper-level exports:')
    for name in ROOT_EXPORTS:
        print(f'  - {name}')


def print_reference(name: str | None = None) -> None:
    records = ARGUMENT_REFERENCE if not name else search_reference(name)
    if not records:
        print(f'No wrapper entries found for: {name}')
        return
    for record in records:
        print(f"\n{record['callable']}{record['signature']}")
        print(f"  {record['summary']}")
        if not record['arguments']:
            print('  arguments: none / see source docstring')
            continue
        for arg in record['arguments']:
            req = 'required' if arg['required'] else 'optional'
            ann = f" : {arg['annotation']}" if arg.get('annotation') else ''
            default = f" = {arg['default']}" if arg.get('default') is not None else ''
            print(f"  - {arg['name']}{ann}{default} [{req}]")
            print(f"      {arg['description']}")
            if arg.get('accepted_values') is not None:
                print(f"      accepted: {arg['accepted_values']}")


if __name__ == '__main__':
    print('pymetaheuristic wrapper argument reference')
    print_root_exports()
