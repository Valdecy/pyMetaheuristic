"""Public package for pymetaheuristic."""

from .src.api import create_optimizer, get_algorithm_info, list_algorithms, optimize
from .src.callbacks import Callback, CallbackList, EarlyStopping, HistoryRecorder, ProgressPrinter
from .src.cooperation import cooperative_optimize, replay_cooperative_result
from .src.graphs import (
    compare_convergence,
    plot_benchmark_summary,
    plot_benchmark_barplots,
    plot_benchmark_boxplots,
    plot_benchmark_rank_heatmap,
    plot_benchmark_runtime,
    plot_benchmark_convergence,
    plot_collaboration_network,
    plot_convergence,
    plot_function,
    plot_function_1d,
    plot_function_2d,
    plot_function_3d,
    plot_function_contour,
    plot_function_nd,
    plot_function_surface,
    plot_island_dynamics,
    plot_population_snapshot,
)
from .src.orchestration import orchestrated_optimize
from .src.reference import (
    ARGUMENT_REFERENCE,
    search_reference,
    print_reference,
    print_root_exports,
)
from .src.schemas import (
    ActionOutcome,
    ActionSpec,
    AgentSnapshot,
    CollaborativeConfig,
    DecisionPlan,
    OrchestratedCooperativeResult,
    OrchestrationSpec,
    OrchestratorSnapshot,
    RulesConfig,
)
from .src.test_functions import FUNCTIONS, get_test_function
from .src.actions import estimate_action_cost
from .src.telemetry import (
    convergence_data,
    export_history_csv,
    export_island_telemetry_csv,
    export_population_snapshots_json,
    export_replay_manifest_json,
    summarize_cooperative_result,
    summarize_result,
)
from .src.termination import Termination
from .src.viz import (
    plot_diversity_chart,
    plot_diversity_comparison,
    plot_explore_exploit_chart,
    plot_global_best_chart,
    plot_run_dashboard,
    plot_runtime_chart,
)
from .src.utils import (
    AVAILABLE_CHAOTIC_MAPS,
    AVAILABLE_INIT_STRATEGIES,
    AVAILABLE_REPAIR_STRATEGIES,
    AVAILABLE_TRANSFER_FUNCTIONS,
    AckleyProblem,
    BinaryAdapter,
    BinaryVar,
    CategoricalVar,
    ChaoticMap,
    FloatVar,
    FunctionalProblem,
    IntegerVar,
    PermutationVar,
    Problem,
    RastriginProblem,
    RosenbrockProblem,
    SphereProblem,
    ZakharovProblem,
    apply_transfer,
    binarize,
    build_problem_spec,
    chaotic_init_function,
    chaotic_population,
    chaotic_sequence,
    decode_position,
    encode_position,
    full_array,
    get_init_function,
    get_repair_function,
    get_test_problem,
    lhs_population,
    levy_flight,
    limit,
    limit_inverse,
    obl_population,
    rand,
    reflect,
    sobol_population,
    sstf_01, sstf_02, sstf_03, sstf_04,
    uniform_population,
    vstf_01, vstf_02, vstf_03, vstf_04,
    wang,
)
from .src.tuner import BenchmarkRunner, HyperparameterTuner
from .src.io import (
    load_checkpoint,
    load_result,
    result_from_json,
    result_to_json,
    save_checkpoint,
    save_result,
)

__version__ = "5.7.2"

__all__ = [
    "FUNCTIONS",
    "ActionOutcome", "ActionSpec", "AgentSnapshot",
    "ARGUMENT_REFERENCE",
    "Callback", "CallbackList", "EarlyStopping", "HistoryRecorder", "ProgressPrinter",
    "CollaborativeConfig", "DecisionPlan",
    "OrchestratedCooperativeResult", "OrchestrationSpec",
    "OrchestratorSnapshot", "RulesConfig",
    "compare_convergence", "cooperative_optimize", "replay_cooperative_result",
    "convergence_data",
    "create_optimizer", "estimate_action_cost",
    "export_history_csv", "export_island_telemetry_csv",
    "export_population_snapshots_json", "export_replay_manifest_json",
    "get_algorithm_info", "get_test_function", "get_test_problem",
    "list_algorithms", "optimize", "orchestrated_optimize",
    "plot_benchmark_summary", "plot_benchmark_barplots", "plot_benchmark_boxplots",
    "plot_benchmark_rank_heatmap", "plot_benchmark_runtime", "plot_benchmark_convergence",
    "plot_collaboration_network",
    "plot_convergence", "plot_function",
    "plot_function_1d", "plot_function_2d", "plot_function_3d",
    "plot_function_contour", "plot_function_nd", "plot_function_surface",
    "plot_island_dynamics", "plot_population_snapshot",
    "print_reference", "print_root_exports", "search_reference",
    "summarize_cooperative_result", "summarize_result",
    "Termination",
    "plot_diversity_chart", "plot_diversity_comparison",
    "plot_explore_exploit_chart", "plot_global_best_chart",
    "plot_run_dashboard", "plot_runtime_chart",
    "BinaryVar", "CategoricalVar", "FloatVar", "IntegerVar", "PermutationVar",
    "build_problem_spec", "decode_position", "encode_position",
    "AVAILABLE_CHAOTIC_MAPS", "ChaoticMap", "chaotic_population", "chaotic_sequence",
    "AVAILABLE_TRANSFER_FUNCTIONS", "BinaryAdapter", "apply_transfer", "binarize",
    "vstf_01", "vstf_02", "vstf_03", "vstf_04",
    "sstf_01", "sstf_02", "sstf_03", "sstf_04",
    "AVAILABLE_REPAIR_STRATEGIES", "get_repair_function", "limit", "limit_inverse",
    "rand", "reflect", "wang",
    "AVAILABLE_INIT_STRATEGIES", "get_init_function", "uniform_population",
    "lhs_population", "obl_population", "sobol_population", "chaotic_init_function",
    "Problem", "FunctionalProblem", "SphereProblem", "RastriginProblem",
    "AckleyProblem", "RosenbrockProblem", "ZakharovProblem", "full_array",
    "levy_flight",
    "HyperparameterTuner", "BenchmarkRunner",
    "load_checkpoint", "load_result",
    "result_from_json", "result_to_json",
    "save_checkpoint", "save_result",
]

from . import src  # noqa: E402
examples = src.examples
reference = src.reference
