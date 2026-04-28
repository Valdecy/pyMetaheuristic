"""pyMetaheuristic src.utils — utility modules."""

from .chaotic import (
    ChaoticMap,
    AVAILABLE_CHAOTIC_MAPS,
    AVAILABLE_TRANSFER_FUNCTIONS,
    BinaryAdapter,
    apply_transfer,
    binarize,
    chaotic_population,
    chaotic_sequence,
    sstf_01, sstf_02, sstf_03, sstf_04,
    vstf_01, vstf_02, vstf_03, vstf_04,
)
from .initialization import (
    AVAILABLE_INIT_STRATEGIES,
    chaotic_init_function,
    get_init_function,
    lhs_population,
    obl_population,
    sobol_population,
    uniform_population,
)
from .problems import (
    Problem,
    FunctionalProblem,
    ConstrainedFunctionalProblem,
    SphereProblem,
    RastriginProblem,
    AckleyProblem,
    RosenbrockProblem,
    ZakharovProblem,
    full_array,
    get_test_problem,
    list_test_problems,
    list_engineering_problems,
    get_engineering_problem,
    get_engineering_problem_spec,
)
from .random import levy_flight
from .repair import (
    AVAILABLE_REPAIR_STRATEGIES,
    get_repair_function,
    limit,
    limit_inverse,
    rand,
    reflect,
    wang,
)
from .space import (
    BaseVar,
    BinaryVar,
    CategoricalVar,
    FloatVar,
    IntegerVar,
    PermutationVar,
    build_problem_spec,
    decode_position,
    encode_position,
)

__all__ = [
    # chaotic
    "ChaoticMap", "AVAILABLE_CHAOTIC_MAPS", "AVAILABLE_TRANSFER_FUNCTIONS",
    "BinaryAdapter", "apply_transfer", "binarize",
    "chaotic_population", "chaotic_sequence",
    "vstf_01", "vstf_02", "vstf_03", "vstf_04",
    "sstf_01", "sstf_02", "sstf_03", "sstf_04",
    # init
    "AVAILABLE_INIT_STRATEGIES", "uniform_population", "lhs_population",
    "obl_population", "sobol_population", "chaotic_init_function",
    "get_init_function",
    # repair
    "AVAILABLE_REPAIR_STRATEGIES", "limit", "limit_inverse", "wang",
    "rand", "reflect", "get_repair_function",
    # random
    "levy_flight",
    # problems
    "Problem", "FunctionalProblem", "ConstrainedFunctionalProblem",
    "SphereProblem", "RastriginProblem",
    "AckleyProblem", "RosenbrockProblem", "ZakharovProblem",
    "full_array", "get_test_problem", "list_test_problems",
    "list_engineering_problems", "get_engineering_problem", "get_engineering_problem_spec",
    # space
    "BaseVar", "BinaryVar", "CategoricalVar", "FloatVar",
    "IntegerVar", "PermutationVar",
    "build_problem_spec", "decode_position", "encode_position",
]
