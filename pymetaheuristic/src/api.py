from __future__ import annotations

from typing import Any

from .engines import REGISTRY, EngineConfig, ProblemSpec
from .utils import Problem, get_init_function, get_repair_function


def list_algorithms() -> list[str]:
    return sorted(REGISTRY.keys())


def get_algorithm_info(algorithm: str) -> dict[str, Any]:
    if algorithm not in REGISTRY:
        raise KeyError(f"Unknown algorithm: {algorithm}. Available: {', '.join(list_algorithms())}")
    cls = REGISTRY[algorithm]
    defaults = getattr(cls, "_DEFAULTS", {})
    capabilities = cls.capabilities
    if capabilities.supports_native_constraints:
        constraint_support = "native"
    elif capabilities.supports_framework_constraints:
        constraint_support = "framework"
    else:
        constraint_support = "none"
    return {
        "algorithm_id": cls.algorithm_id,
        "algorithm_name": cls.algorithm_name,
        "family": cls.family,
        "capabilities": capabilities,
        "constraint_support": constraint_support,
        "defaults": dict(defaults),
        "doi": (getattr(cls, "_REFERENCE", {}) or {}).get("doi"),
        "info": cls.info(),
    }


def _resolve_problem(
    target_function,
    min_values,
    max_values,
    objective: str,
    constraints,
    constraint_handler,
    variable_types,
    metadata: dict[str, Any],
    problem=None,
) -> ProblemSpec:
    problem_obj = problem
    if problem_obj is None and isinstance(target_function, Problem):
        problem_obj = target_function
    if problem_obj is not None:
        if isinstance(problem_obj, ProblemSpec):
            problem_obj.metadata.update(metadata)
            if constraints is not None:
                problem_obj.constraints = constraints
            if constraint_handler is not None:
                problem_obj.constraint_handler = constraint_handler
            if variable_types is not None:
                problem_obj.variable_types = variable_types
            return problem_obj
        if isinstance(problem_obj, Problem):
            return problem_obj.to_problem_spec(
                objective=objective,
                constraints=constraints,
                constraint_handler=constraint_handler,
                variable_types=variable_types,
                metadata=metadata,
            )
        if hasattr(problem_obj, "to_problem_spec"):
            return problem_obj.to_problem_spec(
                objective=objective,
                constraints=constraints,
                constraint_handler=constraint_handler,
                variable_types=variable_types,
                metadata=metadata,
            )
        raise TypeError("problem must be a ProblemSpec, Problem, or implement to_problem_spec().")

    if min_values is None or max_values is None:
        raise ValueError("min_values and max_values are required unless target_function/problem is a Problem object.")

    return ProblemSpec(
        target_function=target_function,
        min_values=list(min_values),
        max_values=list(max_values),
        objective=objective,
        constraints=constraints,
        constraint_handler=constraint_handler,
        variable_types=variable_types,
        metadata=metadata,
    )


def create_optimizer(
    algorithm: str,
    target_function=None,
    min_values=None,
    max_values=None,
    objective: str = "min",
    constraints=None,
    constraint_handler=None,
    variable_types=None,
    repair_function=None,
    repair_name: str | None = None,
    callbacks=None,
    init_function=None,
    init_name: str | None = None,
    problem=None,
    penalty_coefficient: float = 1e6,
    equality_tolerance: float = 1e-6,
    resample_attempts: int = 25,
    max_steps: int | None = None,
    max_evaluations: int | None = None,
    target_fitness: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
    store_history: bool = True,
    store_population_snapshots: bool = False,
    snapshot_interval: int = 1,
    timeout_seconds: float | None = None,
    config: dict[str, Any] | None = None,
    termination=None,
    **params,
):
    if algorithm not in REGISTRY:
        raise KeyError(f"Unknown algorithm: {algorithm}. Available: {', '.join(list_algorithms())}")
    merged_params = {}
    if config:
        merged_params.update(config)
    merged_params.update(params)

    if repair_name is not None:
        repair_function = get_repair_function(repair_name)
    if init_name is not None and init_function is None:
        init_function = get_init_function(init_name)

    _termination_obj = None
    if termination is not None:
        from .termination import Termination
        _termination_obj = Termination._from_any(termination)
        if _termination_obj.max_steps is not None:
            max_steps = _termination_obj.max_steps
        if _termination_obj.max_evaluations is not None:
            max_evaluations = _termination_obj.max_evaluations
        if _termination_obj.target_fitness is not None:
            target_fitness = _termination_obj.target_fitness
        if _termination_obj.max_time is not None:
            timeout_seconds = _termination_obj.max_time

    metadata = {
        "repair_function": repair_function,
        "repair_name": repair_name,
        "use_raw_repair_input": repair_name is not None,
        "penalty_coefficient": penalty_coefficient,
        "equality_tolerance": equality_tolerance,
        "resample_attempts": resample_attempts,
    }

    problem_spec = _resolve_problem(
        target_function=target_function,
        min_values=min_values,
        max_values=max_values,
        objective=objective,
        constraints=constraints,
        constraint_handler=constraint_handler,
        variable_types=variable_types,
        metadata=metadata,
        problem=problem,
    )

    engine_config = EngineConfig(
        max_steps=max_steps,
        max_evaluations=max_evaluations,
        target_fitness=target_fitness,
        seed=seed,
        verbose=verbose,
        store_history=store_history,
        store_population_snapshots=store_population_snapshots,
        snapshot_interval=snapshot_interval,
        timeout_seconds=timeout_seconds,
        params=merged_params,
        callbacks=callbacks,
        init_function=init_function,
    )
    if _termination_obj is not None:
        engine_config._termination_obj = _termination_obj  # type: ignore[attr-defined]

    return REGISTRY[algorithm](problem_spec, engine_config)


def optimize(*args, **kwargs):
    return create_optimizer(*args, **kwargs).run()
