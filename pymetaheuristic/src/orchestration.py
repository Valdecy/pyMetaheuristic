from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .api import create_optimizer
from .cooperation import CooperativeRunner, IslandSpec
from .actions import execute_decision_plan, outcome_to_dict
from .controllers import FixedMigrationController, RuleBasedController
from .schemas import (
    AgentSnapshot,
    CollaborativeConfig,
    DecisionPlan,
    OrchestratedCooperativeResult,
    OrchestratorSnapshot,
)
from .execution import run_engine_chunks


def _cfg_to_dataclass(config):
    if config is None:
        return CollaborativeConfig()
    if isinstance(config, CollaborativeConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError("config must be None, CollaborativeConfig, or dict")
    cfg = CollaborativeConfig()
    for section_name in ("orchestration", "rules"):
        section = config.get(section_name)
        if isinstance(section, dict):
            obj = getattr(cfg, section_name)
            for k, v in section.items():
                setattr(obj, k, v)
    return cfg


def _health_from_observation(obs: dict[str, Any], has_population: bool) -> str:
    stagnation = obs.get("stagnation_steps")
    diversity = obs.get("diversity")
    temperature = obs.get("temperature")
    if has_population:
        if diversity is not None and diversity <= 0.10 and (stagnation or 0) >= 5:
            return "poor"
        if diversity is not None and diversity > 0.10:
            return "healthy"
        return "stable"
    if temperature is not None and temperature <= 1e-3 and (stagnation or 0) >= 5:
        return "frozen"
    if (stagnation or 0) >= 5:
        return "stagnating"
    return "healthy"


def build_snapshot(
    engines: dict[str, Any],
    states: dict[str, Any],
    previous_best: dict[str, float | None],
    stagnation_counter: dict[str, int],
    recent_history: dict[str, list[dict[str, Any]]],
    checkpoint_id: int,
    objective: str,
    dimension: int,
    budget_total: int | None,
    recent_actions: list[dict[str, Any]] | None = None,
    memory: list[str] | None = None,
) -> OrchestratorSnapshot:
    agents: list[AgentSnapshot] = []
    budget_used = 0
    best_label = None
    best_fit = None
    best_pos = None

    for label, engine in engines.items():
        state = states[label]
        obs = dict(engine.observe(state))
        best_fitness = obs.get("best_fitness", state.best_fitness)
        previous = previous_best.get(label)
        delta = None if previous is None or best_fitness is None else float(best_fitness - previous)
        if previous is None or best_fitness is None:
            stagnation_counter[label] = 0
        else:
            improved = engine.problem.is_better(best_fitness, previous)
            stagnation_counter[label] = 0 if improved else stagnation_counter.get(label, 0) + 1
        previous_best[label] = best_fitness
        obs["stagnation_steps"] = stagnation_counter[label]
        obs["health"] = _health_from_observation(obs, engine.capabilities.has_population)
        recent_history.setdefault(label, []).append(obs)
        recent_history[label] = recent_history[label][-8:]

        current_position = None
        if "current" in state.payload:
            current_position = list(state.payload.get("current"))
        current_fitness = obs.get("current_fitness", state.payload.get("current_fit"))
        params_view = {
            k: v
            for k, v in obs.items()
            if k not in {
                "step", "evaluations", "best_fitness", "mean_fitness", "std_fitness",
                "diversity", "current_fitness", "stagnation_steps", "health",
            }
        }

        agent = AgentSnapshot(
            label=label,
            algorithm=engine.algorithm_id,
            family=engine.family,
            has_population=engine.capabilities.has_population,
            supports_injection=engine.capabilities.supports_candidate_injection,
            supports_restart=engine.capabilities.supports_restart,
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            best_position=list(state.best_position) if state.best_position is not None else None,
            current_fitness=current_fitness,
            current_position=current_position,
            delta_best=delta,
            stagnation_steps=stagnation_counter[label],
            diversity=obs.get("diversity"),
            mean_fitness=obs.get("mean_fitness"),
            std_fitness=obs.get("std_fitness"),
            health=obs.get("health"),
            recent_history=list(recent_history[label]),
            params_view=params_view,
            raw_observation=obs,
        )
        agents.append(agent)
        budget_used += state.evaluations
        if state.best_fitness is not None and (
            best_fit is None
            or (state.best_fitness > best_fit if objective == "max" else state.best_fitness < best_fit)
        ):
            best_label, best_fit, best_pos = label, state.best_fitness, list(state.best_position)

    budget_remaining = None if budget_total is None else max(0, int(budget_total - budget_used))
    budget_used_ratio = None if budget_total is None or budget_total <= 0 else float(budget_used / budget_total)
    return OrchestratorSnapshot(
        checkpoint_id=checkpoint_id,
        objective=objective,
        dimension=dimension,
        budget_total=budget_total,
        budget_used=budget_used,
        budget_remaining=budget_remaining,
        budget_used_ratio=budget_used_ratio,
        global_best_label=best_label,
        global_best_fitness=best_fit,
        global_best_position=best_pos,
        agents=agents,
        recent_actions=list(recent_actions or []),
        memory=list(memory or []),
        metadata={"n_agents": len(agents)},
    )


class OrchestratedRunner:
    def __init__(
        self,
        islands: list[IslandSpec | dict[str, Any]],
        target_function,
        min_values,
        max_values,
        objective: str = "min",
        constraints=None,
        constraint_handler=None,
        repair_function=None,
        penalty_coefficient: float = 1e6,
        equality_tolerance: float = 1e-6,
        resample_attempts: int = 25,
        max_steps: int = 100,
        max_evaluations: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
        config: CollaborativeConfig | dict[str, Any] | None = None,
        execution_backend: str = "serial",
        n_jobs: int | None = None,
        parallel_fallback_to_serial: bool = True,
    ) -> None:
        self.island_specs = [s if isinstance(s, IslandSpec) else IslandSpec(**s) for s in islands]
        self.target_function = target_function
        self.min_values = list(min_values)
        self.max_values = list(max_values)
        self.objective = objective
        self.constraints = constraints
        self.constraint_handler = constraint_handler
        self.repair_function = repair_function
        self.penalty_coefficient = penalty_coefficient
        self.equality_tolerance = equality_tolerance
        self.resample_attempts = resample_attempts
        self.max_steps = int(max_steps)
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.verbose = verbose
        self.config = _cfg_to_dataclass(config)
        self.execution_backend = execution_backend
        self.n_jobs = n_jobs
        self.parallel_fallback_to_serial = parallel_fallback_to_serial
        if self.config.orchestration.mode not in {"fixed", "rules"}:
            raise ValueError("orchestration.mode must be 'fixed' or 'rules'.")

    def _build_controller(self):
        mode = self.config.orchestration.mode
        if mode == "fixed":
            return FixedMigrationController()
        if mode == "rules":
            return RuleBasedController(self.config.orchestration, self.config.rules)
        raise ValueError(f"Unsupported orchestration mode: {mode}")

    def run(self) -> OrchestratedCooperativeResult:
        engines: dict[str, Any] = {}
        states: dict[str, Any] = {}
        histories: dict[str, list[dict[str, Any]]] = {}
        previous_best: dict[str, float | None] = {}
        stagnation_counter: dict[str, int] = {}
        recent_history: dict[str, list[dict[str, Any]]] = {}
        checkpoints = []
        decisions = []
        outcomes_all = []
        events = []
        memory: list[str] = []

        for i, spec in enumerate(self.island_specs):
            label = spec.label or f"{spec.algorithm}_{i + 1}"
            engine = create_optimizer(
                algorithm=spec.algorithm,
                target_function=self.target_function,
                min_values=self.min_values,
                max_values=self.max_values,
                objective=self.objective,
                constraints=self.constraints,
                constraint_handler=self.constraint_handler,
                repair_function=self.repair_function,
                penalty_coefficient=self.penalty_coefficient,
                equality_tolerance=self.equality_tolerance,
                resample_attempts=self.resample_attempts,
                max_steps=self.max_steps,
                max_evaluations=self.max_evaluations,
                seed=(spec.seed if spec.seed is not None else (None if self.seed is None else self.seed + i)),
                verbose=False,
                store_history=False,
                config=spec.config,
            )
            state = engine.initialize()
            engines[label] = engine
            states[label] = state
            histories[label] = []
            previous_best[label] = state.best_fitness
            stagnation_counter[label] = 0
            recent_history[label] = []

        controller = self._build_controller()
        checkpoint_interval = max(1, int(self.config.orchestration.checkpoint_interval))
        checkpoint_id = 0
        rounds_since_checkpoint = 0
        recent_action_dicts: list[dict[str, Any]] = []
        init_snapshot = build_snapshot(
            engines=engines,
            states=states,
            previous_best=previous_best,
            stagnation_counter=stagnation_counter,
            recent_history=recent_history,
            checkpoint_id=checkpoint_id,
            objective=self.objective,
            dimension=len(self.min_values),
            budget_total=self.max_evaluations,
            recent_actions=recent_action_dicts,
            memory=memory,
        )
        controller.initialize(init_snapshot)
        checkpoints.append(init_snapshot)

        active = True
        global_history = []
        actual_backend = (self.execution_backend or "serial").lower()
        backend_warning = None
        label_order = list(engines.keys())

        while active:
            chunk_items = []
            active_labels = []
            for label in label_order:
                engine = engines[label]
                state = states[label]
                if engine.should_stop(state):
                    continue
                active_labels.append(label)
                chunk_items.append((label, engine, state, 1))
            active = bool(chunk_items)
            if not active:
                break

            results, actual_backend, warning = run_engine_chunks(
                chunk_items,
                execution_backend=self.execution_backend,
                n_jobs=self.n_jobs,
                fallback_to_serial=self.parallel_fallback_to_serial,
            )
            if warning and backend_warning is None:
                backend_warning = warning
                if self.verbose:
                    print(f"[orchestrated] {warning}")

            for label, result in zip(active_labels, results):
                engine = engines[label]
                states[label] = result.state
                for obs in result.observations:
                    histories[label].append(obs)
                    global_history.append({"label": label, "algorithm": engine.algorithm_id, **obs})
            rounds_since_checkpoint += 1
            if rounds_since_checkpoint < checkpoint_interval:
                continue

            checkpoint_id += 1
            rounds_since_checkpoint = 0
            snapshot = build_snapshot(
                engines=engines,
                states=states,
                previous_best=previous_best,
                stagnation_counter=stagnation_counter,
                recent_history=recent_history,
                checkpoint_id=checkpoint_id,
                objective=self.objective,
                dimension=len(self.min_values),
                budget_total=self.max_evaluations,
                recent_actions=recent_action_dicts,
                memory=memory,
            )
            checkpoints.append(snapshot)

            plan: DecisionPlan = controller.decide(snapshot)
            decisions.append(plan)
            states, outcomes = execute_decision_plan(
                plan,
                engines,
                states,
                objective=self.objective,
                seed=None if self.seed is None else self.seed + checkpoint_id,
            )
            outcomes_all.append(outcomes)
            recent_action_dicts = [outcome_to_dict(o) for o in outcomes]
            events.extend(recent_action_dicts)
            if plan.next_checkpoint_interval is not None:
                checkpoint_interval = max(
                    self.config.orchestration.min_checkpoint_interval,
                    min(self.config.orchestration.max_checkpoint_interval, int(plan.next_checkpoint_interval)),
                )
            if self.verbose:
                print(f"[orchestrated] checkpoint={checkpoint_id} actions={len(plan.actions)} best={snapshot.global_best_fitness}")

        island_results = {}
        best_position = None
        best_fitness = None
        for label, engine in engines.items():
            result = engine.finalize(states[label])
            result.history = histories[label]
            island_results[label] = result
            if best_fitness is None or engine.problem.is_better(result.best_fitness, best_fitness):
                best_fitness = result.best_fitness
                best_position = list(result.best_position)

        return OrchestratedCooperativeResult(
            best_position=best_position,
            best_fitness=best_fitness,
            island_results=island_results,
            history=global_history,
            events=events,
            controller_mode=self.config.orchestration.mode,
            checkpoints=checkpoints if self.config.orchestration.store_snapshots else [],
            decisions=decisions if self.config.orchestration.store_decisions else [],
            outcomes=outcomes_all,
            metadata={
                "checkpoint_interval": self.config.orchestration.checkpoint_interval,
                "constraint_handler": self.constraint_handler or "none",
                "islands": list(engines.keys()),
                "execution_backend_requested": self.execution_backend,
                "execution_backend_used": actual_backend,
                "n_jobs": self.n_jobs,
                "parallel_warning": backend_warning,
            },
        )


def orchestrated_optimize(*args, **kwargs) -> OrchestratedCooperativeResult:
    banned = {
        "llm_client",
        "llms",
        "llm_fusion_strategy",
        "llm_trigger",
        "llm_safe_fallback",
        "llm_confidence_threshold",
        "llm_enabled",
    }
    used_banned = sorted(k for k in kwargs if k in banned)
    if used_banned:
        joined = ", ".join(used_banned)
        raise TypeError(f"Unsupported arguments after LLM purge: {joined}")

    config = kwargs.get("config")
    cfg = _cfg_to_dataclass(config)
    kwargs["config"] = cfg
    if cfg.orchestration.mode == "fixed":
        runner = CooperativeRunner(
            islands=kwargs["islands"] if "islands" in kwargs else args[0],
            target_function=kwargs["target_function"] if "target_function" in kwargs else args[1],
            min_values=kwargs["min_values"] if "min_values" in kwargs else args[2],
            max_values=kwargs["max_values"] if "max_values" in kwargs else args[3],
            objective=kwargs.get("objective", "min"),
            constraints=kwargs.get("constraints"),
            constraint_handler=kwargs.get("constraint_handler"),
            repair_function=kwargs.get("repair_function"),
            penalty_coefficient=kwargs.get("penalty_coefficient", 1e6),
            equality_tolerance=kwargs.get("equality_tolerance", 1e-6),
            resample_attempts=kwargs.get("resample_attempts", 25),
            max_steps=kwargs.get("max_steps", 100),
            migration_interval=cfg.orchestration.checkpoint_interval,
            migration_size=1,
            migration_mode="elite",
            topology="ring",
            seed=kwargs.get("seed"),
            verbose=kwargs.get("verbose", False),
            execution_backend=kwargs.get("execution_backend", "serial"),
            n_jobs=kwargs.get("n_jobs"),
            parallel_fallback_to_serial=kwargs.get("parallel_fallback_to_serial", True),
        )
        legacy = runner.run()
        return OrchestratedCooperativeResult(
            best_position=legacy.best_position,
            best_fitness=legacy.best_fitness,
            island_results=legacy.island_results,
            history=legacy.history,
            events=[asdict(e) if is_dataclass(e) else dict(e) for e in legacy.events],
            controller_mode="fixed",
            checkpoints=[],
            decisions=[],
            outcomes=[],
            metadata=dict(legacy.metadata),
        )
    return OrchestratedRunner(*args, **kwargs).run()
