from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Any

from .api import create_optimizer
from .budget import (
    EvaluationBudgetExceeded,
    FairEvaluationScheduler,
    format_budget_report,
    make_shared_budget,
)
from .cooperation import CooperativeRunner, IslandSpec
from .actions import execute_decision_plan, outcome_to_dict
from .controllers import BanditController, FixedMigrationController, PortfolioAdaptiveController, RuleBasedController
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
    for section_name in ("orchestration", "rules", "bandit", "portfolio"):
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
    budget_used_override: int | None = None,
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

    if budget_used_override is not None:
        budget_used = int(budget_used_override)
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
        variable_types=None,
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
        self.variable_types = None if variable_types is None else list(variable_types)
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
        if self.config.orchestration.mode not in {"fixed", "rules", "bandit", "portfolio_adaptive"}:
            raise ValueError("orchestration.mode must be 'fixed', 'rules', 'bandit', or 'portfolio_adaptive'.")

    def _build_controller(self):
        mode = self.config.orchestration.mode
        if mode == "fixed":
            return FixedMigrationController()
        if mode == "rules":
            return RuleBasedController(self.config.orchestration, self.config.rules)
        if mode == "bandit":
            return BanditController(self.config.orchestration, self.config.rules, self.config.bandit)
        if mode == "portfolio_adaptive":
            return PortfolioAdaptiveController(self.config.orchestration, self.config.rules, self.config.portfolio)
        raise ValueError(f"Unsupported orchestration mode: {mode}")

    def run(self) -> OrchestratedCooperativeResult:
        budget = make_shared_budget(self.target_function, self.max_evaluations, objective=self.objective)
        target_function = budget if budget is not None else self.target_function
        force_serial_budget = budget is not None and (self.execution_backend or "serial").lower() != "serial"
        execution_backend = "serial" if force_serial_budget else self.execution_backend
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
        partial_budget_labels: set[str] = set()
        initialization_budget_exhausted = False

        for i, spec in enumerate(self.island_specs):
            label = spec.label or f"{spec.algorithm}_{i + 1}"
            engine = create_optimizer(
                algorithm=spec.algorithm,
                target_function=target_function,
                min_values=self.min_values,
                max_values=self.max_values,
                objective=self.objective,
                constraints=self.constraints,
                constraint_handler=self.constraint_handler,
                variable_types=self.variable_types,
                repair_function=self.repair_function,
                penalty_coefficient=self.penalty_coefficient,
                equality_tolerance=self.equality_tolerance,
                resample_attempts=self.resample_attempts,
                max_steps=self.max_steps,
                max_evaluations=None,
                seed=(spec.seed if spec.seed is not None else (None if self.seed is None else self.seed + i)),
                verbose=False,
                store_history=False,
                config=spec.config,
            )
            try:
                if budget is not None:
                    with budget.use_label(label), budget.use_category("initialization"):
                        state = engine.initialize()
                else:
                    state = engine.initialize()
            except EvaluationBudgetExceeded:
                details = engine._best_available_evaluation_details(None, label)
                if details is None:
                    initialization_budget_exhausted = True
                    break
                used = 0 if budget is None else int(budget.used_for(label))
                state = engine._prepare_budget_exhausted_state(
                    None,
                    label=label,
                    completed_steps=0,
                    phase="initialization",
                    phase_evaluations=used,
                )
                partial_budget_labels.add(label)
                initialization_budget_exhausted = True
            if budget is not None:
                engine._sync_evaluation_count(state, label)
                engine._remember_best_evaluation_details(state)
            engines[label] = engine
            states[label] = state
            histories[label] = []
            previous_best[label] = state.best_fitness
            stagnation_counter[label] = 0
            recent_history[label] = []
            if initialization_budget_exhausted:
                break

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
            budget_used_override=(int(budget.used) if budget is not None else None),
        )
        controller.initialize(init_snapshot)
        checkpoints.append(init_snapshot)

        global_history = []
        actual_backend = (execution_backend or "serial").lower()
        backend_warning = None
        if force_serial_budget:
            backend_warning = "Strict shared max_evaluations requires serial island execution; process backend was disabled to avoid FE overshoot."
        label_order = list(engines.keys())
        scheduler = FairEvaluationScheduler(label_order) if budget is not None and label_order else None
        scheduler_turns = 0
        scheduler_turns_since_checkpoint = 0

        while True:
            if budget is not None and budget.exhausted:
                break

            active_labels = [
                label
                for label in label_order
                if not engines[label].should_stop(states[label])
            ]
            if not active_labels:
                break

            evaluations_before_by_label: dict[str, int] = {}
            if budget is not None and scheduler is not None:
                selected_label = scheduler.select(active_labels, budget.by_label)
                chunk_labels = [selected_label]
                evaluations_before_by_label[selected_label] = int(budget.used_for(selected_label))
            else:
                chunk_labels = list(active_labels)
            chunk_items = [
                (label, engines[label], states[label], 1)
                for label in chunk_labels
            ]

            results, actual_backend, warning = run_engine_chunks(
                chunk_items,
                execution_backend=execution_backend,
                n_jobs=self.n_jobs,
                fallback_to_serial=self.parallel_fallback_to_serial,
            )
            if warning and backend_warning is None:
                backend_warning = warning
                if self.verbose:
                    print(f"[orchestrated] {warning}")

            for label, result in zip(chunk_labels, results):
                engine = engines[label]
                states[label] = result.state
                if budget is not None:
                    local_after = int(budget.used_for(label))
                    local_before = int(evaluations_before_by_label.get(label, local_after))
                    consumed = max(0, local_after - local_before)
                    global_evaluations = int(budget.used)
                    island_evaluations = local_after
                else:
                    consumed = 0
                    global_evaluations = sum(int(states[l].evaluations) for l in label_order)
                    island_evaluations = int(states[label].evaluations)

                global_best_fitness = None
                for other_label in label_order:
                    fit = states[other_label].best_fitness
                    if fit is None:
                        continue
                    if global_best_fitness is None or (
                        fit > global_best_fitness if self.objective == "max" else fit < global_best_fitness
                    ):
                        global_best_fitness = fit

                observations = list(result.observations)
                if not observations and consumed > 0:
                    fallback_obs = dict(engine.observe(states[label]))
                    fallback_obs["event"] = "budget_cutoff"
                    observations = [fallback_obs]
                for obs in observations:
                    history_item = {
                        **dict(obs),
                        "label": label,
                        "algorithm": engine.algorithm_id,
                        "global_evaluations": global_evaluations,
                        "island_evaluations": island_evaluations,
                        "evaluations_consumed": consumed,
                        "global_best_fitness": global_best_fitness,
                    }
                    histories[label].append(history_item)
                    global_history.append(history_item)

            if budget is not None:
                scheduler_turns += 1
                scheduler_turns_since_checkpoint += 1
                active_count = max(1, len(active_labels))
                checkpoint_due = scheduler_turns_since_checkpoint >= checkpoint_interval * active_count
                if not checkpoint_due:
                    continue
                scheduler_turns_since_checkpoint = 0
            else:
                rounds_since_checkpoint += 1
                if rounds_since_checkpoint < checkpoint_interval:
                    continue
                rounds_since_checkpoint = 0

            checkpoint_id += 1
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
                budget_used_override=(int(budget.used) if budget is not None else None),
            )
            checkpoints.append(snapshot)

            plan: DecisionPlan = controller.decide(snapshot)
            decisions.append(plan)
            evaluations_before_actions = int(budget.used) if budget is not None else None
            states, outcomes = execute_decision_plan(
                plan,
                engines,
                states,
                objective=self.objective,
                seed=None if self.seed is None else self.seed + checkpoint_id,
            )
            if budget is not None:
                for _label, _engine in engines.items():
                    _engine._sync_evaluation_count(states[_label], _label)
                    _engine._remember_best_evaluation_details(states[_label])
                action_evaluations = int(budget.used) - int(evaluations_before_actions or 0)
                if action_evaluations > 0:
                    best_after_actions = None
                    for _label in label_order:
                        fit = states[_label].best_fitness
                        if fit is None:
                            continue
                        if best_after_actions is None or (
                            fit > best_after_actions if self.objective == "max" else fit < best_after_actions
                        ):
                            best_after_actions = fit
                    global_history.append({
                        "event": "orchestration_actions",
                        "label": "__orchestrator__",
                        "algorithm": self.config.orchestration.mode,
                        "step": checkpoint_id,
                        "global_evaluations": int(budget.used),
                        "island_evaluations": None,
                        "evaluations_consumed": action_evaluations,
                        "global_best_fitness": best_after_actions,
                        "best_fitness": best_after_actions,
                    })
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
            state = states[label]
            partial_phase = state.payload.get("_budget_exhausted_phase")
            if label in partial_budget_labels or partial_phase is not None:
                result = engine._make_budget_exhausted_result(
                    state,
                    label=label,
                    phase=str(partial_phase or "initialization"),
                    history=histories[label],
                    snapshots=[],
                    improvement_history=[],
                )
            else:
                result = engine.finalize(state)
            if budget is not None:
                result.evaluations = int(budget.by_label.get(label, 0))
            if result.best_position is not None:
                decoded_best = engine.problem.apply_variable_types(result.best_position).astype(float).tolist()
                result.best_position = decoded_best
                states[label].best_position = decoded_best
            result.history = histories[label]
            island_results[label] = result
            if best_fitness is None or engine.problem.is_better(result.best_fitness, best_fitness):
                best_fitness = result.best_fitness
                best_position = list(result.best_position)

        total_evaluations = (
            int(budget.used)
            if budget is not None
            else sum(int(r.evaluations) for r in island_results.values())
        )
        total_island_steps = sum(int(getattr(result, "steps", 0) or 0) for result in island_results.values())
        if budget is not None and budget.exhausted:
            termination_reason = "max_evaluations"
        elif all(engine.should_stop(states[label]) for label, engine in engines.items()):
            termination_reason = "islands_stopped"
        else:
            termination_reason = "completed"
        budget_report = format_budget_report(
            budget,
            labels=label_order,
            scheduler=scheduler,
            requested_budget=self.max_evaluations,
            prefix="orchestrated-budget",
        )
        if self.verbose and budget is not None:
            print(budget_report)

        named_counts = [int(budget.by_label.get(label, 0)) for label in label_order] if budget is not None else []
        final_evaluation_gap = (max(named_counts) - min(named_counts)) if named_counts else 0
        budget_utilization = (
            None
            if self.max_evaluations in (None, 0)
            else float(total_evaluations / int(self.max_evaluations))
        )

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
                "max_evaluations": self.max_evaluations,
                "total_evaluations": total_evaluations,
                "total_island_steps": total_island_steps,
                "termination_reason": termination_reason,
                "evaluations_by_island": dict(budget.by_label) if budget is not None else {label: int(result.evaluations) for label, result in island_results.items()},
                "evaluations_by_category": dict(budget.by_category) if budget is not None else {},
                "evaluations_by_island_and_category": copy.deepcopy(budget.by_label_category) if budget is not None else {},
                "budget_utilization": budget_utilization,
                "budget_scheduler_policy": scheduler.policy_name if scheduler is not None else None,
                "budget_scheduler_explanation": scheduler.explanation if scheduler is not None else None,
                "scheduler_selection_counts": dict(scheduler.selection_counts) if scheduler is not None else {},
                "scheduler_turns": scheduler_turns,
                "final_island_evaluation_gap": final_evaluation_gap,
                "budget_report": budget_report,
                "checkpoint_interval": self.config.orchestration.checkpoint_interval,
                "constraint_handler": self.constraint_handler or "none",
                "variable_types": list(self.variable_types) if self.variable_types is not None else None,
                "islands": list(engines.keys()),
                "execution_backend_requested": self.execution_backend,
                "execution_backend_used": actual_backend,
                "n_jobs": self.n_jobs,
                "parallel_warning": backend_warning,
                "initialization_budget_exhausted": bool(initialization_budget_exhausted),
                "partial_budget_islands": sorted(partial_budget_labels),
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
            variable_types=kwargs.get("variable_types"),
            repair_function=kwargs.get("repair_function"),
            penalty_coefficient=kwargs.get("penalty_coefficient", 1e6),
            equality_tolerance=kwargs.get("equality_tolerance", 1e-6),
            resample_attempts=kwargs.get("resample_attempts", 25),
            max_steps=kwargs.get("max_steps", 100),
            max_evaluations=kwargs.get("max_evaluations"),
            migration_interval=cfg.orchestration.checkpoint_interval,
            migration_size=cfg.orchestration.migration_size,
            migration_mode=cfg.orchestration.migration_mode,
            topology=cfg.orchestration.topology,
            topology_config=dict(cfg.orchestration.topology_config or {}),
            custom_topology=dict(cfg.orchestration.custom_topology or {}),
            migration_policy=cfg.orchestration.migration_policy,
            donor_strategy=cfg.orchestration.donor_strategy,
            receiver_strategy=cfg.orchestration.receiver_strategy,
            adaptive_checkpointing=cfg.orchestration.adaptive_checkpointing,
            checkpoint_strategy=cfg.orchestration.checkpoint_strategy,
            min_migration_interval=cfg.orchestration.min_migration_interval,
            max_migration_interval=cfg.orchestration.max_migration_interval,
            checkpoint_patience=cfg.orchestration.checkpoint_patience,
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
