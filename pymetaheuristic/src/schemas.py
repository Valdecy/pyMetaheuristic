from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ControllerMode = Literal["fixed", "rules"]

ActionType = Literal[
    "wait",
    "inject",
    "inject_perturbed",
    "rebalance",
    "reconfigure",
    "restart_agent",
    "set_checkpoint",
]
ConfidenceLevel = Literal["low", "medium", "high"]


@dataclass
class OrchestrationSpec:
    mode: ControllerMode = "fixed"
    checkpoint_interval: int = 10
    min_checkpoint_interval: int = 5
    max_checkpoint_interval: int = 100
    max_actions_per_checkpoint: int = 3
    warmup_checkpoints: int = 2
    store_snapshots: bool = True
    store_decisions: bool = True
    allow_disruptive_actions: bool = True
    fallback_action: str = "wait"

    # Fixed/cooperative migration fields.  These are used when
    # orchestrated_optimize(..., mode="fixed") delegates to CooperativeRunner,
    # and they make orchestration and cooperation share the same island-system
    # vocabulary.
    migration_size: int = 1
    migration_mode: str = "elite"
    topology: str = "ring"
    topology_config: dict[str, Any] = field(default_factory=dict)
    custom_topology: dict[str, list[str]] = field(default_factory=dict)
    migration_policy: str = "push"
    donor_strategy: str = "neighbors"
    receiver_strategy: str = "neighbors"
    adaptive_checkpointing: bool = False
    checkpoint_strategy: str = "fixed"
    min_migration_interval: int | None = None
    max_migration_interval: int | None = None
    checkpoint_patience: int = 3


@dataclass
class RulesConfig:
    stagnation_threshold: int = 10
    low_diversity_threshold: float = 0.10
    high_diversity_threshold: float = 0.40
    rebalance_fraction: float = 0.20
    perturbation_sigma: float = 0.05
    reheat_temperature: float | None = None
    exploit_if_budget_used_above: float = 0.80


@dataclass
class CollaborativeConfig:
    orchestration: OrchestrationSpec = field(default_factory=OrchestrationSpec)
    rules: RulesConfig = field(default_factory=RulesConfig)


@dataclass
class AgentSnapshot:
    label: str
    algorithm: str
    family: str
    has_population: bool
    supports_injection: bool
    supports_restart: bool
    step: int
    evaluations: int
    best_fitness: float | None
    best_position: list[float] | None
    current_fitness: float | None = None
    current_position: list[float] | None = None
    delta_best: float | None = None
    stagnation_steps: int | None = None
    diversity: float | None = None
    mean_fitness: float | None = None
    std_fitness: float | None = None
    health: str | None = None
    recent_history: list[dict[str, Any]] = field(default_factory=list)
    params_view: dict[str, Any] = field(default_factory=dict)
    raw_observation: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorSnapshot:
    checkpoint_id: int
    objective: str
    dimension: int
    budget_total: int | None
    budget_used: int
    budget_remaining: int | None
    budget_used_ratio: float | None
    global_best_label: str | None
    global_best_fitness: float | None
    global_best_position: list[float] | None
    agents: list[AgentSnapshot]
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSpec:
    type: ActionType
    source_label: str | None = None
    target_label: str | None = None
    source_mode: Literal["best", "elite", "diverse", "current", "random"] | None = None
    k: int | None = None
    replace_policy: Literal["worst", "random", "current", "native"] | None = None
    perturbation: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_effect: str | None = None
    confidence: ConfidenceLevel | None = None


@dataclass
class DecisionPlan:
    controller_mode: ControllerMode
    controller_name: str
    reasoning: str
    confidence: ConfidenceLevel
    actions: list[ActionSpec]
    next_checkpoint_interval: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionOutcome:
    action: ActionSpec
    executed: bool
    status: Literal["applied", "skipped", "downgraded", "failed"]
    message: str = ""
    source_fitness: float | None = None
    target_fitness_before: float | None = None
    target_fitness_after: float | None = None


@dataclass
class OrchestratedCooperativeResult:
    best_position: list[float]
    best_fitness: float
    island_results: dict[str, Any]
    history: list[dict[str, Any]]
    events: list[dict[str, Any]]
    controller_mode: str
    checkpoints: list[OrchestratorSnapshot]
    decisions: list[DecisionPlan]
    outcomes: list[list[ActionOutcome]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def migration_matrix(self, value: str = "migrants", include_zero: bool = True, objective: str | None = None) -> dict[str, dict[str, float]]:
        from .diagnostics import migration_matrix
        return migration_matrix(self, value=value, include_zero=include_zero, objective=objective)

    def topology_summary(self) -> dict[str, Any]:
        from .diagnostics import topology_summary
        return topology_summary(self)

    def island_contribution(self, objective: str | None = None) -> dict[str, dict[str, Any]]:
        from .diagnostics import island_contribution
        return island_contribution(self, objective=objective)

    def island_roles(self, objective: str | None = None) -> dict[str, dict[str, Any]]:
        from .diagnostics import island_roles
        return island_roles(self, objective=objective)

    def action_effectiveness(self, objective: str | None = None) -> dict[str, Any]:
        from .diagnostics import action_effectiveness
        return action_effectiveness(self, objective=objective)

    def diagnostics_summary(self, objective: str | None = None) -> dict[str, Any]:
        from .diagnostics import diagnostics_summary
        return diagnostics_summary(self, objective=objective)

    def plot_migration_network(self, ax=None, value: str = "migrants", **kwargs):
        from .diagnostics import plot_migration_network
        return plot_migration_network(self, ax=ax, value=value, **kwargs)

    def plot_island_fitness(self, ax=None, objective: str | None = None, **kwargs):
        from .diagnostics import plot_island_fitness
        return plot_island_fitness(self, ax=ax, objective=objective, **kwargs)
