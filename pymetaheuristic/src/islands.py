"""First-class island-system architecture for collaborative optimization.

This module gives the cooperative/orchestrated stack a small, explicit object
model.  It does not replace the existing public functions; it wraps them with
configuration objects that are easier to read, serialize, validate, and extend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from .cooperation import CooperativeResult, IslandSpec, cooperative_optimize
from .orchestration import orchestrated_optimize
from .schemas import CollaborativeConfig, OrchestrationSpec, RulesConfig

IslandMode = Literal["cooperative", "fixed", "orchestrated", "rules", "bandit", "portfolio_adaptive"]


@dataclass
class IslandConfig:
    """Configuration for a single island optimizer.

    Parameters
    ----------
    algorithm:
        Algorithm identifier accepted by :func:`create_optimizer`.
    config:
        Algorithm-specific hyperparameters.
    label:
        Optional stable label used in logs, migration events, and plots.
    seed:
        Optional island-specific seed.  When omitted, the system seed is offset
        by island index, preserving the behavior of the legacy runners.
    role:
        Semantic role used by higher-level orchestration/diagnostics.  Phase I
        stores the role but does not force the old engines to interpret it.
    metadata:
        Free-form user metadata, useful for reports and future ablations.
    """

    algorithm: str
    config: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    seed: int | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: "IslandConfig | IslandSpec | dict[str, Any]") -> "IslandConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, IslandSpec):
            return cls(
                algorithm=value.algorithm,
                config=dict(value.config or {}),
                label=value.label,
                seed=value.seed,
            )
        if isinstance(value, dict):
            allowed = {"algorithm", "config", "label", "seed", "role", "metadata"}
            extra = sorted(set(value) - allowed)
            if extra:
                raise TypeError(f"Unsupported IslandConfig fields: {', '.join(extra)}")
            return cls(**value)
        raise TypeError("island entries must be IslandConfig, IslandSpec, or dict")

    def to_island_spec(self) -> IslandSpec:
        """Return the legacy runner representation."""
        return IslandSpec(
            algorithm=self.algorithm,
            config=dict(self.config or {}),
            label=self.label,
            seed=self.seed,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a dict accepted by the existing cooperative/orchestrated API."""
        return {
            "algorithm": self.algorithm,
            "config": dict(self.config or {}),
            "label": self.label,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TopologyConfig:
    """Topology connecting islands during migration."""

    name: str = "ring"
    config: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: "TopologyConfig | str | dict[str, Any] | None") -> "TopologyConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            if "topology" in value and "name" not in value:
                value = {**value, "name": value["topology"]}
            allowed = {"name", "config", "custom", "topology"}
            extra = sorted(set(value) - allowed)
            if extra:
                raise TypeError(f"Unsupported TopologyConfig fields: {', '.join(extra)}")
            return cls(
                name=value.get("name", value.get("topology", "ring")),
                config=dict(value.get("config") or {}),
                custom=dict(value.get("custom") or {}),
            )
        raise TypeError("topology must be TopologyConfig, str, dict, or None")

    def to_cooperative_kwargs(self) -> dict[str, Any]:
        return {
            "topology": self.name,
            "topology_config": dict(self.config or {}),
            "custom_topology": dict(self.custom or {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationConfig:
    """Migration and checkpointing policy for fixed island cooperation."""

    interval: int = 5
    size: int = 1
    mode: str = "elite"
    policy: str = "push"
    donor_strategy: str = "neighbors"
    receiver_strategy: str = "neighbors"
    adaptive_checkpointing: bool = False
    checkpoint_strategy: str = "fixed"
    min_interval: int | None = None
    max_interval: int | None = None
    checkpoint_patience: int = 3

    @classmethod
    def from_any(cls, value: "MigrationConfig | dict[str, Any] | None") -> "MigrationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            aliases = {
                "migration_interval": "interval",
                "migration_size": "size",
                "migration_mode": "mode",
                "migration_policy": "policy",
                "min_migration_interval": "min_interval",
                "max_migration_interval": "max_interval",
            }
            normalized = {aliases.get(k, k): v for k, v in value.items()}
            allowed = {
                "interval", "size", "mode", "policy", "donor_strategy",
                "receiver_strategy", "adaptive_checkpointing", "checkpoint_strategy",
                "min_interval", "max_interval", "checkpoint_patience",
            }
            extra = sorted(set(normalized) - allowed)
            if extra:
                raise TypeError(f"Unsupported MigrationConfig fields: {', '.join(extra)}")
            return cls(**normalized)
        raise TypeError("migration must be MigrationConfig, dict, or None")

    def to_cooperative_kwargs(self) -> dict[str, Any]:
        return {
            "migration_interval": int(max(1, self.interval)),
            "migration_size": int(max(1, self.size)),
            "migration_mode": self.mode,
            "migration_policy": self.policy,
            "donor_strategy": self.donor_strategy,
            "receiver_strategy": self.receiver_strategy,
            "adaptive_checkpointing": bool(self.adaptive_checkpointing),
            "checkpoint_strategy": self.checkpoint_strategy,
            "min_migration_interval": self.min_interval,
            "max_migration_interval": self.max_interval,
            "checkpoint_patience": int(max(1, self.checkpoint_patience)),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionConfig:
    """Execution backend for island stepping."""

    backend: str = "serial"
    n_jobs: int | None = None
    parallel_fallback_to_serial: bool = True

    @classmethod
    def from_any(cls, value: "ExecutionConfig | dict[str, Any] | None") -> "ExecutionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            aliases = {"execution_backend": "backend"}
            normalized = {aliases.get(k, k): v for k, v in value.items()}
            allowed = {"backend", "n_jobs", "parallel_fallback_to_serial"}
            extra = sorted(set(normalized) - allowed)
            if extra:
                raise TypeError(f"Unsupported ExecutionConfig fields: {', '.join(extra)}")
            return cls(**normalized)
        raise TypeError("execution must be ExecutionConfig, dict, or None")

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "execution_backend": self.backend,
            "n_jobs": self.n_jobs,
            "parallel_fallback_to_serial": self.parallel_fallback_to_serial,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationConfig:
    """Adaptive orchestration settings for :func:`orchestrated_optimize`.

    This object intentionally mirrors ``OrchestrationSpec`` while also carrying
    the fixed-migration fields needed when ``mode='fixed'`` delegates to the
    cooperative runner.
    """

    mode: Literal["fixed", "rules", "bandit", "portfolio_adaptive"] = "fixed"
    checkpoint_interval: int = 10
    min_checkpoint_interval: int = 5
    max_checkpoint_interval: int = 100
    max_actions_per_checkpoint: int = 3
    warmup_checkpoints: int = 2
    store_snapshots: bool = True
    store_decisions: bool = True
    allow_disruptive_actions: bool = True
    fallback_action: str = "wait"

    @classmethod
    def from_any(cls, value: "OrchestrationConfig | OrchestrationSpec | dict[str, Any] | None") -> "OrchestrationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, OrchestrationSpec):
            data = asdict(value)
            allowed = set(cls.__dataclass_fields__)
            return cls(**{k: v for k, v in data.items() if k in allowed})
        if isinstance(value, dict):
            allowed = set(cls.__dataclass_fields__)
            extra = sorted(set(value) - allowed)
            if extra:
                raise TypeError(f"Unsupported OrchestrationConfig fields: {', '.join(extra)}")
            return cls(**value)
        raise TypeError("orchestration must be OrchestrationConfig, OrchestrationSpec, dict, or None")

    def to_spec(self, migration: MigrationConfig | None = None, topology: TopologyConfig | None = None) -> OrchestrationSpec:
        migration = MigrationConfig.from_any(migration)
        topology = TopologyConfig.from_any(topology)
        kwargs = {
            "mode": self.mode,
            "checkpoint_interval": int(max(1, self.checkpoint_interval)),
            "min_checkpoint_interval": int(max(1, self.min_checkpoint_interval)),
            "max_checkpoint_interval": int(max(1, self.max_checkpoint_interval)),
            "max_actions_per_checkpoint": int(max(1, self.max_actions_per_checkpoint)),
            "warmup_checkpoints": int(max(0, self.warmup_checkpoints)),
            "store_snapshots": bool(self.store_snapshots),
            "store_decisions": bool(self.store_decisions),
            "allow_disruptive_actions": bool(self.allow_disruptive_actions),
            "fallback_action": self.fallback_action,
        }
        # Newer OrchestrationSpec versions include fixed-migration fields.  The
        # guard keeps this wrapper compatible with older package snapshots.
        spec_fields = set(getattr(OrchestrationSpec, "__dataclass_fields__", {}))
        migration_fields = {
            "migration_size": int(max(1, migration.size)),
            "migration_mode": migration.mode,
            "topology": topology.name,
            "topology_config": dict(topology.config or {}),
            "custom_topology": dict(topology.custom or {}),
            "migration_policy": migration.policy,
            "donor_strategy": migration.donor_strategy,
            "receiver_strategy": migration.receiver_strategy,
            "adaptive_checkpointing": bool(migration.adaptive_checkpointing),
            "checkpoint_strategy": migration.checkpoint_strategy,
            "min_migration_interval": migration.min_interval,
            "max_migration_interval": migration.max_interval,
            "checkpoint_patience": int(max(1, migration.checkpoint_patience)),
        }
        kwargs.update({k: v for k, v in migration_fields.items() if k in spec_fields})
        return OrchestrationSpec(**kwargs)

    def to_collaborative_config(
        self,
        migration: MigrationConfig | None = None,
        topology: TopologyConfig | None = None,
        rules: RulesConfig | dict[str, Any] | None = None,
    ) -> CollaborativeConfig:
        if rules is None:
            rules_obj = RulesConfig()
        elif isinstance(rules, RulesConfig):
            rules_obj = rules
        elif isinstance(rules, dict):
            rules_obj = RulesConfig(**rules)
        else:
            raise TypeError("rules must be RulesConfig, dict, or None")
        return CollaborativeConfig(
            orchestration=self.to_spec(migration=migration, topology=topology),
            rules=rules_obj,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IslandSystem:
    """First-class object for fixed or orchestrated island optimization.

    The class is intentionally thin: it centralizes configuration and delegates
    execution to the stable public functions already present in the package.
    """

    islands: list[IslandConfig | IslandSpec | dict[str, Any]]
    topology: TopologyConfig | str | dict[str, Any] | None = None
    migration: MigrationConfig | dict[str, Any] | None = None
    orchestration: OrchestrationConfig | OrchestrationSpec | dict[str, Any] | None = None
    rules: RulesConfig | dict[str, Any] | None = None
    execution: ExecutionConfig | dict[str, Any] | None = None
    objective: Literal["min", "max"] = "min"
    constraints: Any = None
    constraint_handler: Any = None
    variable_types: list[Any] | None = None
    repair_function: Any = None
    penalty_coefficient: float = 1e6
    equality_tolerance: float = 1e-6
    resample_attempts: int = 25
    max_steps: int = 50
    max_evaluations: int | None = None
    seed: int | None = None
    verbose: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.islands = [IslandConfig.from_any(x) for x in self.islands]
        self.topology = TopologyConfig.from_any(self.topology)
        self.migration = MigrationConfig.from_any(self.migration)
        self.orchestration = OrchestrationConfig.from_any(self.orchestration)
        self.execution = ExecutionConfig.from_any(self.execution)
        if self.rules is not None and not isinstance(self.rules, (RulesConfig, dict)):
            raise TypeError("rules must be RulesConfig, dict, or None")
        if not self.islands:
            raise ValueError("IslandSystem requires at least one island")
        labels = [island.label for island in self.islands if island.label is not None]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"Duplicate island labels: {', '.join(duplicates)}")
        if self.objective not in {"min", "max"}:
            raise ValueError("objective must be 'min' or 'max'")

    def _common_kwargs(self, target_function, min_values, max_values, **overrides) -> dict[str, Any]:
        kwargs = {
            "islands": [island.to_legacy_dict() for island in self.islands],
            "target_function": target_function,
            "min_values": min_values,
            "max_values": max_values,
            "objective": self.objective,
            "constraints": self.constraints,
            "constraint_handler": self.constraint_handler,
            "variable_types": self.variable_types,
            "repair_function": self.repair_function,
            "penalty_coefficient": self.penalty_coefficient,
            "equality_tolerance": self.equality_tolerance,
            "resample_attempts": self.resample_attempts,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "verbose": self.verbose,
        }
        kwargs.update(self.execution.to_kwargs())
        kwargs.update(overrides)
        return kwargs

    def cooperative_kwargs(self, target_function, min_values, max_values, **overrides) -> dict[str, Any]:
        kwargs = self._common_kwargs(target_function, min_values, max_values, **overrides)
        kwargs.update(self.topology.to_cooperative_kwargs())
        kwargs.update(self.migration.to_cooperative_kwargs())
        # CooperativeRunner currently uses max_steps, not max_evaluations.
        kwargs.pop("max_evaluations", None)
        return kwargs

    def orchestrated_kwargs(self, target_function, min_values, max_values, mode: Literal["fixed", "rules", "bandit", "portfolio_adaptive"] | None = None, **overrides) -> dict[str, Any]:
        kwargs = self._common_kwargs(target_function, min_values, max_values, **overrides)
        if self.max_evaluations is not None:
            kwargs["max_evaluations"] = self.max_evaluations
        orch = self.orchestration
        if mode is not None:
            orch = OrchestrationConfig.from_any({**orch.to_dict(), "mode": mode})
        kwargs["config"] = orch.to_collaborative_config(
            migration=self.migration,
            topology=self.topology,
            rules=self.rules,
        )
        return kwargs

    def cooperative(self, target_function, min_values, max_values, **overrides) -> CooperativeResult:
        """Run fixed island cooperation using the configured migration system."""
        return cooperative_optimize(**self.cooperative_kwargs(target_function, min_values, max_values, **overrides))

    def orchestrated(self, target_function, min_values, max_values, mode: Literal["fixed", "rules", "bandit", "portfolio_adaptive"] | None = None, **overrides):
        """Run checkpoint-based orchestration.

        ``mode='fixed'`` returns the orchestrated-compatible result wrapper while
        delegating migration to the cooperative runner. ``mode='rules'`` uses the
        rule-based adaptive controller.
        """
        return orchestrated_optimize(**self.orchestrated_kwargs(target_function, min_values, max_values, mode=mode, **overrides))

    def optimize(self, target_function, min_values, max_values, mode: IslandMode = "cooperative", **overrides):
        """Run the system using a concise mode selector.

        Modes
        -----
        cooperative:
            Direct fixed island migration; returns ``CooperativeResult``.
        fixed:
            Orchestrated-compatible fixed migration; returns
            ``OrchestratedCooperativeResult``.
        rules / orchestrated:
            Rule-based adaptive orchestration.
        """
        normalized = (mode or "cooperative").lower()
        if normalized == "cooperative":
            return self.cooperative(target_function, min_values, max_values, **overrides)
        if normalized == "fixed":
            return self.orchestrated(target_function, min_values, max_values, mode="fixed", **overrides)
        if normalized in {"rules", "orchestrated"}:
            return self.orchestrated(target_function, min_values, max_values, mode="rules", **overrides)
        if normalized == "bandit":
            return self.orchestrated(target_function, min_values, max_values, mode="bandit", **overrides)
        if normalized in {"portfolio", "portfolio_adaptive"}:
            return self.orchestrated(target_function, min_values, max_values, mode="portfolio_adaptive", **overrides)
        raise ValueError("mode must be one of: 'cooperative', 'fixed', 'rules', 'orchestrated', 'bandit', 'portfolio_adaptive'")

    def describe(self) -> dict[str, Any]:
        """Return a serializable system description."""
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        data = {
            "islands": [island.to_dict() for island in self.islands],
            "topology": self.topology.to_dict(),
            "migration": self.migration.to_dict(),
            "orchestration": self.orchestration.to_dict(),
            "rules": asdict(self.rules) if is_dataclass(self.rules) else dict(self.rules or {}),
            "execution": self.execution.to_dict(),
            "objective": self.objective,
            "constraints": self.constraints,
            "constraint_handler": self.constraint_handler,
            "variable_types": list(self.variable_types) if self.variable_types is not None else None,
            "repair_function": self.repair_function,
            "penalty_coefficient": self.penalty_coefficient,
            "equality_tolerance": self.equality_tolerance,
            "resample_attempts": self.resample_attempts,
            "max_steps": self.max_steps,
            "max_evaluations": self.max_evaluations,
            "seed": self.seed,
            "verbose": self.verbose,
            "metadata": dict(self.metadata or {}),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IslandSystem":
        return cls(**dict(data))


# Friendly short alias.
Island = IslandConfig


__all__ = [
    "Island",
    "IslandConfig",
    "IslandMode",
    "IslandSystem",
    "TopologyConfig",
    "MigrationConfig",
    "ExecutionConfig",
    "OrchestrationConfig",
]
