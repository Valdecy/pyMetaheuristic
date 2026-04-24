from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import random

from .api import create_optimizer
from .execution import run_engine_chunks


@dataclass
class IslandSpec:
    algorithm: str
    config: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    seed: int | None = None


@dataclass
class CooperationEvent:
    global_step: int
    source_label: str
    target_label: str
    source_algorithm: str
    target_algorithm: str
    migrants: int
    best_fitness_after: float | None
    policy: str = "push"
    donor_strategy: str = "neighbors"
    receiver_strategy: str = "neighbors"
    checkpoint_interval: int | None = None


@dataclass
class IslandTelemetryRecord:
    global_step: int
    label: str
    algorithm: str
    step: int
    evaluations: int
    best_fitness: float | None
    delta_best: float | None
    stagnation_steps: int
    diversity: float | None
    mean_fitness: float | None
    std_fitness: float | None
    health: float
    migration_interval: int
    neighbors: list[str] = field(default_factory=list)


@dataclass
class CooperativeResult:
    best_position: list[float]
    best_fitness: float
    island_results: dict[str, Any]
    hall_of_fame: list[dict[str, Any]]
    events: list[CooperationEvent]
    history: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    island_telemetry: dict[str, list[IslandTelemetryRecord]] = field(default_factory=dict)
    replay_manifest: dict[str, Any] = field(default_factory=dict)


def _score_for_objective(fitness: float | None, objective: str) -> float:
    if fitness is None:
        return float('-inf') if objective == 'max' else float('inf')
    return float(fitness)


def _is_better(a: float | None, b: float | None, objective: str) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a > b if objective == 'max' else a < b


def _health_from_observation(obs: dict[str, Any], has_population: bool) -> float:
    diversity = obs.get('diversity')
    std = obs.get('std_fitness')
    stagnation = float(obs.get('stagnation_steps', 0) or 0)
    score = 0.75
    if diversity is not None:
        try:
            score += max(-0.30, min(0.20, float(diversity) - 0.20))
        except Exception:
            pass
    elif has_population and std is not None:
        try:
            score += max(-0.20, min(0.15, float(std) / (abs(float(obs.get('mean_fitness', 1.0) or 1.0)) + 1e-12)))
        except Exception:
            pass
    score -= min(0.45, 0.04 * stagnation)
    return max(0.0, min(1.0, float(score)))


class CooperativeRunner:
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
        max_steps: int = 50,
        migration_interval: int = 5,
        migration_size: int = 1,
        migration_mode: str = "elite",
        topology: str = "ring",
        seed: int | None = None,
        verbose: bool = False,
        execution_backend: str = "serial",
        n_jobs: int | None = None,
        parallel_fallback_to_serial: bool = True,
        topology_config: dict[str, Any] | None = None,
        custom_topology: dict[str, list[str]] | None = None,
        migration_policy: str = "push",
        donor_strategy: str = "neighbors",
        receiver_strategy: str = "neighbors",
        adaptive_checkpointing: bool = False,
        checkpoint_strategy: str = "fixed",
        min_migration_interval: int | None = None,
        max_migration_interval: int | None = None,
        checkpoint_patience: int = 3,
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
        self.migration_interval = int(max(1, migration_interval))
        self.migration_size = int(max(1, migration_size))
        self.migration_mode = migration_mode
        self.topology = topology
        self.seed = seed
        self.verbose = verbose
        self.execution_backend = execution_backend
        self.n_jobs = n_jobs
        self.parallel_fallback_to_serial = parallel_fallback_to_serial
        self.topology_config = dict(topology_config or {})
        self.custom_topology = dict(custom_topology or {})
        self.migration_policy = migration_policy
        self.donor_strategy = donor_strategy
        self.receiver_strategy = receiver_strategy
        self.adaptive_checkpointing = bool(adaptive_checkpointing)
        self.checkpoint_strategy = checkpoint_strategy
        self.min_migration_interval = int(max(1, min_migration_interval or max(1, self.migration_interval // 2)))
        self.max_migration_interval = int(max(self.min_migration_interval, max_migration_interval or max(self.migration_interval, self.migration_interval * 3)))
        self.checkpoint_patience = int(max(1, checkpoint_patience))
        self._rng = random.Random(seed)

    def _adjacency(self, labels: list[str]) -> dict[str, list[str]]:
        n = len(labels)
        if n <= 1:
            return {label: [] for label in labels}
        if self.custom_topology:
            return {label: [nbr for nbr in self.custom_topology.get(label, []) if nbr in labels and nbr != label] for label in labels}
        topo = (self.topology or 'ring').lower()
        adj: dict[str, list[str]] = {label: [] for label in labels}
        if topo == 'ring':
            for i, label in enumerate(labels):
                adj[label] = [labels[(i + 1) % n]]
        elif topo in {'bidirectional_ring', 'ring2'}:
            for i, label in enumerate(labels):
                adj[label] = [labels[(i - 1) % n], labels[(i + 1) % n]]
        elif topo == 'line':
            for i, label in enumerate(labels):
                nbrs = []
                if i > 0:
                    nbrs.append(labels[i - 1])
                if i + 1 < n:
                    nbrs.append(labels[i + 1])
                adj[label] = nbrs
        elif topo == 'star':
            hub = labels[0]
            for label in labels:
                adj[label] = [hub] if label != hub else [x for x in labels[1:]]
        elif topo == 'wheel':
            hub = labels[0]
            for i, label in enumerate(labels):
                if label == hub:
                    adj[label] = [x for x in labels[1:]]
                else:
                    rim = labels[1:]
                    ridx = rim.index(label)
                    adj[label] = [hub, rim[(ridx - 1) % len(rim)], rim[(ridx + 1) % len(rim)]]
        elif topo == 'full':
            for label in labels:
                adj[label] = [j for j in labels if j != label]
        elif topo == 'random_sparse':
            k = max(1, min(n - 1, int(self.topology_config.get('degree', round(n ** 0.5)))))
            for label in labels:
                others = [j for j in labels if j != label]
                adj[label] = self._rng.sample(others, k)
        elif topo == 'clusters':
            size = max(2, int(self.topology_config.get('cluster_size', round(n ** 0.5))))
            clusters = [labels[i:i + size] for i in range(0, n, size)]
            for cluster in clusters:
                for idx, label in enumerate(cluster):
                    nbrs = [x for x in cluster if x != label]
                    if len(clusters) > 1:
                        other_cluster = clusters[(clusters.index(cluster) + 1) % len(clusters)]
                        if other_cluster:
                            nbrs.append(other_cluster[min(idx, len(other_cluster) - 1)])
                    adj[label] = nbrs
        else:
            raise ValueError(f'Unsupported topology: {self.topology}')
        return adj

    def _select_donors(self, labels: list[str], states: dict[str, Any], engines: dict[str, Any], telemetry: dict[str, list[IslandTelemetryRecord]], adjacency: dict[str, list[str]]) -> list[str]:
        strategy = (self.donor_strategy or 'neighbors').lower()
        if strategy in {'all', 'neighbors'}:
            return list(labels)
        if strategy == 'best':
            best = min(labels, key=lambda l: _score_for_objective(states[l].best_fitness, self.objective)) if self.objective == 'min' else max(labels, key=lambda l: _score_for_objective(states[l].best_fitness, self.objective))
            return [best]
        if strategy == 'improving':
            out = []
            for label in labels:
                recs = telemetry.get(label, [])
                if recs and (recs[-1].delta_best is None or (recs[-1].delta_best < 0 if self.objective == 'min' else recs[-1].delta_best > 0)):
                    out.append(label)
            return out or list(labels)
        if strategy == 'diverse':
            def _div_key(l):
                recs = telemetry.get(l, [])
                if not recs:
                    return -1.0
                d = recs[-1].diversity
                return float(d) if d is not None else -1.0
            ranked = sorted(labels, key=_div_key, reverse=True)
            return ranked[: max(1, len(labels) // 2)]
        return list(labels)

    def _candidate_receivers(self, donor: str, labels: list[str], states: dict[str, Any], telemetry: dict[str, list[IslandTelemetryRecord]], adjacency: dict[str, list[str]]) -> list[str]:
        strategy = (self.receiver_strategy or 'neighbors').lower()
        others = [l for l in labels if l != donor]
        if strategy == 'neighbors':
            return [l for l in adjacency.get(donor, []) if l != donor]
        if strategy == 'all':
            return others
        if strategy == 'random':
            return [self._rng.choice(others)] if others else []
        if strategy == 'worst':
            if not others:
                return []
            worst = max(others, key=lambda l: _score_for_objective(states[l].best_fitness, self.objective)) if self.objective == 'min' else min(others, key=lambda l: _score_for_objective(states[l].best_fitness, self.objective))
            return [worst]
        if strategy == 'stagnated':
            ranked = sorted(others, key=lambda l: telemetry.get(l, [])[-1].stagnation_steps if telemetry.get(l, []) else 0, reverse=True)
            return ranked[: max(1, min(2, len(ranked)))]
        if strategy == 'low_diversity':
            def _low_div_key(l):
                recs = telemetry.get(l, [])
                if not recs:
                    return 1e9
                d = recs[-1].diversity
                return float(d) if d is not None else 1e9
            ranked = sorted(others, key=_low_div_key)
            return ranked[: max(1, min(2, len(ranked)))]
        return others

    def _effective_policy(self, states: dict[str, Any], telemetry: dict[str, list[IslandTelemetryRecord]]) -> str:
        policy = (self.migration_policy or 'push').lower()
        if policy != 'adaptive':
            return policy
        low_div = False
        stagnated = False
        for label, recs in telemetry.items():
            if recs:
                low_div = low_div or ((recs[-1].diversity or 1.0) < 0.15)
                stagnated = stagnated or (recs[-1].stagnation_steps >= self.checkpoint_patience)
        if stagnated and low_div:
            return 'broadcast_best'
        if stagnated:
            return 'push_pull'
        return 'push'

    def _update_interval(self, current_interval: int, global_best_trace: list[float | None]) -> int:
        if not self.adaptive_checkpointing or (self.checkpoint_strategy or 'fixed').lower() == 'fixed':
            return current_interval
        if len(global_best_trace) < 2:
            return current_interval
        recent = global_best_trace[-min(len(global_best_trace), self.checkpoint_patience + 1):]
        improvements = 0
        for prev, curr in zip(recent[:-1], recent[1:]):
            if prev is None or curr is None:
                continue
            if _is_better(curr, prev, self.objective):
                improvements += 1
        if improvements == 0:
            return max(self.min_migration_interval, current_interval - 1)
        if improvements >= max(1, len(recent) - 2):
            return min(self.max_migration_interval, current_interval + 1)
        return current_interval

    def run(self) -> CooperativeResult:
        islands = []
        states = []
        histories = []
        events: list[CooperationEvent] = []
        hall_of_fame: list[dict[str, Any]] = []
        global_history: list[dict[str, Any]] = []
        telemetry_by_island: dict[str, list[IslandTelemetryRecord]] = {}
        previous_best: dict[str, float | None] = {}
        stagnation_counter: dict[str, int] = {}
        global_best_trace: list[float | None] = []

        for i, spec in enumerate(self.island_specs):
            label = spec.label or f"{spec.algorithm}_{i+1}"
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
                seed=(spec.seed if spec.seed is not None else (None if self.seed is None else self.seed + i)),
                verbose=False,
                store_history=False,
                config=spec.config,
            )
            state = engine.initialize()
            islands.append((label, engine))
            states.append(state)
            histories.append([])
            telemetry_by_island[label] = []
            previous_best[label] = state.best_fitness
            stagnation_counter[label] = 0
            hall_of_fame.append({
                'label': label,
                'algorithm': engine.algorithm_id,
                'fitness': state.best_fitness,
                'position': list(state.best_position),
            })

        n = len(islands)
        labels = [label for label, _ in islands]
        adjacency = self._adjacency(labels)
        states_map = {label: state for (label, _), state in zip(islands, states)}
        engines_map = {label: engine for label, engine in islands}
        current_interval = self.migration_interval
        rounds_since_migration = 0
        global_step = 0
        actual_backend = (self.execution_backend or 'serial').lower()
        backend_warning = None

        while True:
            chunk_items = []
            active_indices = []
            for idx, (label, engine) in enumerate(islands):
                state = states_map[label]
                if engine.should_stop(state):
                    continue
                active_indices.append(idx)
                chunk_items.append((label, engine, state, 1))
            if not chunk_items:
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
                    print(f"[cooperation] {warning}")

            for idx, result in zip(active_indices, results):
                label, engine = islands[idx]
                state = result.state
                states_map[label] = state
                obs_payload = None
                for obs in result.observations:
                    obs_payload = obs
                    histories[idx].append(obs)
                    global_history.append({'global_step': global_step, 'label': label, 'algorithm': engine.algorithm_id, **obs})
                    global_step += 1
                if obs_payload is None:
                    obs_payload = dict(engine.observe(state))
                best_fit = obs_payload.get('best_fitness', state.best_fitness)
                prev_fit = previous_best.get(label)
                delta = None if prev_fit is None or best_fit is None else float(best_fit - prev_fit)
                improved = False if prev_fit is None or best_fit is None else engine.problem.is_better(best_fit, prev_fit)
                stagnation_counter[label] = 0 if improved else (stagnation_counter.get(label, 0) + 1)
                previous_best[label] = best_fit
                telem = IslandTelemetryRecord(
                    global_step=global_step,
                    label=label,
                    algorithm=engine.algorithm_id,
                    step=state.step,
                    evaluations=state.evaluations,
                    best_fitness=best_fit,
                    delta_best=delta,
                    stagnation_steps=stagnation_counter[label],
                    diversity=obs_payload.get('diversity'),
                    mean_fitness=obs_payload.get('mean_fitness'),
                    std_fitness=obs_payload.get('std_fitness'),
                    health=_health_from_observation({**obs_payload, 'stagnation_steps': stagnation_counter[label]}, engine.capabilities.has_population),
                    migration_interval=current_interval,
                    neighbors=list(adjacency.get(label, [])),
                )
                telemetry_by_island[label].append(telem)

                if state.best_fitness is not None:
                    hall_of_fame.append({
                        'label': label,
                        'algorithm': engine.algorithm_id,
                        'fitness': state.best_fitness,
                        'position': list(state.best_position),
                    })

            best_now = None
            for label, _engine in islands:
                fit = states_map[label].best_fitness
                if best_now is None or _is_better(fit, best_now, self.objective):
                    best_now = fit
            global_best_trace.append(best_now)

            rounds_since_migration += 1
            if rounds_since_migration < current_interval:
                continue
            rounds_since_migration = 0

            effective_policy = self._effective_policy(states_map, telemetry_by_island)
            donors = self._select_donors(labels, states_map, engines_map, telemetry_by_island, adjacency)
            applied_pairs: set[tuple[str, str]] = set()

            def _send(donor_label: str, receiver_label: str):
                if donor_label == receiver_label or (donor_label, receiver_label) in applied_pairs:
                    return
                source_engine = engines_map[donor_label]
                target_engine = engines_map[receiver_label]
                source_state = states_map[donor_label]
                target_state = states_map[receiver_label]
                if not target_engine.capabilities.supports_candidate_injection:
                    return
                migrants = source_engine.export_candidates(source_state, k=self.migration_size, mode=self.migration_mode)
                if not migrants:
                    return
                states_map[receiver_label] = target_engine.inject_candidates(target_state, migrants, policy='native')
                applied_pairs.add((donor_label, receiver_label))
                events.append(CooperationEvent(
                    global_step=global_step,
                    source_label=donor_label,
                    target_label=receiver_label,
                    source_algorithm=source_engine.algorithm_id,
                    target_algorithm=target_engine.algorithm_id,
                    migrants=len(migrants),
                    best_fitness_after=states_map[receiver_label].best_fitness,
                    policy=effective_policy,
                    donor_strategy=self.donor_strategy,
                    receiver_strategy=self.receiver_strategy,
                    checkpoint_interval=current_interval,
                ))

            if effective_policy == 'broadcast_best' and donors:
                donor_label = donors[0]
                if self.objective == 'min':
                    donor_label = min(donors, key=lambda l: _score_for_objective(states_map[l].best_fitness, self.objective))
                else:
                    donor_label = max(donors, key=lambda l: _score_for_objective(states_map[l].best_fitness, self.objective))
                for receiver_label in self._candidate_receivers(donor_label, labels, states_map, telemetry_by_island, adjacency):
                    _send(donor_label, receiver_label)
            else:
                for donor_label in donors:
                    receivers = self._candidate_receivers(donor_label, labels, states_map, telemetry_by_island, adjacency)
                    for receiver_label in receivers:
                        if effective_policy in {'push', 'push_pull'}:
                            _send(donor_label, receiver_label)
                        if effective_policy in {'pull', 'push_pull'}:
                            _send(receiver_label, donor_label)

            current_interval = self._update_interval(current_interval, global_best_trace)

        island_results = {}
        best_position = None
        best_fitness = None
        for (label, engine), hist in zip(islands, histories):
            result = engine.finalize(states_map[label])
            result.history = hist
            island_results[label] = result
            if best_fitness is None or engine.problem.is_better(result.best_fitness, best_fitness):
                best_fitness = result.best_fitness
                best_position = list(result.best_position)

        hall_of_fame_sorted = sorted(
            hall_of_fame,
            key=lambda x: x['fitness'],
            reverse=(self.objective == 'max'),
        )[: max(5, self.migration_size * len(islands))]

        replay_manifest = {
            'runner': 'cooperative',
            'objective': self.objective,
            'min_values': list(self.min_values),
            'max_values': list(self.max_values),
            'max_steps': self.max_steps,
            'seed': self.seed,
            'migration_interval': self.migration_interval,
            'migration_size': self.migration_size,
            'migration_mode': self.migration_mode,
            'topology': self.topology,
            'topology_config': dict(self.topology_config),
            'custom_topology': dict(self.custom_topology),
            'migration_policy': self.migration_policy,
            'donor_strategy': self.donor_strategy,
            'receiver_strategy': self.receiver_strategy,
            'adaptive_checkpointing': self.adaptive_checkpointing,
            'checkpoint_strategy': self.checkpoint_strategy,
            'min_migration_interval': self.min_migration_interval,
            'max_migration_interval': self.max_migration_interval,
            'checkpoint_patience': self.checkpoint_patience,
            'execution_backend': self.execution_backend,
            'n_jobs': self.n_jobs,
            'parallel_fallback_to_serial': self.parallel_fallback_to_serial,
            'islands': [
                {'algorithm': spec.algorithm, 'config': dict(spec.config), 'label': spec.label, 'seed': spec.seed}
                for spec in self.island_specs
            ],
            'events': [event.__dict__ for event in events],
        }

        return CooperativeResult(
            best_position=best_position,
            best_fitness=best_fitness,
            island_results=island_results,
            hall_of_fame=hall_of_fame_sorted,
            events=events,
            history=global_history,
            metadata={
                'topology': self.topology,
                'topology_config': dict(self.topology_config),
                'custom_topology': dict(self.custom_topology),
                'migration_interval': self.migration_interval,
                'final_migration_interval': current_interval,
                'migration_size': self.migration_size,
                'migration_mode': self.migration_mode,
                'migration_policy': self.migration_policy,
                'effective_migration_policy_last': self._effective_policy(states_map, telemetry_by_island),
                'donor_strategy': self.donor_strategy,
                'receiver_strategy': self.receiver_strategy,
                'adaptive_checkpointing': self.adaptive_checkpointing,
                'checkpoint_strategy': self.checkpoint_strategy,
                'checkpoint_patience': self.checkpoint_patience,
                'islands': [label for label, _ in islands],
                'adjacency': adjacency,
                'constraint_handler': self.constraint_handler or 'none',
                'execution_backend_requested': self.execution_backend,
                'execution_backend_used': actual_backend,
                'n_jobs': self.n_jobs,
                'parallel_warning': backend_warning,
                'replay_manifest': replay_manifest,
            },
            island_telemetry=telemetry_by_island,
            replay_manifest=replay_manifest,
        )


def cooperative_optimize(*args, **kwargs) -> CooperativeResult:
    return CooperativeRunner(*args, **kwargs).run()


def replay_cooperative_result(result_or_manifest, target_function, objective: str | None = None, execution_backend: str = 'serial', n_jobs: int | None = None, parallel_fallback_to_serial: bool = True) -> CooperativeResult:
    manifest = result_or_manifest
    if hasattr(result_or_manifest, 'replay_manifest'):
        manifest = getattr(result_or_manifest, 'replay_manifest')
    elif isinstance(result_or_manifest, dict) and 'metadata' in result_or_manifest and isinstance(result_or_manifest['metadata'], dict):
        manifest = result_or_manifest['metadata'].get('replay_manifest', result_or_manifest)
    if not isinstance(manifest, dict):
        raise TypeError('result_or_manifest must be a CooperativeResult or a replay manifest dict')
    runner = CooperativeRunner(
        islands=copy.deepcopy(manifest['islands']),
        target_function=target_function,
        min_values=manifest['min_values'],
        max_values=manifest['max_values'],
        objective=objective or manifest.get('objective', 'min'),
        max_steps=manifest['max_steps'],
        migration_interval=manifest['migration_interval'],
        migration_size=manifest['migration_size'],
        migration_mode=manifest['migration_mode'],
        topology=manifest.get('topology', 'ring'),
        seed=manifest.get('seed'),
        topology_config=copy.deepcopy(manifest.get('topology_config') or {}),
        custom_topology=copy.deepcopy(manifest.get('custom_topology') or {}),
        migration_policy=manifest.get('migration_policy', 'push'),
        donor_strategy=manifest.get('donor_strategy', 'neighbors'),
        receiver_strategy=manifest.get('receiver_strategy', 'neighbors'),
        adaptive_checkpointing=manifest.get('adaptive_checkpointing', False),
        checkpoint_strategy=manifest.get('checkpoint_strategy', 'fixed'),
        min_migration_interval=manifest.get('min_migration_interval'),
        max_migration_interval=manifest.get('max_migration_interval'),
        checkpoint_patience=manifest.get('checkpoint_patience', 3),
        execution_backend=execution_backend,
        n_jobs=n_jobs,
        parallel_fallback_to_serial=parallel_fallback_to_serial,
    )
    return runner.run()
