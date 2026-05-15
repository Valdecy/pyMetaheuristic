"""Diagnostics for cooperative and orchestrated island optimization.

The helpers in this module are intentionally dependency-light.  They operate on
``CooperativeResult`` and ``OrchestratedCooperativeResult``-like objects using
attribute/dictionary access, so old saved result objects remain analyzable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from math import cos, pi, sin
from typing import Any, Iterable


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _objective(result: Any, objective: str | None = None) -> str:
    if objective in {"min", "max"}:
        return str(objective)
    metadata = _get(result, "metadata", {}) or {}
    replay = _get(result, "replay_manifest", {}) or metadata.get("replay_manifest", {}) or {}
    value = replay.get("objective") or metadata.get("objective")
    if value in {"min", "max"}:
        return str(value)
    checkpoints = _get(result, "checkpoints", []) or []
    if checkpoints:
        value = _get(checkpoints[0], "objective")
        if value in {"min", "max"}:
            return str(value)
    return "min"


def _is_better(new: float | None, old: float | None, objective: str) -> bool | None:
    if new is None or old is None:
        return None
    return float(new) > float(old) if objective == "max" else float(new) < float(old)


def _improvement_amount(new: float | None, old: float | None, objective: str) -> float | None:
    if new is None or old is None:
        return None
    if objective == "max":
        return float(new) - float(old)
    return float(old) - float(new)


def _labels(result: Any) -> list[str]:
    island_results = _get(result, "island_results", {}) or {}
    if island_results:
        return list(island_results.keys())
    metadata = _get(result, "metadata", {}) or {}
    labels = metadata.get("islands")
    if labels:
        return list(labels)
    checkpoints = _get(result, "checkpoints", []) or []
    if checkpoints:
        return [_get(agent, "label") for agent in (_get(checkpoints[-1], "agents", []) or []) if _get(agent, "label")]
    island_telemetry = _get(result, "island_telemetry", {}) or {}
    if island_telemetry:
        return list(island_telemetry.keys())
    return []


def _algorithm_by_label(result: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    island_results = _get(result, "island_results", {}) or {}
    for label, res in island_results.items():
        out[label] = _get(res, "algorithm_id") or _get(res, "algorithm") or _get(res, "algorithm_name")
    checkpoints = _get(result, "checkpoints", []) or []
    if checkpoints:
        for agent in _get(checkpoints[-1], "agents", []) or []:
            label = _get(agent, "label")
            if label and label not in out:
                out[label] = _get(agent, "algorithm")
    return out


def _final_fitness_by_label(result: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    island_results = _get(result, "island_results", {}) or {}
    for label, res in island_results.items():
        fit = _get(res, "best_fitness")
        out[label] = None if fit is None else float(fit)
    checkpoints = _get(result, "checkpoints", []) or []
    if checkpoints:
        for agent in _get(checkpoints[-1], "agents", []) or []:
            label = _get(agent, "label")
            if label and label not in out:
                fit = _get(agent, "best_fitness")
                out[label] = None if fit is None else float(fit)
    return out


def _iter_cooperation_events(result: Any) -> Iterable[dict[str, Any]]:
    for event in _get(result, "events", []) or []:
        data = _to_dict(event)
        if not data:
            continue
        # CooperativeResult stores source_label/target_label directly.
        if data.get("source_label") or data.get("target_label"):
            yield {
                "type": data.get("type", "migration"),
                "source_label": data.get("source_label"),
                "target_label": data.get("target_label"),
                "source_algorithm": data.get("source_algorithm"),
                "target_algorithm": data.get("target_algorithm"),
                "migrants": data.get("migrants", data.get("k", 1)),
                "policy": data.get("policy"),
                "target_fitness_before": data.get("target_fitness_before"),
                "target_fitness_after": data.get("target_fitness_after", data.get("best_fitness_after")),
                "status": data.get("status", "applied"),
                "executed": data.get("executed", True),
                "raw": data,
            }
            continue
        # Orchestrated events are serialized ActionOutcome dictionaries.
        action = data.get("action") or {}
        if isinstance(action, dict):
            yield {
                "type": action.get("type", data.get("type", "action")),
                "source_label": action.get("source_label"),
                "target_label": action.get("target_label"),
                "source_algorithm": None,
                "target_algorithm": None,
                "migrants": action.get("k", 1),
                "policy": action.get("replace_policy") or action.get("source_mode"),
                "target_fitness_before": data.get("target_fitness_before"),
                "target_fitness_after": data.get("target_fitness_after"),
                "source_fitness": data.get("source_fitness"),
                "status": data.get("status"),
                "executed": data.get("executed"),
                "message": data.get("message"),
                "raw": data,
            }


def _iter_outcomes(result: Any) -> Iterable[dict[str, Any]]:
    outcomes = _get(result, "outcomes", []) or []
    for group in outcomes:
        for outcome in group or []:
            data = _to_dict(outcome)
            action = data.get("action")
            if not isinstance(action, dict):
                action = _to_dict(action)
            data["action"] = action
            yield data
    # Fixed-orchestration wrapper stores legacy migration events in events, but
    # has no outcomes; expose those too so action_effectiveness is not empty.
    if not outcomes:
        for event in _iter_cooperation_events(result):
            yield {
                "action": {
                    "type": event.get("type") or "migration",
                    "source_label": event.get("source_label"),
                    "target_label": event.get("target_label"),
                    "k": event.get("migrants", 1),
                },
                "executed": event.get("executed", True),
                "status": event.get("status", "applied"),
                "target_fitness_before": event.get("target_fitness_before"),
                "target_fitness_after": event.get("target_fitness_after"),
                "source_fitness": event.get("source_fitness"),
            }


def _telemetry_by_label(result: Any) -> dict[str, list[Any]]:
    telemetry = _get(result, "island_telemetry", {}) or {}
    if telemetry:
        return {label: list(records or []) for label, records in telemetry.items()}
    # Orchestrated results do not currently store island_telemetry, so derive a
    # telemetry-like trace from checkpoint snapshots.
    checkpoints = _get(result, "checkpoints", []) or []
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cp in checkpoints:
        cid = _get(cp, "checkpoint_id", len(out))
        for agent in _get(cp, "agents", []) or []:
            label = _get(agent, "label")
            if not label:
                continue
            out[label].append({
                "global_step": cid,
                "label": label,
                "algorithm": _get(agent, "algorithm"),
                "step": _get(agent, "step"),
                "evaluations": _get(agent, "evaluations"),
                "best_fitness": _get(agent, "best_fitness"),
                "delta_best": _get(agent, "delta_best"),
                "stagnation_steps": _get(agent, "stagnation_steps"),
                "diversity": _get(agent, "diversity"),
                "mean_fitness": _get(agent, "mean_fitness"),
                "std_fitness": _get(agent, "std_fitness"),
                "health": _get(agent, "health"),
            })
    return dict(out)


def _event_success(event: dict[str, Any], objective: str) -> bool | None:
    before = event.get("target_fitness_before")
    after = event.get("target_fitness_after")
    return _is_better(after, before, objective)


def migration_matrix(
    result: Any,
    value: str = "migrants",
    include_zero: bool = True,
    objective: str | None = None,
) -> dict[str, dict[str, float]]:
    """Return a source-by-target matrix for island communication.

    Parameters
    ----------
    result:
        ``CooperativeResult`` or ``OrchestratedCooperativeResult``.
    value:
        ``"migrants"``/``"count"`` counts communication volume;
        ``"events"`` counts events; ``"successes"`` counts target improvements;
        ``"improvement"`` sums positive objective-consistent improvement.
    include_zero:
        Include all island labels as rows/columns even when no communication
        happened.
    objective:
        Optional objective override, ``"min"`` or ``"max"``.
    """
    objective = _objective(result, objective)
    labels = _labels(result)
    matrix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if include_zero:
        for i in labels:
            for j in labels:
                if i != j:
                    matrix[i][j] += 0.0
    for event in _iter_cooperation_events(result):
        source = event.get("source_label")
        target = event.get("target_label")
        if not source or not target or source == target:
            continue
        if value in {"migrants", "count"}:
            amount = float(event.get("migrants") or 1)
        elif value == "events":
            amount = 1.0
        elif value == "successes":
            amount = 1.0 if _event_success(event, objective) is True else 0.0
        elif value == "improvement":
            imp = _improvement_amount(event.get("target_fitness_after"), event.get("target_fitness_before"), objective)
            amount = max(0.0, float(imp)) if imp is not None else 0.0
        else:
            raise ValueError("value must be one of: 'migrants', 'count', 'events', 'successes', 'improvement'")
        matrix[source][target] += amount
    return {src: dict(targets) for src, targets in matrix.items()}


def topology_summary(result: Any) -> dict[str, Any]:
    """Summarize the island communication topology and observed links."""
    labels = _labels(result)
    metadata = _get(result, "metadata", {}) or {}
    adjacency = metadata.get("adjacency") or {}
    if not adjacency:
        adjacency = {label: [] for label in labels}
        for event in _iter_cooperation_events(result):
            src, tgt = event.get("source_label"), event.get("target_label")
            if src and tgt and tgt not in adjacency.setdefault(src, []):
                adjacency[src].append(tgt)
            if src and src not in labels:
                labels.append(src)
            if tgt and tgt not in labels:
                labels.append(tgt)
    adjacency = {label: list(dict.fromkeys(adjacency.get(label, []))) for label in labels}
    n = len(labels)
    directed_edges = [(src, tgt) for src, nbrs in adjacency.items() for tgt in nbrs if src != tgt]
    max_edges = n * (n - 1) if n > 1 else 0
    out_degree = {label: len(adjacency.get(label, [])) for label in labels}
    in_degree = {label: 0 for label in labels}
    for _src, tgt in directed_edges:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1
    observed = migration_matrix(result, value="events", include_zero=False)
    observed_edges = [(s, t) for s, row in observed.items() for t, v in row.items() if v > 0]
    return {
        "topology": metadata.get("topology", metadata.get("controller_mode", None)),
        "n_islands": n,
        "n_directed_edges": len(directed_edges),
        "density": 0.0 if max_edges == 0 else float(len(directed_edges) / max_edges),
        "average_out_degree": 0.0 if n == 0 else float(sum(out_degree.values()) / n),
        "out_degree": out_degree,
        "in_degree": in_degree,
        "hubs": [label for label, deg in out_degree.items() if deg == max(out_degree.values())] if out_degree else [],
        "isolated": [label for label in labels if out_degree.get(label, 0) == 0 and in_degree.get(label, 0) == 0],
        "adjacency": adjacency,
        "observed_edges": observed_edges,
        "observed_n_edges": len(observed_edges),
        "observed_density": 0.0 if max_edges == 0 else float(len(observed_edges) / max_edges),
    }


def island_contribution(result: Any, objective: str | None = None) -> dict[str, dict[str, Any]]:
    """Return per-island contribution diagnostics.

    The metrics combine final solution quality, telemetry improvement, and
    communication behavior.  Unknown quantities are kept as ``None`` rather than
    guessed.
    """
    objective = _objective(result, objective)
    labels = _labels(result)
    algorithms = _algorithm_by_label(result)
    final_fit = _final_fitness_by_label(result)
    telemetry = _telemetry_by_label(result)
    mat_migrants = migration_matrix(result, value="migrants", include_zero=True, objective=objective)
    mat_success = migration_matrix(result, value="successes", include_zero=True, objective=objective)
    mat_events = migration_matrix(result, value="events", include_zero=True, objective=objective)
    known_outcomes_by_source: dict[str, int] = defaultdict(int)
    known_outcomes_by_target: dict[str, int] = defaultdict(int)
    for event in _iter_cooperation_events(result):
        success = _event_success(event, objective)
        if success is None:
            continue
        src = event.get("source_label")
        tgt = event.get("target_label")
        if src:
            known_outcomes_by_source[src] += 1
        if tgt:
            known_outcomes_by_target[tgt] += 1
    out: dict[str, dict[str, Any]] = {}
    valid_fits = {k: v for k, v in final_fit.items() if v is not None}
    ranked = sorted(valid_fits, key=valid_fits.get, reverse=(objective == "max"))
    rank = {label: i + 1 for i, label in enumerate(ranked)}

    for label in labels:
        records = telemetry.get(label, [])
        improvements = []
        diversities = []
        health_values = []
        last_stagnation = None
        for rec in records:
            delta = _get(rec, "delta_best")
            if delta is not None:
                imp = -float(delta) if objective == "min" else float(delta)
                if imp > 0:
                    improvements.append(imp)
            diversity = _get(rec, "diversity")
            if diversity is not None:
                try:
                    diversities.append(float(diversity))
                except Exception:
                    pass
            health = _get(rec, "health")
            if health is not None:
                health_values.append(health)
            st = _get(rec, "stagnation_steps")
            if st is not None:
                last_stagnation = int(st)
        donated_migrants = sum(mat_migrants.get(label, {}).values())
        received_migrants = sum(row.get(label, 0.0) for row in mat_migrants.values())
        donated_events = sum(mat_events.get(label, {}).values())
        received_events = sum(row.get(label, 0.0) for row in mat_events.values())
        donation_successes = sum(mat_success.get(label, {}).values())
        received_successes = sum(row.get(label, 0.0) for row in mat_success.values())
        known_donations = int(known_outcomes_by_source.get(label, 0))
        known_receipts = int(known_outcomes_by_target.get(label, 0))
        out[label] = {
            "label": label,
            "algorithm": algorithms.get(label),
            "final_fitness": final_fit.get(label),
            "rank": rank.get(label),
            "is_global_best": rank.get(label) == 1 if rank else False,
            "improvement_total": float(sum(improvements)) if improvements else 0.0,
            "improvement_events": int(len(improvements)),
            "mean_diversity": None if not diversities else float(sum(diversities) / len(diversities)),
            "last_diversity": None if not diversities else float(diversities[-1]),
            "last_health": None if not health_values else health_values[-1],
            "last_stagnation_steps": last_stagnation,
            "donated_events": int(donated_events),
            "received_events": int(received_events),
            "donated_migrants": int(donated_migrants),
            "received_migrants": int(received_migrants),
            "donation_successes": int(donation_successes),
            "received_successes": int(received_successes),
            "known_donation_outcomes": known_donations,
            "known_receiver_outcomes": known_receipts,
            "donor_success_rate": None if known_donations == 0 else float(donation_successes / known_donations),
            "receiver_success_rate": None if known_receipts == 0 else float(received_successes / known_receipts),
        }
    return out


def island_roles(result: Any, objective: str | None = None) -> dict[str, dict[str, Any]]:
    """Classify islands into interpretable diagnostic roles."""
    contrib = island_contribution(result, objective=objective)
    if not contrib:
        return {}
    donation_counts = [v["donated_events"] for v in contrib.values()]
    received_counts = [v["received_events"] for v in contrib.values()]
    diversities = [v["mean_diversity"] for v in contrib.values() if v["mean_diversity"] is not None]
    median_div = None
    if diversities:
        s = sorted(diversities)
        median_div = s[len(s) // 2]
    max_donated = max(donation_counts or [0])
    max_received = max(received_counts or [0])
    out: dict[str, dict[str, Any]] = {}
    for label, c in contrib.items():
        evidence: list[str] = []
        role = "neutral"
        health = str(c.get("last_health") or "").lower()
        if c.get("is_global_best"):
            role = "global_best_refiner"
            evidence.append("best final fitness")
        if c.get("donated_events", 0) > 0 and c.get("donated_events", 0) == max_donated:
            if (c.get("donor_success_rate") or 0.0) > 0:
                role = "effective_donor"
                evidence.append("highest donation volume with successful transfers")
            elif role == "neutral":
                role = "active_donor"
                evidence.append("highest donation volume")
        if c.get("received_events", 0) > 0 and c.get("received_events", 0) == max_received:
            if (c.get("receiver_success_rate") or 0.0) > 0:
                if role == "neutral":
                    role = "effective_receiver"
                evidence.append("frequent receiver with successful incoming transfers")
            elif role == "neutral":
                role = "receiver"
                evidence.append("frequent receiver")
        if median_div is not None and c.get("mean_diversity") is not None and c["mean_diversity"] >= median_div and c.get("improvement_events", 0) > 0:
            if role == "neutral":
                role = "explorer"
            evidence.append("above-median diversity with improvements")
        if c.get("last_stagnation_steps") is not None and c["last_stagnation_steps"] >= 5:
            role = "stagnating" if role == "neutral" else role
            evidence.append("persistent stagnation")
        if health in {"poor", "frozen", "stagnating"}:
            role = "stagnating" if role == "neutral" else role
            evidence.append(f"health={c.get('last_health')}")
        if role == "neutral" and c.get("improvement_events", 0) > 0:
            role = "local_improver"
            evidence.append("improved during the run")
        if role == "neutral" and c.get("donated_events", 0) == 0 and c.get("received_events", 0) == 0:
            role = "isolated"
            evidence.append("no observed communication")
        out[label] = {
            "label": label,
            "algorithm": c.get("algorithm"),
            "role": role,
            "evidence": evidence,
            "metrics": c,
        }
    return out


def action_effectiveness(result: Any, objective: str | None = None) -> dict[str, Any]:
    """Summarize intervention and migration outcomes."""
    objective = _objective(result, objective)
    by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "applied": 0,
        "skipped": 0,
        "failed": 0,
        "downgraded": 0,
        "improved": 0,
        "worsened": 0,
        "unchanged": 0,
        "unknown_effect": 0,
        "total_positive_improvement": 0.0,
    })
    by_pair: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "applied": 0,
        "improved": 0,
        "unknown_effect": 0,
        "total_positive_improvement": 0.0,
    })
    total = 0
    for outcome in _iter_outcomes(result):
        action = outcome.get("action") or {}
        action_type = action.get("type") or "migration"
        source = action.get("source_label")
        target = action.get("target_label")
        status = outcome.get("status") or "unknown"
        executed = bool(outcome.get("executed", status in {"applied", "downgraded"}))
        before = outcome.get("target_fitness_before")
        after = outcome.get("target_fitness_after")
        success = _is_better(after, before, objective)
        imp = _improvement_amount(after, before, objective)
        total += 1
        row = by_type[action_type]
        row["count"] += 1
        if status in row:
            row[status] += 1
        elif executed:
            row["applied"] += 1
        if success is True:
            row["improved"] += 1
            row["total_positive_improvement"] += max(0.0, float(imp or 0.0))
        elif success is False:
            if imp is not None and imp < 0:
                row["worsened"] += 1
            else:
                row["unchanged"] += 1
        else:
            row["unknown_effect"] += 1
        if source or target:
            key = f"{source or '-'}->{target or '-'}"
            prow = by_pair[key]
            prow["count"] += 1
            if executed:
                prow["applied"] += 1
            if success is True:
                prow["improved"] += 1
                prow["total_positive_improvement"] += max(0.0, float(imp or 0.0))
            elif success is None:
                prow["unknown_effect"] += 1
    for table in (by_type, by_pair):
        for row in table.values():
            known = int(row.get("improved", 0)) + int(row.get("worsened", 0)) + int(row.get("unchanged", 0))
            row["known_effects"] = known
            row["success_rate"] = None if known == 0 else float(row["improved"] / known)
            row["applied_rate"] = None if row["count"] == 0 else float(row["applied"] / row["count"])
    return {
        "objective": objective,
        "total_actions": total,
        "by_type": dict(by_type),
        "by_pair": dict(by_pair),
    }


def diagnostics_summary(result: Any, objective: str | None = None) -> dict[str, Any]:
    """Compact all Phase-II diagnostics into one dictionary."""
    return {
        "topology": topology_summary(result),
        "migration_matrix": migration_matrix(result, objective=objective),
        "island_contribution": island_contribution(result, objective=objective),
        "island_roles": island_roles(result, objective=objective),
        "action_effectiveness": action_effectiveness(result, objective=objective),
    }


def plot_migration_network(result: Any, ax: Any = None, value: str = "migrants", node_size: float = 800.0, **kwargs: Any):
    """Plot a lightweight circular island communication network.

    Returns the Matplotlib axis.  Matplotlib is imported lazily so the core
    package keeps a small import footprint.
    """
    import matplotlib.pyplot as plt

    labels = _labels(result)
    matrix = migration_matrix(result, value=value, include_zero=False)
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 5)))
    ax.set_aspect("equal")
    ax.axis("off")
    n = max(1, len(labels))
    pos = {label: (cos(2 * pi * i / n), sin(2 * pi * i / n)) for i, label in enumerate(labels)}
    max_weight = max([w for row in matrix.values() for w in row.values()] or [1.0])
    for src, row in matrix.items():
        for tgt, weight in row.items():
            if weight <= 0 or src not in pos or tgt not in pos:
                continue
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            lw = 0.5 + 4.0 * float(weight) / float(max_weight)
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", linewidth=lw, alpha=0.55, shrinkA=18, shrinkB=18),
            )
    for label, (x, y) in pos.items():
        ax.scatter([x], [y], s=node_size, zorder=3)
        ax.text(x, y, label, ha="center", va="center", zorder=4)
    ax.set_title(f"Island communication network ({value})")
    return ax


def plot_island_fitness(result: Any, ax: Any = None, objective: str | None = None, **kwargs: Any):
    """Plot best-fitness traces by island using telemetry/checkpoints/history."""
    import matplotlib.pyplot as plt

    telemetry = _telemetry_by_label(result)
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))
    if telemetry:
        for label, records in telemetry.items():
            xs = []
            ys = []
            for idx, rec in enumerate(records):
                fit = _get(rec, "best_fitness")
                if fit is None:
                    continue
                xs.append(_get(rec, "global_step", idx))
                ys.append(float(fit))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
    else:
        history = _get(result, "history", []) or []
        by_label: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, obs in enumerate(history):
            label = _get(obs, "label")
            fit = _get(obs, "best_fitness")
            if label and fit is not None:
                by_label[label].append((i, float(fit)))
        for label, series in by_label.items():
            ax.plot([x for x, _ in series], [y for _, y in series], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("global step / checkpoint")
    ax.set_ylabel("best fitness")
    ax.set_title("Island best-fitness traces")
    if ax.lines:
        ax.legend()
    return ax


__all__ = [
    "migration_matrix",
    "topology_summary",
    "island_contribution",
    "island_roles",
    "action_effectiveness",
    "diagnostics_summary",
    "plot_migration_network",
    "plot_island_fitness",
]
