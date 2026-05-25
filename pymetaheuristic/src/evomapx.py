"""
EvoMapX-style explainability for pyMetaheuristic.

This module provides a package-level, result-agnostic implementation of the
three explainability views commonly associated with EvoMapX-like analysis:

1. Operator/Island Attribution Matrix (OAM/IAM)
2. Convergence Driver Score (CDS)
3. Population Evolution Graph (PEG)

The implementation is deliberately generic. It works with ordinary
OptimizationResult objects, CooperativeResult objects, and
OrchestratedCooperativeResult objects. When algorithm-native operator logs are
not available, the attribution unit falls back to the algorithm, island, or
agent label. When population ancestry is not available, PEG edges are inferred
conservatively from consecutive population snapshots by nearest-neighbour
continuity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


_DARK_BG = "#0d1117"
_PANEL_BG = "#161b22"
_GRID_CLR = "#21262d"
_TEXT_CLR = "#e6edf3"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_AMBER = "#e3b341"
_RED = "#f85149"
_PURPLE = "#a371f7"

_DISCRETE_PALETTE = [
    "#3BC9DB", "#F0A500", "#3FB950", "#F85149",
    "#C084FC", "#FB923C", "#34D399", "#60A5FA",
    "#F472B6", "#A3E635", "#FBBF24", "#818CF8",
]

_FONT = dict(family="Inter, Arial, sans-serif", color=_TEXT_CLR, size=13)
_LAYOUT_BASE = dict(
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_PANEL_BG,
    font=_FONT,
    margin=dict(l=70, r=40, t=75, b=65),
    hoverlabel=dict(bgcolor=_PANEL_BG, font_color=_TEXT_CLR, bordercolor=_GRID_CLR),
)
_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=_GRID_CLR,
    zeroline=False,
    linecolor=_GRID_CLR,
    tickfont=dict(color=_TEXT_CLR, size=11),
    title_font=dict(color=_TEXT_CLR, size=13),
)


@dataclass
class EvoMapXNode:
    """Node in a Population Evolution Graph."""

    id: str
    step: int | None = None
    label: str | None = None
    algorithm: str | None = None
    fitness: float | None = None
    position: list[float] | None = None
    operator: str | None = None
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvoMapXEdge:
    """Directed edge in a Population Evolution Graph."""

    source: str
    target: str
    kind: str = "transition"
    step: int | None = None
    label: str | None = None
    algorithm: str | None = None
    operator: str | None = None
    improvement: float | None = None
    distance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvoMapXReport:
    """Container returned by :func:`evomapx_analysis`."""

    objective: str
    level: str
    steps: list[int]
    labels: list[str]
    raw_attribution: dict[str, list[float]]
    normalized_attribution: dict[str, list[float]]
    cds_raw: dict[str, float]
    cds_normalized: dict[str, float]
    activity: dict[str, dict[str, float]]
    peg: dict[str, Any]
    migration_attribution: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dataframe(self, normalized: bool = True) -> pd.DataFrame:
        """Return the attribution matrix as a tidy pandas DataFrame."""
        matrix = self.normalized_attribution if normalized else self.raw_attribution
        rows: list[dict[str, Any]] = []
        for label in self.labels:
            values = matrix.get(label, [])
            for idx, value in enumerate(values):
                step = self.steps[idx] if idx < len(self.steps) else idx
                rows.append({"step": step, "label": label, "attribution": float(value)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal data helpers
# ---------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(getattr(obj, "__dict__", {}) or {})
    return obj


def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalar values into JSON-serializable objects."""
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _objective(result: Any, objective: str | None = None) -> str:
    if objective in {"min", "max"}:
        return str(objective)
    metadata = _get(result, "metadata", {}) or {}
    obj = metadata.get("objective") or metadata.get("problem_objective")
    if obj in {"min", "max"}:
        return str(obj)
    # Cooperative runners store objective in replay manifests more consistently.
    replay = metadata.get("replay_manifest") or _get(result, "replay_manifest", {}) or {}
    obj = replay.get("objective") if isinstance(replay, dict) else None
    return str(obj) if obj in {"min", "max"} else "min"


def _improvement(before: float | None, after: float | None, objective: str) -> float | None:
    if before is None or after is None:
        return None
    try:
        b = float(before)
        a = float(after)
    except Exception:
        return None
    if not (math.isfinite(b) and math.isfinite(a)):
        return None
    return b - a if objective == "min" else a - b


def _positive(value: float | None, positive_only: bool = True) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, v) if positive_only else v


def _event_dicts(result: Any) -> list[dict[str, Any]]:
    return [_to_plain(e) for e in (_get(result, "events", []) or [])]


def _cooperative_telemetry_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    telemetry = _get(result, "island_telemetry", {}) or {}
    for label, records in telemetry.items():
        for rec in records:
            row = _to_plain(rec)
            if isinstance(row, dict):
                row.setdefault("label", label)
                rows.append(row)
    return rows


def _history_rows(result: Any) -> list[dict[str, Any]]:
    return [_to_plain(row) for row in (_get(result, "history", []) or []) if isinstance(_to_plain(row), dict)]


def _candidate_unit(row: dict[str, Any], default: str, level: str) -> str:
    candidates: list[Any] = []
    if level in {"operator", "auto"}:
        candidates.extend([
            row.get("operator"), row.get("op"), row.get("action"),
            row.get("move"), row.get("strategy"), row.get("controller_action"),
        ])
    if level in {"island", "agent", "auto"}:
        candidates.extend([row.get("label"), row.get("agent"), row.get("island")])
    if level in {"algorithm", "auto"}:
        candidates.extend([row.get("algorithm"), row.get("algorithm_id")])
    for item in candidates:
        if item not in {None, ""}:
            return str(item)
    return str(default)


# ---------------------------------------------------------------------------
# Attribution matrix and CDS
# ---------------------------------------------------------------------------


def attribution_records(
    result: Any,
    objective: str | None = None,
    level: str = "auto",
    positive_only: bool = True,
) -> list[dict[str, Any]]:
    """Return tidy attribution records extracted from a result object.

    Parameters
    ----------
    result:
        Any pyMetaheuristic result-like object. Ordinary runs, cooperative
        runs, and orchestrated cooperative runs are supported.
    objective:
        Optional objective override: ``"min"`` or ``"max"``.
    level:
        Attribution unit. Use ``"auto"``, ``"operator"``, ``"island"``,
        ``"agent"`` or ``"algorithm"``.
    positive_only:
        If True, only objective-consistent improvements contribute.
    """
    objective = _objective(result, objective)
    level = str(level or "auto").lower()
    records: list[dict[str, Any]] = []

    # 1) Cooperative island telemetry is the richest generic source for the
    # article's heterogeneous-island experiments. If an engine, such as CEM,
    # exposes native operator contributions, split one telemetry row into one
    # OAM/CDS record per operator. Otherwise use the island as the unit.
    telemetry_rows = _cooperative_telemetry_rows(result)
    if telemetry_rows:
        has_operator_contribs = any(isinstance(row.get("operator_contributions"), dict) and row.get("operator_contributions") for row in telemetry_rows)
        if level == "operator" and not has_operator_contribs:
            # Fall through to global history, which may contain raw observations
            # with operator fields even when the compact telemetry object does not.
            pass
        elif (level == "operator" or (level == "auto" and has_operator_contribs)) and has_operator_contribs:
            for row in telemetry_rows:
                contribs = row.get("operator_contributions") or {}
                if not isinstance(contribs, dict) or not contribs:
                    continue
                base_label = str(row.get("label") or row.get("algorithm") or "island")
                for op, value in contribs.items():
                    label = f"{base_label}:{op}" if level == "operator" else str(op)
                    records.append({
                        "step": int(row.get("global_step", row.get("step", len(records)))),
                        "label": label,
                        "algorithm": row.get("algorithm"),
                        "raw_improvement": float(value or 0.0),
                        "positive_improvement": _positive(value, positive_only=positive_only),
                        "source": "island_operator_telemetry",
                        "metadata": {**dict(row), "operator": str(op)},
                    })
            if records:
                return records
        else:
            for row in telemetry_rows:
                delta = row.get("delta_best")
                # delta_best = current_best - previous_best in CooperativeRunner.
                imp = None if delta is None else (-float(delta) if objective == "min" else float(delta))
                records.append({
                    "step": int(row.get("global_step", row.get("step", len(records)))),
                    "label": _candidate_unit(row, row.get("label", "island"), "island" if level == "auto" else level),
                    "algorithm": row.get("algorithm"),
                    "raw_improvement": float(imp) if imp is not None else 0.0,
                    "positive_improvement": _positive(imp, positive_only=positive_only),
                    "source": "island_telemetry",
                    "metadata": dict(row),
                })
            return records

    # 2) Orchestrated/cooperative global history often contains rows with a
    # label and per-agent best_fitness. Compute within-label deltas.
    history = _history_rows(result)
    labelled = [r for r in history if r.get("label") is not None and r.get("best_fitness") is not None]
    if labelled:
        prev: dict[str, float] = {}
        for row in labelled:
            label = str(row.get("label"))
            curr = row.get("best_fitness")
            before = prev.get(label)
            imp = _improvement(before, curr, objective) if before is not None else 0.0
            prev[label] = float(curr)
            records.append({
                "step": int(row.get("global_step", row.get("step", len(records)))),
                "label": _candidate_unit(row, label, "island" if level == "auto" else level),
                "algorithm": row.get("algorithm"),
                "raw_improvement": float(imp or 0.0),
                "positive_improvement": _positive(imp, positive_only=positive_only),
                "source": "history_by_label",
                "metadata": dict(row),
            })
        return records

    # 3) Explicit records captured by EvoMapXRecorder.
    metadata_records = (_get(result, "metadata", {}) or {}).get("evomapx_records", [])
    if metadata_records:
        alg = _get(result, "algorithm_id", None) or _get(result, "algorithm", None) or "algorithm"
        for i, row0 in enumerate(metadata_records):
            row = _to_plain(row0)
            if not isinstance(row, dict):
                continue
            imp = row.get("raw_improvement", row.get("positive_improvement", 0.0))
            records.append({
                "step": int(row.get("step", i + 1)),
                "label": _candidate_unit(row, str(alg), "operator" if level == "auto" else level),
                "algorithm": row.get("algorithm", alg),
                "raw_improvement": float(imp or 0.0),
                "positive_improvement": _positive(imp, positive_only=positive_only),
                "source": "evomapx_records",
                "metadata": dict(row),
            })
        return records

    # 4) Ordinary single-run history. Attribution unit is the algorithm unless
    # the engine provides operator/action keys or a native operator_contributions
    # dictionary in observations. CEM uses this path for sampling/elite-selection
    # attribution.
    if history:
        alg = _get(result, "algorithm_id", None) or _get(result, "algorithm", None) or "algorithm"
        prev_fit: float | None = None
        for i, row in enumerate(history):
            curr = row.get("global_best_fitness", row.get("best_fitness"))
            if curr is None:
                continue
            contribs = row.get("operator_contributions") or row.get("evomapx_operator_contributions")
            if isinstance(contribs, dict) and contribs and level in {"operator", "auto"}:
                for op, value in contribs.items():
                    records.append({
                        "step": int(row.get("step", i + 1)),
                        "label": str(op),
                        "algorithm": alg,
                        "raw_improvement": float(value or 0.0),
                        "positive_improvement": _positive(value, positive_only=positive_only),
                        "source": "operator_contributions",
                        "metadata": {**dict(row), "operator": str(op)},
                    })
                prev_fit = float(curr)
                continue
            imp = _improvement(prev_fit, curr, objective) if prev_fit is not None else 0.0
            prev_fit = float(curr)
            records.append({
                "step": int(row.get("step", i + 1)),
                "label": _candidate_unit(row, str(alg), "algorithm" if level == "auto" else level),
                "algorithm": alg,
                "raw_improvement": float(imp or 0.0),
                "positive_improvement": _positive(imp, positive_only=positive_only),
                "source": "history",
                "metadata": dict(row),
            })
        return records

    # 5) Events are a fallback, useful when only migration logs exist.
    for event in _event_dicts(result):
        before = event.get("target_fitness_before")
        after = event.get("target_fitness_after", event.get("best_fitness_after"))
        imp = _improvement(before, after, objective)
        label = event.get("source_label") or event.get("source_algorithm") or "event"
        records.append({
            "step": int(event.get("global_step", len(records))),
            "label": str(label),
            "algorithm": event.get("source_algorithm"),
            "raw_improvement": float(imp or 0.0),
            "positive_improvement": _positive(imp, positive_only=positive_only),
            "source": "events",
            "metadata": dict(event),
        })
    return records


def operator_attribution_matrix(
    result: Any,
    objective: str | None = None,
    level: str = "auto",
    normalize: bool = True,
    positive_only: bool = True,
) -> dict[str, Any]:
    """Compute the EvoMapX-style attribution matrix.

    The returned dictionary contains a raw matrix and a column-normalized matrix.
    Rows are attribution units and columns are time steps.
    """
    records = attribution_records(result, objective=objective, level=level, positive_only=positive_only)
    if not records:
        return {
            "objective": _objective(result, objective),
            "level": level,
            "steps": [],
            "labels": [],
            "raw": {},
            "normalized": {},
            "records": [],
        }

    steps = sorted({int(r["step"]) for r in records})
    labels = sorted({str(r["label"]) for r in records})
    step_idx = {s: i for i, s in enumerate(steps)}
    raw = {label: [0.0 for _ in steps] for label in labels}
    for rec in records:
        label = str(rec["label"])
        raw[label][step_idx[int(rec["step"])]] += float(rec.get("positive_improvement", 0.0))

    normalized_matrix = {label: list(vals) for label, vals in raw.items()}
    if normalize:
        for j in range(len(steps)):
            total = sum(max(0.0, raw[label][j]) for label in labels)
            if total > 0:
                for label in labels:
                    normalized_matrix[label][j] = max(0.0, raw[label][j]) / total
            else:
                for label in labels:
                    normalized_matrix[label][j] = 0.0

    return {
        "objective": _objective(result, objective),
        "level": level,
        "steps": steps,
        "labels": labels,
        "raw": raw,
        "normalized": normalized_matrix,
        "records": records,
    }


def convergence_driver_score(
    attribution: dict[str, Any] | EvoMapXReport,
    normalized: bool = False,
    normalize_scores: bool = True,
) -> dict[str, float]:
    """Compute Convergence Driver Score from an attribution matrix."""
    if isinstance(attribution, EvoMapXReport):
        matrix = attribution.normalized_attribution if normalized else attribution.raw_attribution
    else:
        matrix = attribution.get("normalized" if normalized else "raw", {}) or {}
    scores = {str(label): float(np.nansum(values)) for label, values in matrix.items()}
    if normalize_scores:
        total = sum(max(0.0, v) for v in scores.values())
        if total > 0:
            return {label: max(0.0, val) / total for label, val in scores.items()}
    return scores


def attribution_activity(attribution: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return activity, persistence, and intensity metrics for each unit."""
    raw = attribution.get("raw", {}) or {}
    out: dict[str, dict[str, float]] = {}
    n_steps = max(1, len(attribution.get("steps", []) or []))
    for label, values in raw.items():
        vals = np.asarray(values, dtype=float)
        nz = np.where(vals > 0)[0]
        total = float(np.nansum(vals))
        active = int(len(nz))
        span = int(nz[-1] - nz[0] + 1) if active else 0
        out[str(label)] = {
            "total_contribution": total,
            "mean_contribution": float(np.nanmean(vals)) if vals.size else 0.0,
            "max_contribution": float(np.nanmax(vals)) if vals.size else 0.0,
            "active_steps": float(active),
            "activity_rate": float(active / n_steps),
            "influence_span": float(span),
            "influence_density": float(active / span) if span > 0 else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# PEG construction
# ---------------------------------------------------------------------------


def _iter_population_snapshots(result: Any) -> Iterable[tuple[str, str, list[dict[str, Any]]]]:
    """Yield (label, algorithm, snapshots) triples from result-like objects."""
    direct = _get(result, "population_snapshots", None)
    if direct:
        label = _get(result, "algorithm_id", None) or _get(result, "algorithm", None) or "algorithm"
        yield str(label), str(label), list(direct)
    for label, sub_result in (_get(result, "island_results", {}) or {}).items():
        snapshots = _get(sub_result, "population_snapshots", None)
        if snapshots:
            alg = _get(sub_result, "algorithm_id", None) or str(label)
            yield str(label), str(alg), list(snapshots)


def population_evolution_graph(
    result: Any,
    objective: str | None = None,
    max_nodes: int = 5000,
    infer_edges: bool = True,
    include_migration_edges: bool = True,
) -> dict[str, Any]:
    """Build a generic Population Evolution Graph.

    Exact ancestry is used only if the result/snapshot records provide parent
    identifiers. Otherwise, consecutive snapshots are connected by nearest
    neighbours, which should be interpreted as continuity, not biological
    parentage.
    """
    objective = _objective(result, objective)
    nodes: list[EvoMapXNode] = []
    edges: list[EvoMapXEdge] = []

    for label, algorithm, snapshots in _iter_population_snapshots(result):
        previous_ids: list[str] = []
        previous_pos: np.ndarray | None = None
        previous_fit: list[float | None] = []
        for snap_idx, snap in enumerate(snapshots):
            step = int(snap.get("step", snap_idx))
            pop = list(snap.get("population", []) or [])
            current_ids: list[str] = []
            current_pos: list[list[float]] = []
            current_fit: list[float | None] = []
            for idx, cand in enumerate(pop):
                if len(nodes) >= max_nodes:
                    break
                pos = cand.get("position") if isinstance(cand, dict) else None
                fit = cand.get("fitness") if isinstance(cand, dict) else None
                node_id = str(cand.get("id") or f"{label}:s{step}:i{idx}") if isinstance(cand, dict) else f"{label}:s{step}:i{idx}"
                parents = cand.get("parent_ids", cand.get("parents", [])) if isinstance(cand, dict) else []
                operator = cand.get("operator", cand.get("op")) if isinstance(cand, dict) else None
                nodes.append(EvoMapXNode(
                    id=node_id,
                    step=step,
                    label=label,
                    algorithm=algorithm,
                    fitness=None if fit is None else float(fit),
                    position=None if pos is None else [float(x) for x in pos],
                    operator=None if operator is None else str(operator),
                    role="population",
                ))
                current_ids.append(node_id)
                current_fit.append(None if fit is None else float(fit))
                if pos is not None:
                    current_pos.append([float(x) for x in pos])
                else:
                    current_pos.append([])
                if parents:
                    for parent in parents:
                        edges.append(EvoMapXEdge(
                            source=str(parent),
                            target=node_id,
                            kind="parent",
                            step=step,
                            label=label,
                            algorithm=algorithm,
                            operator=None if operator is None else str(operator),
                        ))
            if len(nodes) >= max_nodes:
                break

            if infer_edges and previous_ids and current_ids and current_pos and all(current_pos):
                curr_pos = np.asarray(current_pos, dtype=float)
                if previous_pos is not None and previous_pos.size and curr_pos.ndim == 2:
                    for c_idx, cvec in enumerate(curr_pos):
                        d = np.linalg.norm(previous_pos - cvec[None, :], axis=1)
                        p_idx = int(np.argmin(d))
                        before = previous_fit[p_idx] if p_idx < len(previous_fit) else None
                        after = current_fit[c_idx] if c_idx < len(current_fit) else None
                        edges.append(EvoMapXEdge(
                            source=previous_ids[p_idx],
                            target=current_ids[c_idx],
                            kind="nearest_snapshot",
                            step=step,
                            label=label,
                            algorithm=algorithm,
                            improvement=_improvement(before, after, objective),
                            distance=float(d[p_idx]),
                        ))
            previous_ids = current_ids
            previous_fit = current_fit
            previous_pos = np.asarray(current_pos, dtype=float) if current_pos and all(current_pos) else None

    if include_migration_edges:
        for i, event in enumerate(_event_dicts(result)):
            source = event.get("source_label") or event.get("source_algorithm")
            target = event.get("target_label") or event.get("target_algorithm")
            if source and target:
                # Add event-level pseudo-nodes so migration is visible even when
                # no population snapshots were stored.
                s_id = f"migration:{i}:source:{source}"
                t_id = f"migration:{i}:target:{target}"
                step = int(event.get("global_step", i))
                before = event.get("target_fitness_before")
                after = event.get("target_fitness_after", event.get("best_fitness_after"))
                nodes.append(EvoMapXNode(
                    id=s_id,
                    step=step,
                    label=str(source),
                    algorithm=event.get("source_algorithm"),
                    fitness=event.get("source_fitness"),
                    role="migration_source",
                    metadata=dict(event),
                ))
                nodes.append(EvoMapXNode(
                    id=t_id,
                    step=step,
                    label=str(target),
                    algorithm=event.get("target_algorithm"),
                    fitness=after,
                    role="migration_target",
                    metadata=dict(event),
                ))
                edges.append(EvoMapXEdge(
                    source=s_id,
                    target=t_id,
                    kind="migration",
                    step=step,
                    label=f"{source}->{target}",
                    algorithm=event.get("source_algorithm"),
                    improvement=_improvement(before, after, objective),
                    metadata=dict(event),
                ))

    node_dicts = [asdict(n) for n in nodes[:max_nodes]]
    node_ids = {n["id"] for n in node_dicts}
    edge_dicts = [asdict(e) for e in edges if e.source in node_ids and e.target in node_ids]
    return {
        "objective": objective,
        "nodes": node_dicts,
        "edges": edge_dicts,
        "n_nodes": len(node_dicts),
        "n_edges": len(edge_dicts),
        "edge_types": {str(k): int(v) for k, v in (pd.Series([e["kind"] for e in edge_dicts]).value_counts().items())} if edge_dicts else {},
        "inferred_edges": bool(infer_edges),
        "note": "nearest_snapshot edges are continuity approximations unless parent_ids are available.",
    }


def peg_summary(peg: dict[str, Any]) -> dict[str, Any]:
    """Summarize PEG structure without requiring networkx."""
    nodes = peg.get("nodes", []) or []
    edges = peg.get("edges", []) or []
    n = len(nodes)
    m = len(edges)
    labels = sorted({str(nod.get("label")) for nod in nodes if nod.get("label") is not None})
    algorithms = sorted({str(nod.get("algorithm")) for nod in nodes if nod.get("algorithm") is not None})
    edge_types = {str(k): int(v) for k, v in (pd.Series([e.get("kind", "edge") for e in edges]).value_counts().items())} if edges else {}
    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s is not None:
            out_degree[str(s)] = out_degree.get(str(s), 0) + 1
        if t is not None:
            in_degree[str(t)] = in_degree.get(str(t), 0) + 1
    return {
        "n_nodes": n,
        "n_edges": m,
        "density_directed": float(m / (n * (n - 1))) if n > 1 else 0.0,
        "labels": labels,
        "algorithms": algorithms,
        "edge_types": edge_types,
        "mean_in_degree": float(np.mean(list(in_degree.values()))) if in_degree else 0.0,
        "mean_out_degree": float(np.mean(list(out_degree.values()))) if out_degree else 0.0,
    }


# ---------------------------------------------------------------------------
# Migration attribution
# ---------------------------------------------------------------------------


def migration_attribution(result: Any, objective: str | None = None) -> dict[str, Any]:
    """Summarize migration-driven improvements source -> target."""
    objective = _objective(result, objective)
    rows: list[dict[str, Any]] = []
    for event in _event_dicts(result):
        source = event.get("source_label") or event.get("source_algorithm")
        target = event.get("target_label") or event.get("target_algorithm")
        if not source or not target:
            continue
        before = event.get("target_fitness_before")
        after = event.get("target_fitness_after", event.get("best_fitness_after"))
        imp = _improvement(before, after, objective)
        rows.append({
            "step": event.get("global_step"),
            "source": str(source),
            "target": str(target),
            "source_algorithm": event.get("source_algorithm"),
            "target_algorithm": event.get("target_algorithm"),
            "migrants": int(event.get("migrants") or 1),
            "improvement": _positive(imp, positive_only=True),
            "raw_improvement": 0.0 if imp is None else float(imp),
            "success": _positive(imp, positive_only=True) > 0,
            "policy": event.get("policy"),
            "metadata": dict(event),
        })
    matrix: dict[str, dict[str, float]] = {}
    for row in rows:
        matrix.setdefault(row["source"], {}).setdefault(row["target"], 0.0)
        matrix[row["source"]][row["target"]] += float(row["improvement"])
    return {
        "objective": objective,
        "rows": rows,
        "matrix": matrix,
        "n_events": len(rows),
        "n_successes": int(sum(1 for r in rows if r["success"])),
        "total_migration_improvement": float(sum(r["improvement"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# Full analysis and explanations
# ---------------------------------------------------------------------------


def evomapx_analysis(
    result: Any,
    objective: str | None = None,
    level: str = "auto",
    normalize: bool = True,
    positive_only: bool = True,
    build_peg: bool = True,
    max_peg_nodes: int = 5000,
) -> EvoMapXReport:
    """Compute a complete EvoMapX-style report for a result object."""
    objective = _objective(result, objective)
    oam = operator_attribution_matrix(
        result,
        objective=objective,
        level=level,
        normalize=normalize,
        positive_only=positive_only,
    )
    cds_raw = convergence_driver_score(oam, normalized=False, normalize_scores=False)
    cds_norm = convergence_driver_score(oam, normalized=False, normalize_scores=True)
    activity = attribution_activity(oam)
    peg = population_evolution_graph(result, objective=objective, max_nodes=max_peg_nodes) if build_peg else {
        "objective": objective, "nodes": [], "edges": [], "n_nodes": 0, "n_edges": 0, "edge_types": {},
    }
    mig = migration_attribution(result, objective=objective)
    summary = {
        "top_driver": max(cds_norm.items(), key=lambda kv: kv[1])[0] if cds_norm else None,
        "top_driver_score": max(cds_norm.values()) if cds_norm else 0.0,
        "n_attribution_units": len(oam.get("labels", []) or []),
        "n_attribution_steps": len(oam.get("steps", []) or []),
        "total_positive_improvement": float(sum(cds_raw.values())) if cds_raw else 0.0,
        "peg": peg_summary(peg),
        "migration": {
            "n_events": mig.get("n_events", 0),
            "n_successes": mig.get("n_successes", 0),
            "total_improvement": mig.get("total_migration_improvement", 0.0),
        },
    }
    return EvoMapXReport(
        objective=objective,
        level=level,
        steps=oam.get("steps", []),
        labels=oam.get("labels", []),
        raw_attribution=oam.get("raw", {}),
        normalized_attribution=oam.get("normalized", {}),
        cds_raw=cds_raw,
        cds_normalized=cds_norm,
        activity=activity,
        peg=peg,
        migration_attribution=mig,
        summary=summary,
    )


def explain_evomapx(report: EvoMapXReport, top_k: int = 3) -> str:
    """Generate a concise textual interpretation for manuscripts/reports."""
    if not report.labels:
        return "EvoMapX analysis found no usable telemetry or history records. Store history or island telemetry before running the analysis."
    ranked = sorted(report.cds_normalized.items(), key=lambda kv: kv[1], reverse=True)
    raw_ranked = sorted(report.cds_raw.items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        "EvoMapX-style explainability analysis",
        f"Objective: {report.objective}; attribution level: {report.level}.",
        f"The attribution matrix contains {len(report.labels)} units across {len(report.steps)} time steps.",
    ]
    if ranked:
        leader, score = ranked[0]
        lines.append(f"The strongest convergence driver was {leader} with normalized CDS = {score:.4f}.")
        if len(ranked) > 1:
            tail = ", ".join(f"{k}={v:.4f}" for k, v in ranked[1:top_k])
            if tail:
                lines.append(f"Other relevant drivers were {tail}.")
    active = sorted(report.activity.items(), key=lambda kv: kv[1].get("activity_rate", 0.0), reverse=True)
    if active:
        name, metrics = active[0]
        lines.append(
            f"The most persistent active unit was {name}, active in {100.0 * metrics.get('activity_rate', 0.0):.1f}% of recorded steps."
        )
    peg_info = report.summary.get("peg", {}) if report.summary else {}
    if peg_info.get("n_nodes", 0) > 0:
        lines.append(
            f"The PEG contains {peg_info.get('n_nodes', 0)} nodes and {peg_info.get('n_edges', 0)} directed edges; "
            f"edge types: {peg_info.get('edge_types', {})}."
        )
    mig = report.summary.get("migration", {}) if report.summary else {}
    if mig.get("n_events", 0) > 0:
        lines.append(
            f"Migration analysis recorded {mig.get('n_events', 0)} events, {mig.get('n_successes', 0)} with immediate target improvement."
        )
    if raw_ranked and raw_ranked[0][1] <= 0:
        lines.append("No positive objective-consistent improvement was detected; check whether the objective direction and stored telemetry are correct.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_plotly(fig: go.Figure, filepath: str | Path | None, scale: int = 2) -> None:
    if filepath is None:
        return
    p = Path(str(filepath))
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext in {"", ".html"}:
        if ext == "":
            p = p.with_suffix(".html")
        fig.write_html(str(p))
    elif ext in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"}:
        fig.write_image(str(p), scale=scale)
    else:
        fig.write_html(str(p.with_suffix(".html")))


def _show(fig: go.Figure, show: bool = False, renderer: str = "browser") -> go.Figure:
    if show:
        if renderer:
            pio.renderers.default = renderer
        fig.show()
    return fig


def _base_layout(fig: go.Figure, title: str, width: int = 950, height: int = 560) -> go.Figure:
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=width,
        height=height,
    )
    return fig


def _as_report(report_or_result: Any, **kwargs) -> EvoMapXReport:
    return report_or_result if isinstance(report_or_result, EvoMapXReport) else evomapx_analysis(report_or_result, **kwargs)


def plot_attribution_heatmap(
    report_or_result: Any,
    normalized: bool = True,
    filepath: str | Path | None = None,
    show: bool = False,
    renderer: str = "browser",
    title: str | None = None,
    **analysis_kwargs,
) -> go.Figure:
    """Plot the OAM/IAM as a heatmap."""
    report = _as_report(report_or_result, **analysis_kwargs)
    matrix = report.normalized_attribution if normalized else report.raw_attribution
    z = [matrix.get(label, []) for label in report.labels]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=report.steps,
        y=report.labels,
        colorscale="Viridis",
        colorbar=dict(title="Attribution" if normalized else "Improvement"),
        hovertemplate="step=%{x}<br>unit=%{y}<br>value=%{z:.6g}<extra></extra>",
    ))
    _base_layout(fig, title or ("EvoMapX Attribution Matrix" if normalized else "EvoMapX Raw Improvement Matrix"), height=max(440, 120 + 32 * len(report.labels)))
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Attribution unit")
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_cds_bar(
    report_or_result: Any,
    normalized: bool = True,
    filepath: str | Path | None = None,
    show: bool = False,
    renderer: str = "browser",
    title: str | None = None,
    **analysis_kwargs,
) -> go.Figure:
    """Plot Convergence Driver Score as a sorted bar chart."""
    report = _as_report(report_or_result, **analysis_kwargs)
    scores = report.cds_normalized if normalized else report.cds_raw
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    labels = [i[0] for i in items]
    values = [float(i[1]) for i in items]
    colors = [_DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)] for i in range(len(labels))]
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.25)", width=1)),
        hovertemplate="unit=%{x}<br>CDS=%{y:.6g}<extra></extra>",
    ))
    _base_layout(fig, title or ("EvoMapX Normalized Convergence Driver Score" if normalized else "EvoMapX Raw Convergence Driver Score"))
    fig.update_xaxes(title_text="Attribution unit")
    fig.update_yaxes(title_text="CDS")
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_cds_time_series(
    report_or_result: Any,
    normalized: bool = True,
    filepath: str | Path | None = None,
    show: bool = False,
    renderer: str = "browser",
    title: str | None = None,
    **analysis_kwargs,
) -> go.Figure:
    """Plot attribution values through time for each unit."""
    report = _as_report(report_or_result, **analysis_kwargs)
    matrix = report.normalized_attribution if normalized else report.raw_attribution
    fig = go.Figure()
    for i, label in enumerate(report.labels):
        fig.add_trace(go.Scatter(
            x=report.steps,
            y=matrix.get(label, []),
            mode="lines+markers",
            name=label,
            line=dict(color=_DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)], width=2.2),
            marker=dict(size=5),
            hovertemplate="step=%{x}<br>value=%{y:.6g}<extra></extra>",
        ))
    _base_layout(fig, title or "EvoMapX Attribution Time Series")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Attribution" if normalized else "Improvement")
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_population_evolution_graph(
    report_or_result: Any,
    filepath: str | Path | None = None,
    show: bool = False,
    renderer: str = "browser",
    title: str | None = None,
    max_nodes: int = 800,
    **analysis_kwargs,
) -> go.Figure:
    """Plot the Population Evolution Graph using a lightweight layout."""
    report = _as_report(report_or_result, build_peg=True, max_peg_nodes=max_nodes, **analysis_kwargs)
    nodes = list(report.peg.get("nodes", []) or [])[:max_nodes]
    node_ids = [n.get("id") for n in nodes]
    id_set = set(node_ids)
    edges = [e for e in (report.peg.get("edges", []) or []) if e.get("source") in id_set and e.get("target") in id_set]
    if not nodes:
        fig = go.Figure()
        fig.add_annotation(text="No population snapshots or migration events available for PEG.", x=0.5, y=0.5, showarrow=False, font=dict(color=_TEXT_CLR, size=15))
        _base_layout(fig, title or "EvoMapX Population Evolution Graph")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        _save_plotly(fig, filepath)
        return _show(fig, show=show, renderer=renderer)

    steps = np.array([float(n.get("step") or 0.0) for n in nodes], dtype=float)
    labels = [str(n.get("label") or n.get("algorithm") or "node") for n in nodes]
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    # Deterministic layered layout: x=time, y=label lane + small deterministic jitter.
    x = steps
    y = np.array([label_map[l] for l in labels], dtype=float)
    for idx in range(len(y)):
        y[idx] += ((idx * 37) % 17 - 8) / 40.0
    pos = {str(n.get("id")): (float(xi), float(yi)) for n, xi, yi in zip(nodes, x, y)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for e in edges:
        s, t = str(e.get("source")), str(e.get("target"))
        if s in pos and t in pos:
            edge_x += [pos[s][0], pos[t][0], None]
            edge_y += [pos[s][1], pos[t][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(220,220,220,0.25)"),
        hoverinfo="skip",
        name="evolution edges",
    ))
    node_colors = [_DISCRETE_PALETTE[label_map[l] % len(_DISCRETE_PALETTE)] for l in labels]
    fitness = [n.get("fitness") for n in nodes]
    hover = [
        f"id={n.get('id')}<br>label={n.get('label')}<br>algorithm={n.get('algorithm')}<br>step={n.get('step')}<br>fitness={n.get('fitness')}<br>role={n.get('role')}"
        for n in nodes
    ]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=7, color=node_colors, line=dict(color="rgba(255,255,255,0.25)", width=0.6)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="population nodes",
    ))
    _base_layout(fig, title or "EvoMapX Population Evolution Graph", width=1050, height=max(480, 180 + 80 * max(1, len(label_map))))
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Algorithm / island lane", tickmode="array", tickvals=list(label_map.values()), ticktext=list(label_map.keys()))
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_evomapx_json(report_or_result: Any, filepath: str | Path, **analysis_kwargs) -> str:
    """Export an EvoMapX report to JSON."""
    report = _as_report(report_or_result, **analysis_kwargs)
    path = Path(str(filepath))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report.to_dict()), f, indent=2)
    return str(path)


def export_evomapx_csv(report_or_result: Any, filepath: str | Path, normalized: bool = True, **analysis_kwargs) -> str:
    """Export the attribution matrix to a tidy CSV file."""
    report = _as_report(report_or_result, **analysis_kwargs)
    path = Path(str(filepath))
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_dataframe(normalized=normalized).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Callback for ordinary optimizer runs
# ---------------------------------------------------------------------------


try:
    from .callbacks import Callback
except Exception:  # pragma: no cover - only relevant during unusual import cycles
    Callback = object  # type: ignore


class EvoMapXRecorder(Callback):
    """Callback that stores EvoMapX-compatible step records during a run.

    This is useful when users want a compact telemetry stream even if they do
    not store the full optimization history. The recorder attaches its records
    to ``result.metadata['evomapx_records']`` after the run.
    """

    def __init__(self, include_population_summary: bool = True) -> None:
        super().__init__()
        self.include_population_summary = bool(include_population_summary)
        self.records: list[dict[str, Any]] = []
        self._previous_best: float | None = None

    def before_run(self, **kwargs) -> None:
        self.records = []
        self._previous_best = None

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        state = kwargs.get("state")
        obs = dict(kwargs.get("observation") or {})
        objective = getattr(getattr(self.engine, "problem", None), "objective", "min")
        imp = _improvement(self._previous_best, best_fitness, objective) if self._previous_best is not None else 0.0
        self._previous_best = None if best_fitness is None else float(best_fitness)
        rec = {
            "step": getattr(state, "step", obs.get("step", len(self.records) + 1)),
            "evaluations": getattr(state, "evaluations", obs.get("evaluations")),
            "algorithm": getattr(self.engine, "algorithm_id", None),
            "best_fitness": None if best_fitness is None else float(best_fitness),
            "raw_improvement": float(imp or 0.0),
            "positive_improvement": _positive(imp, positive_only=True),
            "operator": obs.get("operator") or obs.get("action") or getattr(self.engine, "algorithm_id", None),
            "observation": obs,
        }
        if self.include_population_summary and fitness is not None:
            arr = np.asarray(fitness, dtype=float)
            if arr.size:
                rec.update({
                    "population_mean_fitness": float(np.mean(arr)),
                    "population_std_fitness": float(np.std(arr)),
                    "population_best_fitness": float(np.min(arr) if objective == "min" else np.max(arr)),
                })
        self.records.append(rec)

    def after_run(self, **kwargs) -> None:
        result = kwargs.get("result")
        if result is not None:
            result.metadata.setdefault("evomapx_records", list(self.records))


__all__ = [
    "EvoMapXNode",
    "EvoMapXEdge",
    "EvoMapXReport",
    "EvoMapXRecorder",
    "attribution_records",
    "operator_attribution_matrix",
    "convergence_driver_score",
    "attribution_activity",
    "population_evolution_graph",
    "peg_summary",
    "migration_attribution",
    "evomapx_analysis",
    "explain_evomapx",
    "plot_attribution_heatmap",
    "plot_cds_bar",
    "plot_cds_time_series",
    "plot_population_evolution_graph",
    "export_evomapx_json",
    "export_evomapx_csv",
]
