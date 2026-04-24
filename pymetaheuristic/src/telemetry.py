from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def history_to_rows(result) -> list[dict[str, Any]]:
    return list(getattr(result, "history", []) or [])


def improvement_rows(result) -> list[dict[str, Any]]:
    metadata = getattr(result, "metadata", {}) or {}
    return list(metadata.get("improvement_history", []) or [])


def convergence_data(result, x_axis: str = "steps") -> tuple[list[float], list[float]]:
    """Return convergence arrays indexed by steps or evaluations."""
    axis = str(x_axis).strip().lower()
    if axis in {"steps", "step", "iters", "iterations"}:
        history = history_to_rows(result)
        if not history:
            best = getattr(result, "best_fitness", None)
            if best is None:
                return [], []
            return [0.0], [float(best)]
        x = [float(row.get("step", i + 1)) for i, row in enumerate(history)]
        y = [float(row.get("global_best_fitness", row.get("best_fitness"))) for row in history]
        return x, y
    if axis in {"evaluations", "evals", "evaluation"}:
        improvements = improvement_rows(result)
        if improvements:
            x = [float(row.get("evaluations", 0)) for row in improvements]
            y = [float(row.get("best_fitness")) for row in improvements]
            return x, y
        history = history_to_rows(result)
        if history:
            x = [float(row.get("evaluations", i + 1)) for i, row in enumerate(history)]
            y = [float(row.get("global_best_fitness", row.get("best_fitness"))) for row in history]
            return x, y
        best = getattr(result, "best_fitness", None)
        evals = getattr(result, "evaluations", 0)
        if best is None:
            return [], []
        return [float(evals)], [float(best)]
    raise ValueError("x_axis must be one of: 'steps', 'iterations', 'evaluations', 'evals'.")


def island_telemetry_to_rows(result) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    telemetry = getattr(result, "island_telemetry", {}) or {}
    for label, records in telemetry.items():
        for rec in records:
            row = dict(getattr(rec, '__dict__', {}))
            row.setdefault('label', label)
            out.append(row)
    return out


def summarize_result(result) -> dict[str, Any]:
    history = history_to_rows(result)
    metadata = getattr(result, "metadata", {}) or {}
    first_best = history[0]["best_fitness"] if history else getattr(result, "best_fitness", None)
    last_best = history[-1]["best_fitness"] if history else getattr(result, "best_fitness", None)
    improvements = improvement_rows(result)
    summary = {
        "algorithm_id": getattr(result, "algorithm_id", None),
        "best_fitness": getattr(result, "best_fitness", None),
        "steps": getattr(result, "steps", None),
        "evaluations": getattr(result, "evaluations", None),
        "termination_reason": getattr(result, "termination_reason", None),
        "history_points": len(history),
        "initial_best": first_best,
        "final_best": last_best,
        "improvement": (None if first_best is None or last_best is None else first_best - last_best),
        "elapsed_time": metadata.get("elapsed_time"),
        "n_improvements": len(improvements),
    }
    if improvements:
        summary["first_improvement_eval"] = improvements[0].get("evaluations")
        summary["last_improvement_eval"] = improvements[-1].get("evaluations")
    if history:
        for key in ("mean_fitness", "std_fitness", "diversity", "temperature", "accepted_ratio"):
            if key in history[-1]:
                summary[f"last_{key}"] = history[-1][key]
    if getattr(result, 'island_telemetry', None):
        summary['n_islands'] = len(result.island_telemetry)
        summary['n_migrations'] = len(getattr(result, 'events', []) or [])
    return summary


def summarize_results(results: Iterable) -> list[dict[str, Any]]:
    return [summarize_result(r) for r in results]


def summarize_cooperative_result(result) -> dict[str, Any]:
    telemetry = island_telemetry_to_rows(result)
    metadata = getattr(result, 'metadata', {}) or {}
    return {
        'best_fitness': getattr(result, 'best_fitness', None),
        'n_islands': len(getattr(result, 'island_telemetry', {}) or {}),
        'n_events': len(getattr(result, 'events', []) or []),
        'topology': metadata.get('topology'),
        'migration_policy': metadata.get('migration_policy'),
        'donor_strategy': metadata.get('donor_strategy'),
        'receiver_strategy': metadata.get('receiver_strategy'),
        'adaptive_checkpointing': metadata.get('adaptive_checkpointing'),
        'final_migration_interval': metadata.get('final_migration_interval'),
        'telemetry_points': len(telemetry),
    }


def export_history_csv(result, filepath: str | Path) -> str:
    rows = history_to_rows(result)
    filepath = str(filepath)
    if not rows:
        with open(filepath, "w", newline="") as f:
            f.write("")
        return filepath
    keys = sorted({k for row in rows for k in row.keys()})
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def export_island_telemetry_csv(result, filepath: str | Path) -> str:
    rows = island_telemetry_to_rows(result)
    filepath = str(filepath)
    if not rows:
        with open(filepath, "w", newline="") as f:
            f.write("")
        return filepath
    keys = sorted({k for row in rows for k in row.keys()})
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def export_population_snapshots_json(result, filepath: str | Path) -> str:
    filepath = str(filepath)
    with open(filepath, "w") as f:
        json.dump(getattr(result, "population_snapshots", []) or [], f, indent=2)
    return filepath


def export_replay_manifest_json(result, filepath: str | Path) -> str:
    filepath = str(filepath)
    with open(filepath, 'w') as f:
        json.dump(getattr(result, 'replay_manifest', {}) or {}, f, indent=2)
    return filepath
