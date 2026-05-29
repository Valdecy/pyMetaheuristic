"""Validation utilities for package-wide EvoMapX support.

The main entry point, :func:`validate_evomapx_engines`, is a lightweight
quality-control helper.  It runs one or more registered engines on a simple
benchmark problem, builds the EvoMapX analysis, and reports which explainability
artifacts were produced.

The function is intentionally conservative: it validates runtime availability of
telemetry/artifacts, not the scientific validity of an algorithm or the
mathematical optimality of a result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable
import json
import math
import time
import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd

from .api import get_algorithm_info, list_algorithms, optimize
from .evomapx import (
    evomapx_analysis,
    export_evomapx_csv,
    export_evomapx_json,
    plot_attribution_heatmap,
    plot_cds_bar,
    plot_population_evolution_graph,
)
from .evomapx_profiles import get_evomapx_profile


_SIZE_PARAMETER_NAMES: set[str] = {
    "population_size",
    "pop_size",
    "size",
    "colony_size",
    "swarm_size",
    "pack_size",
    "school_size",
    "solutions",
    "sample_count",
    "food_sources",
    "employed_bees",
    "outlookers_bees",
    "hawks",
    "grasshoppers",
    "universes",
    "jellyfishes",
    "search_agents",
    "n_agents",
    "n_particles",
    "n_ants",
    "n_bats",
    "n_bees",
    "n_fireflies",
    "n_wolves",
    "n_whales",
    "n_hawks",
    "n_nests",
    "n_sources",
    "n_food_sources",
    "n_harmonies",
    "harmony_memory_size",
    "construction_pool_size",
    "candidate_pool_size",
    "n_initial_points",
    "max_samples",
    "No",
    "Ne",
}

_SMALL_COUNT_PARAMETER_NAMES: set[str] = {
    "local_search_steps",
    "max_ls",
    "ls_steps",
    "starts_per_step",
    "n_estimators",
    "batch_size",
    "polish_points",
    "temperature_iterations",
    "attempts",
    "max_swim_steps",
    "n_trials",
    "num_tests",
    "num_searches",
    "num_searches_best",
    "neighbour_size",
    "neighborhood_size",
    "tabu_size",
}


def _sphere(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.dot(arr, arr))


def _as_engine_list(engines: Any = None, target_engines: Any = None) -> list[str]:
    raw = target_engines if target_engines is not None else engines
    if raw is None:
        return list_algorithms()
    if isinstance(raw, str):
        if raw.strip().lower() in {"", "all", "*"}:
            return list_algorithms()
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def _safe_number(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
    except Exception:
        return value


def _compact_engine_params(
    algorithm: str,
    population_size: int,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return conservative speed-oriented params for one engine.

    Only keys declared by the engine defaults are modified.  This avoids
    injecting arbitrary parameter names into engines that might interpret them
    unexpectedly.
    """
    try:
        defaults = get_algorithm_info(algorithm).get("defaults", {}) or {}
    except Exception:
        defaults = {}

    n = max(2, int(population_size))
    params: dict[str, Any] = {}
    for key, value in defaults.items():
        lower = str(key).lower()
        if key in _SIZE_PARAMETER_NAMES or lower in {k.lower() for k in _SIZE_PARAMETER_NAMES}:
            params[key] = n
        elif key in _SMALL_COUNT_PARAMETER_NAMES or lower in {k.lower() for k in _SMALL_COUNT_PARAMETER_NAMES}:
            # Keep local-search/surrogate auxiliary loops small but nonzero.
            params[key] = max(1, min(n, 8))

    # A few dependency-aware repairs for known parameter relations.
    if "number_groups" in defaults and "population_size" in defaults:
        groups = int(defaults.get("number_groups", 5) or 5)
        params["population_size"] = max(n, groups + 1)
    if "min_population_size" in defaults and "population_size" in defaults:
        min_pop = int(defaults.get("min_population_size", 4) or 4)
        params["population_size"] = max(params.get("population_size", n), min_pop + 1)
    if "min_population" in defaults and "population_size" in defaults:
        min_pop = int(defaults.get("min_population", 4) or 4)
        params["population_size"] = max(params.get("population_size", n), min_pop + 1)
    if "archive_size" in defaults and "population_size" in defaults:
        # Archive cannot be smaller than a few methods expect, but keep it compact.
        params["archive_size"] = max(2, min(max(n, 4), int(defaults.get("archive_size") or n)))

    if extra_params:
        params.update(dict(extra_params))
    return params


def _classify_fidelity(result: Any, report: Any) -> tuple[str, str]:
    metadata = getattr(result, "metadata", {}) or {}
    history = getattr(result, "history", []) or []
    records = metadata.get("evomapx_records", []) or []

    runtime_values = {
        str(row.get("evomapx_fidelity_runtime"))
        for row in history
        if isinstance(row, dict) and row.get("evomapx_fidelity_runtime") is not None
    }
    record_sources = {
        str(row.get("source"))
        for row in records
        if isinstance(row, dict) and row.get("source") is not None
    }

    joined = " ".join(sorted(runtime_values | record_sources)).lower()
    if "native" in joined:
        return "native", "engine-native EvoMapX/operator telemetry detected"
    if "lineage_delta" in joined or "passive_probe_lineage" in joined:
        return "inferred", "passive parent-child lineage attribution detected"
    if "event_fallback" in joined or "passive_probe_event" in joined:
        return "inferred", "passive event-level operator attribution detected"
    if getattr(report, "labels", None) and getattr(report, "steps", None):
        return "fallback", "generic history/progress attribution detected"
    return "unavailable", "no usable EvoMapX attribution records detected"


def _safe_error(exc: BaseException) -> tuple[str, str]:
    return type(exc).__name__, str(exc)[:1000]


def _export_artifacts(
    result: Any,
    report: Any,
    algorithm: str,
    export_dir: str | Path,
    export_formats: Iterable[str],
    test_pdf: bool,
    test_plots: bool,
) -> tuple[dict[str, str], list[str]]:
    exported: dict[str, str] = {}
    errors: list[str] = []
    base = Path(export_dir)
    base.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(algorithm))
    formats = {str(fmt).lower().strip() for fmt in export_formats}

    if "json" in formats:
        path = base / f"{safe_name}_evomapx.json"
        try:
            exported["json"] = export_evomapx_json(report, path)
        except Exception as exc:  # pragma: no cover - depends on filesystem
            et, msg = _safe_error(exc)
            errors.append(f"json:{et}: {msg}")
    if "csv" in formats:
        path = base / f"{safe_name}_evomapx.csv"
        try:
            exported["csv"] = export_evomapx_csv(report, path)
        except Exception as exc:  # pragma: no cover - depends on filesystem
            et, msg = _safe_error(exc)
            errors.append(f"csv:{et}: {msg}")
    if test_plots or "html" in formats:
        try:
            exported["heatmap_html"] = str(base / f"{safe_name}_oam.html")
            plot_attribution_heatmap(report, filepath=exported["heatmap_html"], show=False)
        except Exception as exc:
            et, msg = _safe_error(exc)
            errors.append(f"heatmap_html:{et}: {msg}")
        try:
            exported["cds_html"] = str(base / f"{safe_name}_cds.html")
            plot_cds_bar(report, filepath=exported["cds_html"], show=False)
        except Exception as exc:
            et, msg = _safe_error(exc)
            errors.append(f"cds_html:{et}: {msg}")
    if test_pdf or "pdf" in formats:
        try:
            exported["cds_pdf"] = str(base / f"{safe_name}_cds.pdf")
            plot_cds_bar(report, filepath=exported["cds_pdf"], show=False)
        except Exception as exc:
            et, msg = _safe_error(exc)
            errors.append(f"cds_pdf:{et}: {msg}")
        try:
            exported["oam_pdf"] = str(base / f"{safe_name}_oam.pdf")
            plot_attribution_heatmap(report, filepath=exported["oam_pdf"], show=False)
        except Exception as exc:
            et, msg = _safe_error(exc)
            errors.append(f"oam_pdf:{et}: {msg}")
        try:
            exported["peg_pdf"] = str(base / f"{safe_name}_peg.pdf")
            plot_population_evolution_graph(report, filepath=exported["peg_pdf"], show=False)
        except Exception as exc:
            et, msg = _safe_error(exc)
            errors.append(f"peg_pdf:{et}: {msg}")
    return exported, errors


def validate_evomapx_engines(
    engines: str | Iterable[str] | None = None,
    *,
    target_engines: str | Iterable[str] | None = None,
    target_function: Callable[[Iterable[float]], float] | None = None,
    dimensions: int = 2,
    min_values: Iterable[float] | None = None,
    max_values: Iterable[float] | None = None,
    bounds: tuple[float, float] = (-5.0, 5.0),
    objective: str = "min",
    population_size: int = 12,
    max_steps: int = 3,
    max_evaluations: int | None = 1000,
    timeout_seconds: float | None = 10.0,
    seed: int | None = 42,
    store_history: bool = True,
    store_population_snapshots: bool = True,
    snapshot_interval: int = 1,
    evomapx_level: int = 4,
    level: str = "auto",
    build_peg: bool = True,
    max_peg_nodes: int = 1000,
    strict: bool = False,
    engine_params: dict[str, Any] | None = None,
    per_engine_params: dict[str, dict[str, Any]] | None = None,
    export: bool = False,
    export_dir: str | Path = "evomapx_validation",
    export_formats: Iterable[str] = ("json", "csv"),
    test_pdf: bool = False,
    test_plots: bool = False,
    return_dataframe: bool = True,
    verbose: bool = True,
    suppress_warnings: bool = True,
    raise_on_error: bool = False,
) -> pd.DataFrame | list[dict[str, Any]]:
    """Validate EvoMapX runtime support for all or selected engines.

    Parameters
    ----------
    engines, target_engines:
        Engines to test.  ``None``, ``"all"`` or ``"*"`` tests every registered
        engine.  A string such as ``"ga,de,pso"`` or a list such as
        ``["ga", "de", "pso"]`` tests only those engines.  ``target_engines`` is
        an alias intended for newly added engines.
    target_function:
        Objective function used for validation.  Defaults to Sphere.
    dimensions, min_values, max_values, bounds:
        Problem definition.  ``min_values``/``max_values`` override ``bounds``.
    population_size:
        Compact population size applied to recognized engine size parameters.
    max_steps, max_evaluations, timeout_seconds:
        Runtime limits for each smoke test.
    strict:
        If True, status is ``"warning"`` unless OAM, CDS and PEG are all present.
        If False, OAM/CDS availability is enough for ``"ok"`` and PEG absence is
        reported separately.
    export, export_formats, test_pdf, test_plots:
        Optional artifact-export checks.  PDF export depends on Plotly/Kaleido.
    return_dataframe:
        Return a pandas DataFrame when True, otherwise a list of dictionaries.
    suppress_warnings:
        Suppress engine warnings during validation smoke tests.

    Returns
    -------
    pandas.DataFrame or list[dict]
        One validation row per requested engine.
    """
    selected = _as_engine_list(engines=engines, target_engines=target_engines)
    registered = set(list_algorithms())
    dim = max(1, int(dimensions))
    if min_values is None:
        lo = [float(bounds[0])] * dim
    else:
        lo = [float(v) for v in min_values]
    if max_values is None:
        hi = [float(bounds[1])] * len(lo)
    else:
        hi = [float(v) for v in max_values]
    if len(lo) != len(hi):
        raise ValueError("min_values and max_values must have the same length.")
    fn = target_function or _sphere

    rows: list[dict[str, Any]] = []
    total = len(selected)
    for idx, algorithm in enumerate(selected, start=1):
        start = time.perf_counter()
        base_row: dict[str, Any] = {
            "algorithm": algorithm,
            "requested_index": idx,
            "requested_total": total,
            "status": None,
            "fidelity": None,
            "profile_fidelity": None,
            "family": None,
            "steps": None,
            "evaluations": None,
            "termination_reason": None,
            "history_rows": 0,
            "population_snapshots": 0,
            "probe_enabled": None,
            "probe_records": 0,
            "n_labels": 0,
            "n_steps": 0,
            "has_oam": False,
            "has_cds": False,
            "has_peg": False,
            "peg_nodes": 0,
            "peg_edges": 0,
            "top_driver": None,
            "top_driver_score": 0.0,
            "exported": {},
            "export_errors": [],
            "elapsed_seconds": None,
            "message": "",
            "error_type": None,
            "error": None,
        }

        if algorithm not in registered:
            base_row.update({
                "status": "failed",
                "fidelity": "unavailable",
                "message": "engine is not registered in pymetaheuristic.src.engines.REGISTRY",
                "error_type": "KeyError",
                "error": f"Unknown algorithm: {algorithm}",
                "elapsed_seconds": float(time.perf_counter() - start),
            })
            rows.append(base_row)
            if verbose:
                print(f"[{idx:>3}/{total}] FAILED {algorithm}: not registered")
            if raise_on_error:
                raise KeyError(base_row["error"])
            continue

        try:
            info = get_algorithm_info(algorithm)
            base_row["family"] = info.get("family")
            try:
                profile = get_evomapx_profile(algorithm)
                base_row["profile_fidelity"] = getattr(profile, "fidelity", None)
            except Exception:
                base_row["profile_fidelity"] = None

            params = _compact_engine_params(algorithm, population_size, engine_params)
            if per_engine_params and algorithm in per_engine_params:
                params.update(dict(per_engine_params[algorithm] or {}))
            params.update({
                "evomapx": True,
                "enable_evomapx": True,
                "evomapx_level": int(evomapx_level),
            })

            warning_context = warnings.catch_warnings() if suppress_warnings else nullcontext()
            with warning_context:
                if suppress_warnings:
                    warnings.simplefilter("ignore")
                result = optimize(
                    algorithm,
                    target_function=fn,
                    min_values=lo,
                    max_values=hi,
                    objective=objective,
                    max_steps=max_steps,
                    max_evaluations=max_evaluations,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                    verbose=False,
                    store_history=store_history,
                    store_population_snapshots=store_population_snapshots,
                    snapshot_interval=snapshot_interval,
                    **params,
                )
            report = evomapx_analysis(
                result,
                objective=objective,
                level=level,
                build_peg=build_peg,
                max_peg_nodes=max_peg_nodes,
            )

            peg_info = (report.summary.get("peg") or {}) if getattr(report, "summary", None) else {}
            has_oam = bool(report.labels and report.steps and report.raw_attribution)
            has_cds = bool(report.cds_raw)
            has_peg = bool(int(peg_info.get("n_nodes", 0) or 0) > 0)
            fidelity, fidelity_message = _classify_fidelity(result, report)
            metadata = getattr(result, "metadata", {}) or {}
            probe = metadata.get("evomapx_probe", {}) or {}
            top_driver = report.summary.get("top_driver") if report.summary else None
            top_score = report.summary.get("top_driver_score", 0.0) if report.summary else 0.0

            if has_oam and has_cds and ((not strict) or has_peg):
                status = "ok"
            elif has_oam or has_cds or has_peg:
                status = "warning"
            else:
                status = "unavailable"

            exported: dict[str, str] = {}
            export_errors: list[str] = []
            if export or test_pdf or test_plots:
                exported, export_errors = _export_artifacts(
                    result=result,
                    report=report,
                    algorithm=algorithm,
                    export_dir=export_dir,
                    export_formats=export_formats,
                    test_pdf=test_pdf,
                    test_plots=test_plots,
                )
                if export_errors and status == "ok":
                    status = "warning"

            message_parts = [fidelity_message]
            if not has_peg:
                message_parts.append("PEG nodes unavailable or not generated for this run")
            if export_errors:
                message_parts.append("one or more export checks failed")

            base_row.update({
                "status": status,
                "fidelity": fidelity,
                "steps": int(getattr(result, "steps", 0) or 0),
                "evaluations": int(getattr(result, "evaluations", 0) or 0),
                "termination_reason": getattr(result, "termination_reason", None),
                "history_rows": len(getattr(result, "history", []) or []),
                "population_snapshots": len(getattr(result, "population_snapshots", []) or []),
                "probe_enabled": probe.get("enabled"),
                "probe_records": int(probe.get("records", 0) or 0),
                "n_labels": len(report.labels or []),
                "n_steps": len(report.steps or []),
                "has_oam": has_oam,
                "has_cds": has_cds,
                "has_peg": has_peg,
                "peg_nodes": int(peg_info.get("n_nodes", 0) or 0),
                "peg_edges": int(peg_info.get("n_edges", 0) or 0),
                "top_driver": top_driver,
                "top_driver_score": float(_safe_number(top_score) or 0.0),
                "exported": exported,
                "export_errors": export_errors,
                "elapsed_seconds": float(time.perf_counter() - start),
                "message": "; ".join(message_parts),
            })
            rows.append(base_row)
            if verbose:
                print(
                    f"[{idx:>3}/{total}] {status.upper():<11} {algorithm:<22} "
                    f"fidelity={fidelity:<11} labels={base_row['n_labels']:<3} "
                    f"steps={base_row['n_steps']:<3} peg={base_row['peg_nodes']}/{base_row['peg_edges']}"
                )
        except Exception as exc:
            et, msg = _safe_error(exc)
            base_row.update({
                "status": "failed",
                "fidelity": "unavailable",
                "error_type": et,
                "error": msg,
                "message": f"validation run failed: {et}: {msg}",
                "elapsed_seconds": float(time.perf_counter() - start),
            })
            rows.append(base_row)
            if verbose:
                print(f"[{idx:>3}/{total}] FAILED      {algorithm:<22} {et}: {msg[:160]}")
            if raise_on_error:
                raise

    if export:
        outdir = Path(export_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        summary_path = outdir / "evomapx_validation_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
        try:
            pd.DataFrame(rows).to_csv(outdir / "evomapx_validation_summary.csv", index=False)
        except Exception:
            pass

    return pd.DataFrame(rows) if return_dataframe else rows


__all__ = ["validate_evomapx_engines"]
