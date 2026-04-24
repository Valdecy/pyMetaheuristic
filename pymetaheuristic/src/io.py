"""
pyMetaheuristic src — IO (Save / Load / Checkpoint)
====================================================
Feature 9: Serialise and restore OptimizationResult objects and
running engine state for checkpoint-and-resume workflows.

Public API
----------
``save_result(result, path)``      — pickle an OptimizationResult to disk.
``load_result(path)``              — restore a pickled OptimizationResult.
``save_checkpoint(engine, state, path)`` — pickle a running engine + state.
``load_checkpoint(path)``          — restore engine + state; ready to resume.
``result_to_json(result, path)``   — export a JSON-safe summary to disk.
``result_from_json(path)``         — re-read a JSON-exported summary.

Usage
-----
::

    import pymetaheuristic
    from pymetaheuristic.src.io import (
        save_result, load_result,
        save_checkpoint, load_checkpoint,
        result_to_json,
    )

    # --- save / load a completed result ---
    result = pymetaheuristic.optimize("pso", fn, lb, ub, max_steps=500)
    save_result(result, "run_pso.pkl")
    result2 = load_result("run_pso.pkl")

    # --- checkpoint an in-progress run ---
    engine = pymetaheuristic.create_optimizer("pso", fn, lb, ub, max_steps=1000)
    state  = engine.initialize()
    for _ in range(100):
        state = engine.step(state)
    save_checkpoint(engine, state, "checkpoint_100.pkl")

    # ... later, resume:
    engine2, state2 = load_checkpoint("checkpoint_100.pkl")
    for _ in range(900):
        state2 = engine2.step(state2)
    result = engine2.finalize(state2)

    # --- export to JSON (human-readable, no binary) ---
    result_to_json(result, "run_pso.json")
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Tuple


def _ensure_pkl_ext(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".pkl":
        p = p.with_suffix(".pkl")
    return p


def _ensure_json_ext(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    return p


# ===========================================================================
# Result save / load
# ===========================================================================

def save_result(result, path: str | Path) -> Path:
    """
    Pickle an ``OptimizationResult`` to disk.

    Parameters
    ----------
    result : OptimizationResult returned by optimize() / run().
    path   : Output path.  ``.pkl`` extension is appended if absent.

    Returns
    -------
    Resolved Path where the file was written.

    Notes
    -----
    If the result contains large ``population_snapshots``, consider
    clearing them before saving to reduce file size::

        result.population_snapshots = []
        save_result(result, "run.pkl")
    """
    p = _ensure_pkl_ext(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_result(path: str | Path):
    """
    Restore an ``OptimizationResult`` from disk.

    Parameters
    ----------
    path : Path written by ``save_result()``.

    Returns
    -------
    OptimizationResult
    """
    p = _ensure_pkl_ext(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# Checkpoint save / load
# ===========================================================================

def save_checkpoint(engine, state, path: str | Path) -> Path:
    """
    Pickle a running engine together with its current state.

    This allows the run to be resumed later with ``load_checkpoint()``.

    Parameters
    ----------
    engine : BaseEngine instance (already initialised).
    state  : EngineState at the checkpoint moment.
    path   : Output path.

    Returns
    -------
    Resolved Path.

    Warnings
    --------
    Checkpoint files contain the full engine, including the problem spec and
    all algorithm state.  They are not guaranteed to be portable across
    package versions or Python minor versions.
    """
    p = _ensure_pkl_ext(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine":   engine,
        "state":    state,
        "metadata": {
            "algorithm_id":  engine.algorithm_id,
            "step":          state.step,
            "evaluations":   state.evaluations,
            "best_fitness":  state.best_fitness,
        },
    }
    with open(p, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_checkpoint(path: str | Path) -> Tuple[Any, Any]:
    """
    Restore an engine and its state from a checkpoint file.

    Parameters
    ----------
    path : Path written by ``save_checkpoint()``.

    Returns
    -------
    (engine, state)  — ready to continue stepping.

    Example
    -------
    ::

        engine, state = load_checkpoint("checkpoint_100.pkl")
        while not engine.should_stop(state):
            state = engine.step(state)
        result = engine.finalize(state)
    """
    p = _ensure_pkl_ext(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    engine = payload["engine"]
    state  = payload["state"]
    meta   = payload.get("metadata", {})
    print(
        f"[load_checkpoint] Loaded {meta.get('algorithm_id', '?')} "
        f"at step={meta.get('step', '?')} "
        f"(evals={meta.get('evaluations', '?')}, "
        f"best={meta.get('best_fitness', '?')})"
    )
    return engine, state


# ===========================================================================
# JSON export / import (human-readable, no binary)
# ===========================================================================

def _json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    return str(obj)


def result_to_json(result, path: str | Path, indent: int = 2) -> Path:
    """
    Export a JSON-safe summary of an ``OptimizationResult`` to disk.

    The exported file is human-readable and can be re-read with
    ``result_from_json()``.  Note that this is a *summary* — the full
    ``history`` list and ``population_snapshots`` are included only if
    they are present and non-empty.

    Parameters
    ----------
    result : OptimizationResult.
    path   : Output path.  ``.json`` extension is appended if absent.
    indent : JSON indentation width.

    Returns
    -------
    Resolved Path.
    """
    p = _ensure_json_ext(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "algorithm_id":       getattr(result, "algorithm_id",       None),
        "best_fitness":       getattr(result, "best_fitness",        None),
        "best_position":      getattr(result, "best_position",       None),
        "steps":              getattr(result, "steps",               None),
        "evaluations":        getattr(result, "evaluations",         None),
        "termination_reason": getattr(result, "termination_reason",  None),
        "metadata":           getattr(result, "metadata",            {}),
        "history":            getattr(result, "history",             []),
    }
    population_snapshots = getattr(result, "population_snapshots", [])
    if population_snapshots:
        data["population_snapshots"] = population_snapshots

    with open(p, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=indent, ensure_ascii=False)
    return p


def result_from_json(path: str | Path) -> dict:
    """
    Read a JSON file written by ``result_to_json()``.

    Returns a plain Python dict (not an OptimizationResult instance) so
    that loading does not require the exact package version that wrote it.

    Parameters
    ----------
    path : Path written by ``result_to_json()``.

    Returns
    -------
    dict
    """
    p = _ensure_json_ext(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON result file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
