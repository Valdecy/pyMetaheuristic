"""
pyMetaheuristic Web UI — FastAPI Backend  v4
=============================================
Supports: Single (standard / constrained / binary) ·
          Benchmark Runner · Collaborative Islands · Orchestrated Islands

Place as  pymetaheuristic/web/server.py
Run with: uvicorn pymetaheuristic.web.server:app --reload --port 8000
"""
from __future__ import annotations

import inspect
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ── Package imports ──────────────────────────────────────────────────────────
try:
    import pymetaheuristic as pmh
    from pymetaheuristic.src.api import list_algorithms, get_algorithm_info, optimize, create_optimizer
    from pymetaheuristic.src.cooperation import cooperative_optimize
    from pymetaheuristic.src.orchestration import orchestrated_optimize
    from pymetaheuristic.src.schemas import CollaborativeConfig, OrchestrationSpec, RulesConfig
    from pymetaheuristic.src.utils.chaotic import BinaryAdapter
    from pymetaheuristic.src.tuner import BenchmarkRunner
    from pymetaheuristic.src.termination import Termination
    import pymetaheuristic.src.test_functions as _tf
    _AVAILABLE_TRANSFER = getattr(pmh, "AVAILABLE_TRANSFER_FUNCTIONS",
                                  ["v1","v2","v3","v4","s1","s2","s3","s4"])
except ImportError:
    try:
        from ..src.api import list_algorithms, get_algorithm_info, optimize, create_optimizer
        from ..src.cooperation import cooperative_optimize
        from ..src.orchestration import orchestrated_optimize
        from ..src.schemas import CollaborativeConfig, OrchestrationSpec, RulesConfig
        from ..src.utils.chaotic import BinaryAdapter
        from ..src.tuner import BenchmarkRunner
        from ..src.termination import Termination
        from ..src import test_functions as _tf
        _AVAILABLE_TRANSFER = ["v1","v2","v3","v4","s1","s2","s3","s4"]
    except ImportError as e:
        raise RuntimeError(f"Cannot import pymetaheuristic: {e}") from e

app    = FastAPI(title="pyMetaheuristic Lab", docs_url=None, redoc_url=None)
_jobs: dict[str, dict] = {}

# ── Benchmark-function catalogue ─────────────────────────────────────────────
_KNOWN: dict[str, dict] = {
    "ackley":{"label":"Ackley","min":-32.768,"max":32.768,"optimum":0.0},
    "alpine_1":{"label":"Alpine 1","min":-10,"max":10,"optimum":0.0},
    "alpine_2":{"label":"Alpine 2","min":0,"max":10},
    "axis_parallel_hyper_ellipsoid":{"label":"Axis-Parallel Hyper-Ellipsoid","min":-5.12,"max":5.12,"optimum":0.0},
    "beale":{"label":"Beale","min":-4.5,"max":4.5,"optimum":0.0,"fixed_dims":2},
    "bent_cigar":{"label":"Bent Cigar","min":-100,"max":100,"optimum":0.0},
    "bohachevsky_1":{"label":"Bohachevsky 1","min":-100,"max":100,"optimum":0.0,"fixed_dims":2},
    "bohachevsky_2":{"label":"Bohachevsky 2","min":-100,"max":100,"optimum":0.0,"fixed_dims":2},
    "bohachevsky_3":{"label":"Bohachevsky 3","min":-100,"max":100,"optimum":0.0,"fixed_dims":2},
    "booth":{"label":"Booth","min":-10,"max":10,"optimum":0.0,"fixed_dims":2},
    "branin_rcos":{"label":"Branin RCOS","min":-5,"max":15,"optimum":0.398,"fixed_dims":2},
    "bukin_6":{"label":"Bukin 6","min":-15,"max":5,"fixed_dims":2},
    "chung_reynolds":{"label":"Chung-Reynolds","min":-100,"max":100,"optimum":0.0},
    "cosine_mixture":{"label":"Cosine Mixture","min":-1,"max":1},
    "cross_in_tray":{"label":"Cross-in-Tray","min":-10,"max":10,"optimum":-2.063,"fixed_dims":2},
    "csendes":{"label":"Csendes","min":-1,"max":1,"optimum":0.0},
    "de_jong_1":{"label":"De Jong 1 (Sphere)","min":-5.12,"max":5.12,"optimum":0.0},
    "discus":{"label":"Discus","min":-100,"max":100,"optimum":0.0},
    "dixon_price":{"label":"Dixon-Price","min":-10,"max":10,"optimum":0.0},
    "drop_wave":{"label":"Drop-Wave","min":-5.12,"max":5.12,"optimum":-1.0,"fixed_dims":2},
    "easom":{"label":"Easom","min":-100,"max":100,"optimum":-1.0,"fixed_dims":2},
    "eggholder":{"label":"Eggholder","min":-512,"max":512,"optimum":-959.64,"fixed_dims":2},
    "elliptic":{"label":"High-Conditioned Elliptic","min":-100,"max":100,"optimum":0.0},
    "goldstein_price":{"label":"Goldstein-Price","min":-2,"max":2,"optimum":3.0,"fixed_dims":2},
    "griewangk_8":{"label":"Griewangk","min":-600,"max":600,"optimum":0.0},
    "happy_cat":{"label":"Happy Cat","min":-2,"max":2,"optimum":0.0},
    "hgbat":{"label":"HGBat","min":-2,"max":2,"optimum":0.0},
    "himmelblau":{"label":"Himmelblau","min":-5,"max":5,"optimum":0.0,"fixed_dims":2},
    "holder_table":{"label":"Hölder Table","min":-10,"max":10,"optimum":-19.209,"fixed_dims":2},
    "katsuura":{"label":"Katsuura","min":0,"max":100},
    "levi_13":{"label":"Lévi N.13","min":-10,"max":10,"optimum":0.0,"fixed_dims":2},
    "levy":{"label":"Lévy","min":-10,"max":10,"optimum":0.0},
    "matyas":{"label":"Matyas","min":-10,"max":10,"optimum":0.0,"fixed_dims":2},
    "mccormick":{"label":"McCormick","min":-3,"max":4,"optimum":-1.913,"fixed_dims":2},
    "michalewicz":{"label":"Michalewicz","min":0,"max":3.14159},
    "modified_schwefel":{"label":"Modified Schwefel","min":-500,"max":500,"optimum":0.0},
    "perm":{"label":"Perm","min":-10,"max":10,"optimum":0.0},
    "pinter":{"label":"Pintér","min":-10,"max":10,"optimum":0.0},
    "powell":{"label":"Powell","min":-4,"max":5,"optimum":0.0,"fixed_dims":4},
    "qing":{"label":"Qing","min":-500,"max":500,"optimum":0.0},
    "quintic":{"label":"Quintic","min":-10,"max":10,"optimum":0.0},
    "rastrigin":{"label":"Rastrigin","min":-5.12,"max":5.12,"optimum":0.0},
    "ridge":{"label":"Ridge","min":-5,"max":5},
    "rosenbrocks_valley":{"label":"Rosenbrock","min":-5,"max":10,"optimum":0.0},
    "salomon":{"label":"Salomon","min":-100,"max":100,"optimum":0.0},
    "schaffer_2":{"label":"Schaffer N.2","min":-100,"max":100,"optimum":0.0,"fixed_dims":2},
    "schaffer_4":{"label":"Schaffer N.4","min":-100,"max":100,"fixed_dims":2},
    "schaffer_6":{"label":"Schaffer N.6","min":-100,"max":100,"optimum":0.0,"fixed_dims":2},
    "schumer_steiglitz":{"label":"Schumer-Steiglitz","min":-100,"max":100,"optimum":0.0},
    "schwefel":{"label":"Schwefel","min":-500,"max":500,"optimum":0.0},
    "schwefel_221":{"label":"Schwefel 2.21","min":-100,"max":100,"optimum":0.0},
    "schwefel_222":{"label":"Schwefel 2.22","min":-10,"max":10,"optimum":0.0},
    "six_hump_camel_back":{"label":"Six-Hump Camel Back","min":-5,"max":5,"optimum":-1.032,"fixed_dims":2},
    "sphere_2":{"label":"Sphere","min":-5.12,"max":5.12,"optimum":0.0},
    "sphere_3":{"label":"Sphere 3","min":-5.12,"max":5.12,"optimum":0.0},
    "step":{"label":"Step","min":-5.12,"max":5.12,"optimum":0.0},
    "step_2":{"label":"Step 2","min":-5.12,"max":5.12},
    "step_3":{"label":"Step 3","min":-5.12,"max":5.12},
    "stepint":{"label":"Stepint","min":-5.12,"max":5.12},
    "styblinski_tang":{"label":"Styblinski-Tang","min":-5,"max":5,"optimum":-39.166},
    "three_hump_camel_back":{"label":"Three-Hump Camel Back","min":-5,"max":5,"optimum":0.0,"fixed_dims":2},
    "trid":{"label":"Trid","min":-100,"max":100},
    "weierstrass":{"label":"Weierstrass","min":-0.5,"max":0.5,"optimum":0.0},
    "whitley":{"label":"Whitley","min":-10.24,"max":10.24,"optimum":0.0},
    "zakharov":{"label":"Zakharov","min":-5,"max":10,"optimum":0.0},
}
_EXCLUDE = {"get_test_function", "list_test_functions"}


def _catalogue() -> list[dict]:
    names: list[str] = []
    if hasattr(_tf, "list_test_functions"):
        try:
            names = list(_tf.list_test_functions(include_engineering=False))
        except TypeError:
            names = list(_tf.list_test_functions())
        except Exception:
            pass
    if not names and hasattr(_tf, "FUNCTIONS") and isinstance(_tf.FUNCTIONS, dict):
        names = list(_tf.FUNCTIONS.keys())
    if hasattr(_tf, "list_engineering_benchmarks"):
        try:
            engineering_ids = set(_tf.list_engineering_benchmarks())
            names = [name for name in names if name not in engineering_ids]
        except Exception:
            pass
    if not names:
        names = [n for n, o in inspect.getmembers(_tf, inspect.isfunction)
                 if not n.startswith("_") and n not in _EXCLUDE]
    out = []
    for n in sorted(names):
        if not hasattr(_tf, n):
            continue
        m = _KNOWN.get(n, {})
        out.append({"id": n, "label": m.get("label", n.replace("_", " ").title()),
                    "min": m.get("min", -100.0), "max": m.get("max", 100.0),
                    "optimum": m.get("optimum"), "fixed_dims": m.get("fixed_dims")})
    return out


# ── Progress callback ─────────────────────────────────────────────────────────
class _CB:
    """Minimal callback — works whether the package uses a Callback base class or not."""

    def __init__(self, state: dict) -> None:
        self._s = state
        self._lk = threading.Lock()

    def after_iteration(self, population, fitness, best_x, best_fitness, **kw) -> None:
        st  = kw.get("state")
        obs = kw.get("observation", {}) or {}
        step  = getattr(st, "step", 0) if st else obs.get("step", 0)
        evals = getattr(st, "evaluations", 0) if st else obs.get("evaluations", 0)
        bf    = float(best_fitness)
        diversity        = obs.get("diversity")
        exploitation_ratio = obs.get("exploitation_ratio", obs.get("exploitation"))

        with self._lk:
            self._s["step"]          = step
            self._s["evaluations"]   = evals
            self._s["best_fitness"]  = bf
            self._s["best_position"] = [float(v) for v in best_x]
            h = self._s["history"]
            # thin history to ≤ 3000 points
            if len(h) < 3000 or step % max(1, step // 3000) == 0:
                entry: dict = {"step": step, "fitness": bf}
                if diversity is not None:
                    entry["diversity"] = float(diversity)
                if exploitation_ratio is not None:
                    entry["exploitation_ratio"] = float(exploitation_ratio)
                if evals:
                    entry["evaluations"] = evals
                h.append(entry)

        if self._s.get("_cancel"):
            # Signal stop — try common stop mechanisms
            raise StopIteration("user_cancelled")


def _make_cb(state: dict):
    """Return a callback compatible with whichever interface pymetaheuristic uses."""
    cb = _CB(state)
    # If the package requires a Callback subclass, wrap it
    try:
        from pymetaheuristic.src.callbacks import Callback  # type: ignore
        class _Wrapped(Callback):
            def after_iteration(self, *a, **kw):
                cb.after_iteration(*a, **kw)
        return _Wrapped()
    except Exception:
        return cb


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_fn(name: str, code: str = ""):
    if name == "custom":
        compiled = compile(code, "<custom>", "exec")
        def fn(v):
            ns = {"variables_values": v}
            exec(compiled, ns)  # noqa: S102
            if "result" not in ns:
                raise NameError("Custom code must assign to `result`.")
            return float(ns["result"])
        return fn
    fn = getattr(_tf, name, None)
    if fn is None:
        fn = getattr(_tf, "FUNCTIONS", {}).get(name)
    if fn is None:
        raise ValueError(f"Unknown test function: {name!r}")
    return fn



def _compile_custom_function(code: str):
    compiled = compile(code or "", "<custom_function>", "exec")
    def fn(v):
        ns = {"variables_values": v, "np": np}
        exec(compiled, ns)  # noqa: S102
        if "result" not in ns:
            raise NameError("Custom function code must assign to `result`.")
        return float(ns["result"])
    return fn


def _function_specs(req: dict, default_dims: int | None = None) -> list[dict]:
    """Return executable function specs from built-ins, custom functions, or explicit problem_specs.

    problem_specs is used by the web UI for collaborative/orchestrated modes so each
    selected problem can carry its own dimensions and bounds. Legacy fields
    functions/custom_functions remain supported for backward compatibility.
    """
    specs: list[dict] = []

    # New explicit problem-set contract: one problem = one complete spec.
    # Accepted shape for built-ins:
    #   {type: "builtin", function: "ackley", label: "Ackley", dimensions: 30, min: -32.768, max: 32.768}
    # Accepted shape for custom functions:
    #   {type: "custom", id: "my_fn", label: "My Fn", code: "...", dimensions: 10, min: -5, max: 5}
    problem_specs = list(req.get("problem_specs") or [])
    if problem_specs:
        for idx, ps in enumerate(problem_specs, start=1):
            ptype = str(ps.get("type") or "builtin").lower()
            if ptype == "custom":
                cid = str(ps.get("id") or f"custom_{idx}")
                dims = int(ps.get("dimensions") or default_dims or req.get("dimensions", 10) or 10)
                specs.append({
                    "id": cid,
                    "label": ps.get("label") or cid,
                    "fn": _compile_custom_function(ps.get("code", "")),
                    "min": float(ps.get("min", (req.get("min_values") or [-5.12])[0])),
                    "max": float(ps.get("max", (req.get("max_values") or [5.12])[0])),
                    "dims": dims,
                    "optimum": ps.get("optimum"),
                })
                continue

            fn_id = str(ps.get("function") or ps.get("id") or "")
            if not fn_id:
                raise ValueError(f"Problem spec {idx} has no built-in function id.")
            fn = _resolve_fn(fn_id)
            meta = _KNOWN.get(fn_id, {})
            dims = int(ps.get("dimensions") or meta.get("fixed_dims") or default_dims or req.get("dimensions", 10) or 10)
            specs.append({
                "id": fn_id,
                "label": ps.get("label") or meta.get("label", fn_id),
                "fn": fn,
                "min": float(ps.get("min", meta.get("min", (req.get("min_values") or [-100.0])[0]))),
                "max": float(ps.get("max", meta.get("max", (req.get("max_values") or [100.0])[0]))),
                "dims": dims,
                "optimum": ps.get("optimum", meta.get("optimum")),
            })
        return specs

    fn_ids = list(req.get("functions") or [])
    # Backward-compatible fallback for single-objective requests.
    if not fn_ids and req.get("target_function") and not req.get("custom_functions"):
        fn_ids = [req["target_function"]]

    for fn_id in fn_ids:
        if fn_id == "custom":
            dims = default_dims or len(req.get("min_values") or []) or int(req.get("dimensions", 10) or 10)
            specs.append({
                "id": "custom",
                "label": "Custom",
                "fn": _compile_custom_function(req.get("custom_code", "")),
                "min": float((req.get("min_values") or [-5.12])[0]),
                "max": float((req.get("max_values") or [5.12])[0]),
                "dims": dims,
            })
            continue
        fn = _resolve_fn(fn_id)
        meta = _KNOWN.get(fn_id, {})
        dims = int(meta.get("fixed_dims") or default_dims or req.get("dimensions", 10) or len(req.get("min_values") or []) or 10)
        specs.append({
            "id": fn_id,
            "label": meta.get("label", fn_id),
            "fn": fn,
            "min": float(meta.get("min", (req.get("min_values") or [-100.0])[0])),
            "max": float(meta.get("max", (req.get("max_values") or [100.0])[0])),
            "dims": dims,
        })

    for idx, cf in enumerate(req.get("custom_functions") or [], start=1):
        cid = str(cf.get("id") or f"custom_{idx}")
        dims = int(cf.get("dimensions") or default_dims or req.get("dimensions", 10) or len(req.get("min_values") or []) or 10)
        specs.append({
            "id": cid,
            "label": cf.get("label") or cid,
            "fn": _compile_custom_function(cf.get("code", "")),
            "min": float(cf.get("min", (req.get("min_values") or [-5.12])[0])),
            "max": float(cf.get("max", (req.get("max_values") or [5.12])[0])),
            "dims": dims,
            "optimum": cf.get("optimum"),
        })
    return specs

def _parse_constraints(text: str) -> list:
    """
    Parse constraint expressions from a textarea (one per line).
    Lines starting with # are skipped.
    Returns a list of callables c(x) where c(x) <= 0 means satisfied.
    """
    constraints = []
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        try:
            compiled = compile(line, "<constraint>", "eval")
            def make_fn(c):
                def fn(x):
                    result = eval(c, {"x": x, "np": np, "__builtins__": {}})  # noqa: S307
                    if isinstance(result, dict):
                        # equality constraint: {"type": "eq", "value": ...}
                        return result
                    return float(result)
                return fn
            constraints.append(make_fn(compiled))
        except Exception as e:
            raise ValueError(f"Cannot parse constraint: {line!r} — {e}") from e
    return constraints


def _new_state(mode: str = "single") -> dict:
    return {
        "mode": mode, "status": "running",
        "step": 0, "evaluations": 0,
        "best_fitness": None, "best_position": None,
        "history": [], "result": None, "error": None,
        "start_time": time.time(), "elapsed": 0.0,
        "termination_reason": None, "_cancel": False,
        "metadata": {},
        "population_snapshots": [],
        # collab extras
        "per_island_history": {}, "migration_events": [],
        "island_telemetry": {}, "hall_of_fame": [],
        "island_summary": {}, "decisions": [], "outcomes": [],
        # benchmark extras
        "status_text": "", "partial_rows": [],
    }


def _safe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    """Recursively convert NumPy/pymetaheuristic scalar outputs into JSON-safe Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ── ① SINGLE ─────────────────────────────────────────────────────────────────
def _run_single(req: dict, state: dict) -> None:
    try:
        sub_mode = req.get("sub_mode", "standard")
        fn       = _resolve_fn(req["target_function"], req.get("custom_code", ""))
        cb       = _make_cb(state)
        store_snaps = req.get("store_population_snapshots", False)

        common = dict(
            algorithm       = req["algorithm"],
            target_function = fn,
            min_values      = req["min_values"],
            max_values      = req["max_values"],
            objective       = req.get("objective", "min"),
            max_steps       = req.get("max_steps", 500),
            max_evaluations = req.get("max_evaluations") or None,
            target_fitness  = req.get("target_fitness")  or None,
            timeout_seconds = req.get("timeout_seconds") or None,
            seed            = req.get("seed") or None,
            verbose         = False,
            store_history   = True,
            store_population_snapshots = store_snaps,
        )
        # merge algorithm-specific params
        common.update(req.get("params", {}))

        # ── Standard ─────────────────────────────────────────────────
        if sub_mode == "standard":
            try:
                res = optimize(**common, callbacks=cb)
            except StopIteration:
                state["status"] = "cancelled"
                return

        # ── Constrained ──────────────────────────────────────────────
        elif sub_mode == "constrained":
            constraints_text = req.get("constraints_text", "")
            try:
                constraints = _parse_constraints(constraints_text)
            except ValueError as e:
                state.update({"status": "error", "error": str(e),
                              "elapsed": time.time() - state["start_time"]})
                return
            handler = req.get("constraint_handler", "deb")
            coeff   = req.get("penalty_coeff", 1e6)
            try:
                res = optimize(
                    **common,
                    callbacks          = cb,
                    constraints        = constraints if constraints else None,
                    constraint_handler = handler,
                    penalty_coefficient = coeff,
                )
            except StopIteration:
                state["status"] = "cancelled"
                return

        # ── Binary ───────────────────────────────────────────────────
        elif sub_mode == "binary":
            transfer_fn = req.get("transfer_fn", "v2")
            init_name   = req.get("init_name", "") or None
            engine_kwargs = {k: v for k, v in common.items()
                             if k not in ("algorithm", "target_function", "callbacks",
                                          "store_history", "store_population_snapshots")}
            try:
                engine = create_optimizer(
                    algorithm       = req["algorithm"],
                    target_function = fn,
                    **engine_kwargs,
                    **({"init_name": init_name} if init_name else {}),
                )
                adapter = BinaryAdapter(engine, transfer_fn=transfer_fn)
                res = adapter.run()
            except StopIteration:
                state["status"] = "cancelled"
                return

        else:
            state.update({"status": "error",
                          "error": f"Unknown sub_mode: {sub_mode!r}",
                          "elapsed": time.time() - state["start_time"]})
            return

        # ── Flush final history point ─────────────────────────────────
        h = state["history"]
        if not h or h[-1].get("step") != res.steps:
            entry: dict = {"step": res.steps, "fitness": res.best_fitness}
            if res.steps:
                entry["evaluations"] = res.evaluations
            h.append(entry)

        # ── Metadata ──────────────────────────────────────────────────
        meta = res.metadata if isinstance(res.metadata, dict) else {}
        metadata: dict = {
            "mean_diversity":      _safe_float(meta.get("mean_diversity")),
            "exploitation_ratio":  _safe_float(meta.get("exploitation_ratio")),
            "elapsed_time":        _safe_float(meta.get("elapsed_time")),
            "best_raw_fitness":    _safe_float(meta.get("best_raw_fitness")),
            "best_violation":      _safe_float(meta.get("best_violation")),
            "best_is_feasible":    meta.get("best_is_feasible"),
            "binary_best_position": meta.get("binary_best_position"),
        }

        # ── Population snapshots ──────────────────────────────────────
        snapshots = []
        if store_snaps and hasattr(res, "population_snapshots") and res.population_snapshots:
            for snap in res.population_snapshots:
                step_n = snap.get("step", 0) if isinstance(snap, dict) else getattr(snap, "step", 0)
                pop    = snap.get("population", []) if isinstance(snap, dict) else getattr(snap, "population", [])
                snapshots.append({
                    "step": step_n,
                    "population": [
                        {"position": [float(v) for v in (a.get("position", []) if isinstance(a, dict) else getattr(a, "position", []))],
                         "fitness":  float(a.get("fitness", 0) if isinstance(a, dict) else getattr(a, "fitness", 0))}
                        for a in pop
                    ],
                })

        state.update({
            "status":               "done",
            "best_fitness":         float(res.best_fitness),
            "best_position":        [float(v) for v in res.best_position],
            "step":                 res.steps,
            "evaluations":          res.evaluations,
            "termination_reason":   res.termination_reason,
            "elapsed":              time.time() - state["start_time"],
            "metadata":             metadata,
            "population_snapshots": snapshots,
            "result":               {"best_fitness": float(res.best_fitness),
                                     "best_position": [float(v) for v in res.best_position],
                                     "metadata": metadata},
        })

    except Exception:
        state.update({"status": "error",
                      "error": traceback.format_exc(limit=10),
                      "elapsed": time.time() - state["start_time"]})


# ── ② COLLABORATIVE ───────────────────────────────────────────────────────────
def _build_islands(req: dict) -> list[dict]:
    islands = []
    for isl in req["islands"]:
        islands.append({
            "algorithm": isl["algorithm"],
            "label":     isl.get("label") or isl["algorithm"],
            "config":    isl.get("config", {}),
        })
    return islands


def _run_collaborative(req: dict, state: dict) -> None:
    try:
        islands = _build_islands(req)
        specs = _function_specs(req, default_dims=req.get("dimensions") or len(req.get("min_values") or []) or 10)
        if not specs:
            raise ValueError("No target function selected.")

        rows: list[dict] = []
        last_result = None
        base_seed = req.get("seed") or None
        for i, spec in enumerate(specs):
            if state.get("_cancel"):
                state["status"] = "cancelled"
                return
            t0 = time.time()
            try:
                result = cooperative_optimize(
                    islands            = islands,
                    target_function    = spec["fn"],
                    min_values         = [float(spec["min"])] * int(spec["dims"]),
                    max_values         = [float(spec["max"])] * int(spec["dims"]),
                    objective          = req.get("objective", "min"),
                    max_steps          = req.get("max_steps", 20),
                    migration_interval = req.get("migration_interval", 2),
                    migration_size     = req.get("migration_size", 4),
                    topology           = req.get("topology", "star"),
                    seed               = (int(base_seed) + i if base_seed is not None else None),
                    verbose            = False,
                )
                last_result = result
                rows.append({
                    "function": spec["id"],
                    "fn_label": spec.get("label", spec["id"]),
                    "best_fitness": float(result.best_fitness),
                    "steps": req.get("max_steps", 20),
                    "elapsed_s": round(time.time() - t0, 3),
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "function": spec["id"],
                    "fn_label": spec.get("label", spec["id"]),
                    "best_fitness": None,
                    "steps": None,
                    "elapsed_s": round(time.time() - t0, 3),
                    "error": str(e),
                })
            state["step"] = i + 1
            state["status_text"] = f"{spec.get('label', spec['id'])} — {i+1}/{len(specs)}"
            state["partial_rows"] = rows[:]

        if last_result is None:
            state.update({"status": "done", "elapsed": time.time() - state["start_time"], "result": {"rows": rows}})
            return

        _finish_collab(last_result, req, state, is_orchestrated=False)
        state["partial_rows"] = rows[:]
        state.setdefault("result", {})["rows"] = rows
        state["result"]["functions"] = [s["id"] for s in specs]

    except Exception:
        state.update({"status": "error",
                      "error": traceback.format_exc(limit=10),
                      "elapsed": time.time() - state["start_time"]})


# ── ③ ORCHESTRATED ────────────────────────────────────────────────────────────
def _run_orchestrated(req: dict, state: dict) -> None:
    try:
        orch_cfg = req.get("orchestration", {})
        rules_cfg = req.get("rules", {})

        config = CollaborativeConfig(
            orchestration = OrchestrationSpec(
                mode                       = orch_cfg.get("mode", "rules"),
                checkpoint_interval        = int(orch_cfg.get("checkpoint_interval", 5)),
                max_actions_per_checkpoint = int(orch_cfg.get("max_actions_per_checkpoint", 2)),
                warmup_checkpoints         = int(orch_cfg.get("warmup_checkpoints", 1)),
            ),
            rules = RulesConfig(
                stagnation_threshold     = int(rules_cfg.get("stagnation_threshold", 1)),
                low_diversity_threshold  = float(rules_cfg.get("low_diversity_threshold", 0.05)),
                high_diversity_threshold = float(rules_cfg.get("high_diversity_threshold", 0.25)),
                perturbation_sigma       = float(rules_cfg.get("perturbation_sigma", 0.05)),
            ),
        )

        islands = _build_islands(req)
        specs = _function_specs(req, default_dims=req.get("dimensions") or len(req.get("min_values") or []) or 10)
        if not specs:
            raise ValueError("No target function selected.")

        rows: list[dict] = []
        last_result = None
        base_seed = req.get("seed") or None
        for i, spec in enumerate(specs):
            if state.get("_cancel"):
                state["status"] = "cancelled"
                return
            t0 = time.time()
            try:
                result = orchestrated_optimize(
                    islands         = islands,
                    target_function = spec["fn"],
                    min_values      = [float(spec["min"])] * int(spec["dims"]),
                    max_values      = [float(spec["max"])] * int(spec["dims"]),
                    objective       = req.get("objective", "min"),
                    max_steps       = req.get("max_steps", 20),
                    seed            = (int(base_seed) + i if base_seed is not None else None),
                    config          = config,
                    verbose         = False,
                )
                last_result = result
                rows.append({
                    "function": spec["id"],
                    "fn_label": spec.get("label", spec["id"]),
                    "best_fitness": float(result.best_fitness),
                    "steps": req.get("max_steps", 20),
                    "elapsed_s": round(time.time() - t0, 3),
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "function": spec["id"],
                    "fn_label": spec.get("label", spec["id"]),
                    "best_fitness": None,
                    "steps": None,
                    "elapsed_s": round(time.time() - t0, 3),
                    "error": str(e),
                })
            state["step"] = i + 1
            state["status_text"] = f"{spec.get('label', spec['id'])} — {i+1}/{len(specs)}"
            state["partial_rows"] = rows[:]

        if last_result is None:
            state.update({"status": "done", "elapsed": time.time() - state["start_time"], "result": {"rows": rows}})
            return

        _finish_collab(last_result, req, state, is_orchestrated=True)
        state["partial_rows"] = rows[:]
        state.setdefault("result", {})["rows"] = rows
        state["result"]["functions"] = [s["id"] for s in specs]

    except Exception:
        state.update({"status": "error",
                      "error": traceback.format_exc(limit=10),
                      "elapsed": time.time() - state["start_time"]})


def _finish_collab(result, req: dict, state: dict, *, is_orchestrated: bool) -> None:
    """Shared post-processing for both cooperative and orchestrated results."""
    obj = req.get("objective", "min")

    # ── Global convergence: best-so-far from history ──────────────────────
    global_conv: list[dict] = []
    best_so_far: float | None = None
    for rec in (getattr(result, "history", None) or []):
        step = rec.get("global_step", rec.get("step", len(global_conv))) if isinstance(rec, dict) else getattr(rec, "global_step", len(global_conv))
        bf   = rec.get("best_fitness") if isinstance(rec, dict) else getattr(rec, "best_fitness", None)
        if bf is None:
            continue
        bf = float(bf)
        if best_so_far is None or (obj == "min" and bf < best_so_far) or (obj == "max" and bf > best_so_far):
            best_so_far = bf
        global_conv.append({"step": step, "fitness": best_so_far})

    # ── Per-island convergence + telemetry from island_telemetry ─────────
    per_island: dict[str, list[dict]] = {}
    island_telem: dict[str, list[dict]] = {}
    raw_telem = getattr(result, "island_telemetry", None) or {}
    if isinstance(raw_telem, dict):
        for label, records in raw_telem.items():
            per_island[label] = []
            island_telem[label] = []
            for r in (records or []):
                gs   = getattr(r, "global_step", 0) if hasattr(r, "global_step") else r.get("global_step", 0)
                bf_r = getattr(r, "best_fitness", None) if hasattr(r, "best_fitness") else r.get("best_fitness")
                div  = getattr(r, "diversity", None)    if hasattr(r, "diversity")    else r.get("diversity")
                hlth = getattr(r, "health", None)       if hasattr(r, "health")       else r.get("health")
                stag = getattr(r, "stagnation_steps",0) if hasattr(r, "stagnation_steps") else r.get("stagnation_steps", 0)
                if bf_r is not None:
                    per_island[label].append({"step": gs, "fitness": float(bf_r)})
                island_telem[label].append({
                    "step": gs, "global_step": gs,
                    "diversity": _safe_float(div),
                    "health":    _safe_float(hlth),
                    "stagnation_steps": stag,
                })

    # ── Migration events ──────────────────────────────────────────────────
    events: list[dict] = []
    for e in (getattr(result, "events", None) or []):
        if isinstance(e, dict):
            events.append({"step": e.get("global_step", 0), "from": e.get("source_label",""),
                           "to": e.get("target_label",""), "migrants": e.get("migrants",0),
                           "fit_after": e.get("best_fitness_after")})
        else:
            events.append({"step":      getattr(e,"global_step",0),
                           "from":      getattr(e,"source_label",""),
                           "to":        getattr(e,"target_label",""),
                           "migrants":  getattr(e,"migrants",0),
                           "fit_after": getattr(e,"best_fitness_after",None)})

    # ── Hall of fame ──────────────────────────────────────────────────────
    hof_raw = getattr(result, "hall_of_fame", None) or []
    hof = []
    for h in hof_raw[:10]:
        if isinstance(h, dict):
            hof.append(h)
        else:
            hof.append({
                "label":     getattr(h,"label",""),
                "algorithm": getattr(h,"algorithm",""),
                "fitness":   _safe_float(getattr(h,"fitness",None) or getattr(h,"best_fitness",None)),
            })

    # ── Island summary ────────────────────────────────────────────────────
    island_summary: dict[str, float] = {}
    ir = getattr(result, "island_results", None) or {}
    if isinstance(ir, dict):
        for label, res_i in ir.items():
            bf = getattr(res_i,"best_fitness",None) if not isinstance(res_i,dict) else res_i.get("best_fitness")
            if bf is not None:
                island_summary[label] = float(bf)
    # fallback: use last known telemetry fitness
    if not island_summary:
        for label, pts in per_island.items():
            if pts:
                island_summary[label] = pts[-1]["fitness"]

    # ── Orchestrated extras: decisions + outcomes ─────────────────────────
    decisions: list[dict] = []
    outcomes: list[list[dict]] = []
    if is_orchestrated:
        for dec in (getattr(result, "decisions", None) or []):
            if isinstance(dec, dict):
                decisions.append(dec)
            else:
                actions = getattr(dec, "actions", None) or []
                decisions.append({
                    "controller_mode": getattr(dec,"controller_mode",""),
                    "controller_name": getattr(dec,"controller_name",""),
                    "confidence":      _safe_float(getattr(dec,"confidence",None)),
                    "reasoning":       getattr(dec,"reasoning",""),
                    "n_actions":       len(actions),
                })
        for cp_outcomes in (getattr(result, "outcomes", None) or []):
            cp_list = []
            for out in (cp_outcomes or []):
                if isinstance(out, dict):
                    cp_list.append(out)
                else:
                    action = getattr(out, "action", None)
                    cp_list.append({
                        "action_type":        getattr(action,"type","") if action else "",
                        "source_label":       getattr(action,"source_label","") if action else "",
                        "target_label":       getattr(action,"target_label","") if action else "",
                        "executed":           getattr(out,"executed",False),
                        "status":             getattr(out,"status",""),
                        "target_fitness_after": _safe_float(getattr(out,"target_fitness_after",None)),
                    })
            outcomes.append(cp_list)

    state.update({
        "status":             "done",
        "best_fitness":       float(result.best_fitness),
        "best_position":      [float(v) for v in result.best_position],
        "history":            global_conv,
        "step":               req.get("max_steps", 20),
        "per_island_history": per_island,
        "island_telemetry":   island_telem,
        "migration_events":   events,
        "hall_of_fame":       hof,
        "island_summary":     island_summary,
        "decisions":          decisions,
        "outcomes":           outcomes,
        "elapsed":            time.time() - state["start_time"],
        "result": {
            "best_fitness":   float(result.best_fitness),
            "best_position":  [float(v) for v in result.best_position],
            "island_summary": island_summary,
            "migration_count": len(events),
            "n_islands":      len(req.get("islands", [])),
            "topology":       req.get("topology", "star"),
        },
    })


# ── ④ BENCHMARK ───────────────────────────────────────────────────────────────
def _run_benchmark(req: dict, state: dict) -> None:
    try:
        algorithms = req["algorithms"]
        n_trials   = int(req.get("n_trials", 5))
        max_steps  = int(req.get("max_steps", 300))
        dims       = int(req.get("dimensions", 10))
        objective  = req.get("objective", "min")
        base_seed  = int(req.get("seed") or 0)
        alg_configs = req.get("algorithm_configs", {}) or {}
        fn_specs = _function_specs(req, default_dims=dims)
        rows: list[dict] = []
        total = len(algorithms) * len(fn_specs) * n_trials
        done  = 0

        for spec in fn_specs:
            fn_id = spec["id"]
            fn = spec["fn"]
            d = int(spec["dims"])
            bmin = [float(spec["min"])] * d
            bmax = [float(spec["max"])] * d
            label = spec.get("label", fn_id)

            for alg_id in algorithms:
                alg_params = dict(alg_configs.get(alg_id, {}) or {})
                for trial in range(n_trials):
                    if state.get("_cancel"):
                        state["status"] = "cancelled"
                        return
                    t0 = time.time()
                    try:
                        res = optimize(
                            algorithm       = alg_id,
                            target_function = fn,
                            min_values      = bmin,
                            max_values      = bmax,
                            objective       = objective,
                            max_steps       = max_steps,
                            seed            = base_seed + trial,
                            verbose         = False,
                            **alg_params,
                        )
                        rows.append({
                            "algorithm":    alg_id,
                            "function":     fn_id,
                            "fn_label":     label,
                            "trial":        trial + 1,
                            "best_fitness": float(res.best_fitness),
                            "steps":        res.steps,
                            "evaluations":  res.evaluations,
                            "elapsed_s":    round(time.time() - t0, 3),
                            "error":        None,
                        })
                    except Exception as e:
                        rows.append({
                            "algorithm": alg_id, "function": fn_id,
                            "fn_label": label,
                            "trial": trial + 1, "best_fitness": None,
                            "steps": None, "evaluations": None,
                            "elapsed_s": round(time.time() - t0, 3),
                            "error": str(e),
                        })
                    done += 1
                    state["step"]         = done
                    state["status_text"]  = f"{alg_id} × {fn_id} — trial {trial+1}/{n_trials}"
                    state["partial_rows"] = rows[:]

        state.update({
            "status":  "done",
            "elapsed": time.time() - state["start_time"],
            "result":  {
                "rows":       rows,
                "algorithms": algorithms,
                "functions":  [spec["id"] for spec in fn_specs],
                "n_trials":   n_trials,
                "dimensions": dims,
            },
        })

    except Exception:
        state.update({"status": "error",
                      "error": traceback.format_exc(limit=10),
                      "elapsed": time.time() - state["start_time"]})


# ── Pydantic models ───────────────────────────────────────────────────────────
class SingleReq(BaseModel):
    algorithm:                 str
    target_function:           str
    custom_code:               str   = ""
    min_values:                list[float]
    max_values:                list[float]
    objective:                 str   = "min"
    max_steps:                 int   = 500
    max_evaluations:           int   | None = None
    target_fitness:            float | None = None
    timeout_seconds:           float | None = None
    seed:                      int   | None = None
    params:                    dict[str, Any] = Field(default_factory=dict)
    store_history:             bool  = True
    store_population_snapshots:bool  = False
    # sub-mode fields
    sub_mode:                  str   = "standard"         # standard | constrained | binary
    constraint_handler:        str   = "deb"
    penalty_coeff:             float = 1e6
    constraints_text:          str   = ""
    transfer_fn:               str   = "v2"
    init_name:                 str   = "chaotic:tent"


class IslandDef(BaseModel):
    algorithm: str
    label:     str            = ""
    config:    dict[str, Any] = Field(default_factory=dict)


class CollabReq(BaseModel):
    islands:            list[IslandDef]
    target_function:    str
    custom_code:        str   = ""
    functions:          list[str] = Field(default_factory=list)
    custom_functions:   list[dict[str, Any]] = Field(default_factory=list)
    problem_specs:      list[dict[str, Any]] = Field(default_factory=list)
    dimensions:         int | None = None
    min_values:         list[float]
    max_values:         list[float]
    objective:          str   = "min"
    max_steps:          int   = 20
    migration_interval: int   = 2
    migration_size:     int   = 4
    topology:           str   = "star"
    seed:               int   | None = None


class OrchReq(BaseModel):
    islands:         list[IslandDef]
    target_function: str
    custom_code:     str   = ""
    functions:       list[str] = Field(default_factory=list)
    custom_functions:list[dict[str, Any]] = Field(default_factory=list)
    problem_specs:   list[dict[str, Any]] = Field(default_factory=list)
    dimensions:      int | None = None
    min_values:      list[float]
    max_values:      list[float]
    objective:       str   = "min"
    max_steps:       int   = 20
    seed:            int   | None = None
    orchestration:   dict[str, Any] = Field(default_factory=dict)
    rules:           dict[str, Any] = Field(default_factory=dict)


class BenchmarkReq(BaseModel):
    algorithms: list[str]
    functions:  list[str]
    custom_functions: list[dict[str, Any]] = Field(default_factory=list)
    problem_specs: list[dict[str, Any]] = Field(default_factory=list)
    algorithm_configs: dict[str, Any] = Field(default_factory=dict)
    n_trials:   int   = 5
    max_steps:  int   = 300
    dimensions: int   = 10
    objective:  str   = "min"
    seed:       int   | None = 0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse((Path(__file__).parent / "app.html").read_text(encoding="utf-8"))


@app.get("/api/algorithms")
def get_algs() -> list[dict]:
    out = []
    for alg_id in list_algorithms():
        info = get_algorithm_info(alg_id)
        caps = info.get("capabilities")
        cd   = ({k: getattr(caps, k) for k in caps.__dataclass_fields__}
                if caps and hasattr(caps, "__dataclass_fields__") else {})
        out.append({
            "id":                 info["algorithm_id"],
            "name":               info["algorithm_name"],
            "family":             info.get("family"),
            "constraint_support": info.get("constraint_support"),
            "caps":               cd,
            "doi":                info.get("doi"),
            "defaults":           info.get("defaults", {}),
        })
    return out


@app.get("/api/algorithms/{alg_id}")
def get_alg(alg_id: str) -> dict:
    try:
        info = get_algorithm_info(alg_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    caps = info.get("capabilities")
    return {
        "id":                 info["algorithm_id"],
        "name":               info["algorithm_name"],
        "family":             info.get("family"),
        "constraint_support": info.get("constraint_support"),
        "defaults":           info.get("defaults", {}),
        "doi":                info.get("doi"),
        "caps": ({k: getattr(caps, k) for k in caps.__dataclass_fields__}
                 if caps and hasattr(caps, "__dataclass_fields__") else {}),
    }


@app.get("/api/test-functions")
def get_fns() -> list[dict]:
    return _catalogue()


@app.get("/api/transfer-functions")
def get_transfer_fns() -> list[str]:
    return list(_AVAILABLE_TRANSFER)


# ── Job creation ──────────────────────────────────────────────────────────────
def _start(mode: str, fn, req_data: dict) -> dict:
    jid   = uuid.uuid4().hex[:10]
    state = _new_state(mode)
    state["request"] = req_data
    _jobs[jid] = state
    threading.Thread(target=fn, args=(req_data, state), daemon=True).start()
    return {"job_id": jid}


@app.post("/api/jobs", status_code=202)
def create_single(req: SingleReq):
    return _start("single", _run_single, req.model_dump())


@app.post("/api/jobs/collaborative", status_code=202)
def create_collab(req: CollabReq):
    return _start("collaborative", _run_collaborative, req.model_dump())


@app.post("/api/jobs/orchestrated", status_code=202)
def create_orch(req: OrchReq):
    return _start("orchestrated", _run_orchestrated, req.model_dump())


@app.post("/api/jobs/benchmark", status_code=202)
def create_benchmark(req: BenchmarkReq):
    return _start("benchmark", _run_benchmark, req.model_dump())


# ── Polling ───────────────────────────────────────────────────────────────────
@app.get("/api/jobs/{jid}")
def poll(jid: str) -> dict:
    s = _jobs.get(jid)
    if not s:
        raise HTTPException(404, "Job not found")
    return _json_safe({
        "id":               jid,
        "mode":             s["mode"],
        "status":           s["status"],
        "step":             s["step"],
        "evaluations":      s["evaluations"],
        "best_fitness":     s["best_fitness"],
        "best_position":    s["best_position"],
        "history":          s["history"],
        "error":            s["error"],
        "elapsed":          round(time.time() - s["start_time"], 2),
        "termination_reason": s.get("termination_reason"),
        "metadata":           s.get("metadata", {}),
        "population_snapshots": s.get("population_snapshots", []),
        # collab extras
        "per_island_history": s.get("per_island_history", {}),
        "island_telemetry":   s.get("island_telemetry", {}),
        "migration_events":   s.get("migration_events", []),
        "hall_of_fame":       s.get("hall_of_fame", []),
        "island_summary":     s.get("island_summary", {}),
        "decisions":          s.get("decisions", []),
        "outcomes":           s.get("outcomes", []),
        # benchmark extras
        "status_text":        s.get("status_text", ""),
        "partial_rows":       s.get("partial_rows", []),
        "request":            s.get("request"),
    })


@app.delete("/api/jobs/{jid}")
def cancel(jid: str):
    s = _jobs.get(jid)
    if not s:
        raise HTTPException(404, "Job not found")
    s["_cancel"] = True
    return {"ok": True}


@app.get("/api/jobs/{jid}/result")
def get_result(jid: str):
    s = _jobs.get(jid)
    if not s:
        raise HTTPException(404, "Job not found")
    if s["status"] != "done":
        raise HTTPException(409, "Job not finished yet")
    return _json_safe(s["result"])
