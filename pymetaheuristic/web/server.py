"""
pyMetaheuristic Web UI — FastAPI Backend  v4
=============================================
Supports: Single Algorithm · Benchmark Runner · Benchmark Study · Island System

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
    from pymetaheuristic.src.schemas import (CollaborativeConfig, OrchestrationSpec, RulesConfig, BanditConfig, PortfolioConfig)
    from pymetaheuristic.src.islands import IslandSystem, Island, TopologyConfig, MigrationConfig, OrchestrationConfig
    from pymetaheuristic.src.benchmarks import BenchmarkStudy, BenchmarkProblem
    from pymetaheuristic.src.utils.chaotic import BinaryAdapter
    from pymetaheuristic.src.tuner import BenchmarkRunner
    from pymetaheuristic.src.termination import Termination
    import pymetaheuristic.src.test_functions as _tf
    from pymetaheuristic.src.evomapx import evomapx_analysis, explain_evomapx
    _AVAILABLE_TRANSFER = getattr(pmh, "AVAILABLE_TRANSFER_FUNCTIONS",
                                  ["v1","v2","v3","v4","s1","s2","s3","s4"])
except ImportError:
    try:
        from ..src.api import list_algorithms, get_algorithm_info, optimize, create_optimizer
        from ..src.cooperation import cooperative_optimize
        from ..src.orchestration import orchestrated_optimize
        from ..src.schemas import (CollaborativeConfig, OrchestrationSpec, RulesConfig, BanditConfig, PortfolioConfig)
        from ..src.islands import IslandSystem, Island, TopologyConfig, MigrationConfig, OrchestrationConfig
        from ..src.benchmarks import BenchmarkStudy, BenchmarkProblem
        from ..src.utils.chaotic import BinaryAdapter
        from ..src.tuner import BenchmarkRunner
        from ..src.termination import Termination
        from ..src import test_functions as _tf
        from ..src.evomapx import evomapx_analysis, explain_evomapx
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


_README_2D_IDS = {
    "ackley", "beale", "bohachevsky_1", "bohachevsky_2", "bohachevsky_3",
    "booth", "branin_rcos", "bukin_6", "cross_in_tray", "drop_wave", "easom",
    "eggholder", "goldstein_price", "himmelblau", "holder_table", "levi_13",
    "matyas", "mccormick", "schaffer_2", "schaffer_4", "schaffer_6",
    "six_hump_camel_back", "three_hump_camel_back",
}
_README_D_IDS = {
    "alpine_1", "alpine_2", "axis_parallel_hyper_ellipsoid", "bent_cigar",
    "chung_reynolds", "cosine_mixture", "csendes", "de_jong_1", "discus",
    "dixon_price", "elliptic", "expanded_griewank_plus_rosenbrock", "griewangk_8",
    "happy_cat", "hgbat", "katsuura", "levy", "michalewicz", "modified_schwefel",
    "perm", "pinter", "powell", "qing", "quintic", "rastrigin", "ridge",
    "rosenbrocks_valley", "salomon", "schumer_steiglitz", "schwefel", "schwefel_221",
    "schwefel_222", "sphere_2", "sphere_3", "step", "step_2", "step_3",
    "stepint", "styblinski_tang", "trid", "weierstrass", "whitley", "zakharov",
}
_CATEGORY_ORDER = {
    "2-Dimensional Functions": 10,
    "D-Dimensional Functions": 20,
    "CEC 2022 Functions": 30,
    "BBOB Functions": 40,
    "Engineering Design Benchmarks": 50,
    "Other Functions": 90,
}
_CEC_2022_OPTIMUMS = {
    "cec_2022_f01": 300.0, "cec_2022_f02": 400.0, "cec_2022_f03": 600.0,
    "cec_2022_f04": 800.0, "cec_2022_f05": 900.0, "cec_2022_f06": 1800.0,
    "cec_2022_f07": 2000.0, "cec_2022_f08": 2200.0, "cec_2022_f09": 2300.0,
    "cec_2022_f10": 2400.0, "cec_2022_f11": 2600.0, "cec_2022_f12": 2700.0,
}


try:
    _ENGINEERING_IDS = set(_tf.list_engineering_benchmarks())
except Exception:
    _ENGINEERING_IDS = set()


def _is_engineering_function(name: str) -> bool:
    return str(name or "").strip().lower() in _ENGINEERING_IDS


def _engineering_benchmark_info(name: str) -> dict | None:
    key = str(name or "").strip().lower()
    if not key or key not in _ENGINEERING_IDS:
        return None
    try:
        return dict(_tf.get_engineering_benchmark(key))
    except Exception:
        return None


def _bound_vector(value, dims: int, fallback) -> list[float]:
    if value is None:
        value = fallback
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(v) for v in list(value)]
        if len(vals) == 1:
            return vals * int(dims)
        if len(vals) != int(dims):
            raise ValueError(f"Expected {dims} bounds, received {len(vals)}.")
        return vals
    return [float(value)] * int(dims)


def _spec_bounds(spec: dict) -> tuple[list[float], list[float]]:
    dims = int(spec.get("dims") or len(spec.get("min_values") or []) or len(spec.get("max_values") or []) or 1)
    lo = spec.get("min_values")
    hi = spec.get("max_values")
    if lo is None:
        lo = spec.get("min")
    if hi is None:
        hi = spec.get("max")
    return _bound_vector(lo, dims, -100.0), _bound_vector(hi, dims, 100.0)


def _constraint_kwargs_from_spec(spec: dict) -> dict[str, Any]:
    constraints = spec.get("constraints")
    if not constraints:
        return {}
    return {
        "constraints": constraints,
        "constraint_handler": spec.get("constraint_handler") or "deb",
        "penalty_coefficient": float(spec.get("penalty_coefficient", 1.0e6) or 1.0e6),
    }


def _penalized_engineering_objective(fn, constraints, objective: str = "min", penalty_coefficient: float = 1.0e6):
    # BenchmarkStudy currently accepts BenchmarkProblem callables but does not
    # forward framework constraints.  Engineering problems therefore use a
    # conservative exterior-penalty wrapper in that specific study path.
    sign = -1.0 if objective == "max" else 1.0
    def wrapped(x):
        raw = float(fn(x))
        violation = 0.0
        for c in constraints or []:
            v = c(x)
            if isinstance(v, dict):
                if v.get("type") == "eq":
                    violation += abs(float(v.get("value", 0.0)))
                else:
                    violation += max(0.0, float(v.get("value", 0.0)))
            else:
                violation += max(0.0, float(v))
        return raw + sign * float(penalty_coefficient) * violation
    return wrapped


def _category_for_function(name: str) -> str:
    if _is_engineering_function(name):
        return "Engineering Design Benchmarks"
    if name.startswith("cec_2022_"):
        return "CEC 2022 Functions"
    if name.startswith("bbob_f"):
        return "BBOB Functions"
    if name in _README_2D_IDS:
        return "2-Dimensional Functions"
    if name in _README_D_IDS:
        return "D-Dimensional Functions"
    return "Other Functions"


def _catalogue_info(name: str) -> dict:
    info = getattr(_tf, "TEST_FUNCTIONS", {}).get(name, {}) or {}
    if not info and name.startswith("bbob_f"):
        info = getattr(_tf, "BBOB_METADATA", {}).get(name, {}) or {}
    return info


def _function_optimum_text(name: str, info: dict, meta: dict, optimum: Any) -> str:
    text = info.get("optimum") or meta.get("optimum_text") or ""
    if name.startswith("cec_2022_") and not text:
        text = f"f*={_CEC_2022_OPTIMUMS.get(name)} at the official shifted optimum."
    if name.startswith("bbob_f") and not text:
        text = "Use get_bbob_optimum(function_id, dimension, instance) for the shifted optimizer x* and f*."
    if text:
        return str(text)
    if optimum is not None:
        return f"f*={optimum}; optimizer x* not encoded in the catalogue."
    return "Global minimum not encoded in the catalogue."


def _catalogue() -> list[dict]:
    names: list[str] = []
    if hasattr(_tf, "list_test_functions"):
        try:
            names = list(_tf.list_test_functions(include_engineering=True))
        except TypeError:
            names = list(_tf.list_test_functions())
        except Exception:
            pass
    if not names and hasattr(_tf, "FUNCTIONS") and isinstance(_tf.FUNCTIONS, dict):
        names = list(_tf.FUNCTIONS.keys())
    if not names:
        names = [n for n, o in inspect.getmembers(_tf, inspect.isfunction)
                 if not n.startswith("_") and n not in _EXCLUDE]
    # Be explicit: engineering benchmarks are constrained Problem objects in the
    # UI and must remain discoverable even when list_test_functions() omits them.
    names = sorted(set(names) | set(_ENGINEERING_IDS))
    out = []
    for n in sorted(names):
        if not hasattr(_tf, n) and not _is_engineering_function(n):
            continue
        info = _catalogue_info(n)
        if _is_engineering_function(n):
            eng = _engineering_benchmark_info(n) or {}
            min_values = [float(v) for v in eng.get("min_values", [])]
            max_values = [float(v) for v in eng.get("max_values", [])]
            best_x = [float(v) for v in eng.get("best_known_position", [])]
            best_f = eng.get("best_known_fitness")
            try:
                best_f = float(best_f) if best_f is not None else None
            except Exception:
                best_f = None
            category = _category_for_function(n)
            label = eng.get("name") or info.get("name") or n.replace("_", " ").title()
            constraints_count = len(eng.get("constraints") or [])
            gm = info.get("optimum") or ""
            if not gm and best_f is not None:
                gm = f"f*≈{best_f}; x*≈{tuple(best_x)}"
            out.append({
                "id": n,
                "label": label,
                "category": category,
                "category_order": _CATEGORY_ORDER.get(category, 90),
                "min": min_values[0] if len(set(min_values)) == 1 and min_values else min_values,
                "max": max_values[0] if len(set(max_values)) == 1 and max_values else max_values,
                "min_values": min_values,
                "max_values": max_values,
                "domain": info.get("domain") or eng.get("notes"),
                "optimum": best_f,
                "optimum_value": best_f,
                "optimum_x": best_x,
                "global_minimum": str(gm) if gm else "Best-known engineering optimum not encoded in the catalogue.",
                "fixed_dims": len(min_values) if min_values else None,
                "is_engineering": True,
                "constraints_count": constraints_count,
                "constraint_handler": "deb",
                "notes": eng.get("notes"),
            })
            continue
        m = _KNOWN.get(n, {})
        category = _category_for_function(n)
        optimum = m.get("optimum")
        if optimum is None and n in _CEC_2022_OPTIMUMS:
            optimum = _CEC_2022_OPTIMUMS[n]
        min_bound = m.get("min")
        max_bound = m.get("max")
        if min_bound is None or max_bound is None:
            if n.startswith("bbob_f"):
                min_bound, max_bound = -5.0, 5.0
            elif n.startswith("cec_2022_"):
                min_bound, max_bound = -100.0, 100.0
            else:
                min_bound, max_bound = -100.0, 100.0
        label = m.get("label") or info.get("name") or n.replace("_", " ").title()
        out.append({
            "id": n,
            "label": label,
            "category": category,
            "category_order": _CATEGORY_ORDER.get(category, 90),
            "min": min_bound,
            "max": max_bound,
            "min_values": None,
            "max_values": None,
            "domain": info.get("domain"),
            "optimum": optimum,
            "optimum_value": optimum,
            "optimum_x": m.get("optimum_x"),
            "global_minimum": _function_optimum_text(n, info, m, optimum),
            "fixed_dims": m.get("fixed_dims"),
        })
    return sorted(out, key=lambda x: (x.get("category_order", 90), str(x.get("label", ""))))



def _df_records(obj) -> list[dict]:
    """Return a JSON-safe list of records from a pandas DataFrame-like object."""
    if obj is None:
        return []
    try:
        return _json_safe(obj.to_dict(orient="records"))
    except Exception:
        return []


def _compact_scientific_summary(summary: dict) -> dict:
    """Drop embedded DataFrames from BenchmarkResult.scientific_summary()."""
    out = dict(summary or {})
    out.pop("summary", None)
    out.pop("rank_table", None)
    return _json_safe(out)


def _benchmark_problem_from_spec(spec: dict, objective: str) -> BenchmarkProblem:
    dims = int(spec.get("dims", 10) or 10)
    min_values, max_values = _spec_bounds(spec)
    fn = spec["fn"]
    if spec.get("constraints"):
        fn = _penalized_engineering_objective(
            fn,
            spec.get("constraints") or [],
            objective=objective,
            penalty_coefficient=float(spec.get("penalty_coefficient", 1.0e6) or 1.0e6),
        )
    return BenchmarkProblem(
        function=fn,
        min_values=min_values,
        max_values=max_values,
        name=str(spec.get("label") or spec.get("id") or "problem"),
        objective=objective,
        optimum=spec.get("optimum"),
        metadata={"id": spec.get("id"), "dimension": dims, "is_engineering": bool(spec.get("is_engineering"))},
    )

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
                    "min": _bound_vector(ps.get("min_values", ps.get("min")), dims, (req.get("min_values") or [-5.12])[0])[0],
                    "max": _bound_vector(ps.get("max_values", ps.get("max")), dims, (req.get("max_values") or [5.12])[0])[0],
                    "min_values": _bound_vector(ps.get("min_values", ps.get("min")), dims, (req.get("min_values") or [-5.12])[0]),
                    "max_values": _bound_vector(ps.get("max_values", ps.get("max")), dims, (req.get("max_values") or [5.12])[0]),
                    "dims": dims,
                    "optimum": ps.get("optimum"),
                })
                continue

            fn_id = str(ps.get("function") or ps.get("id") or "")
            if not fn_id:
                raise ValueError(f"Problem spec {idx} has no built-in function id.")
            if _is_engineering_function(fn_id):
                eng = _engineering_benchmark_info(fn_id) or {}
                fn = eng.get("objective") or _resolve_fn(fn_id)
                min_values = [float(v) for v in eng.get("min_values", [])]
                max_values = [float(v) for v in eng.get("max_values", [])]
                dims = len(min_values) or int(ps.get("dimensions") or default_dims or req.get("dimensions", 10) or 10)
                min_values = _bound_vector(ps.get("min_values", min_values), dims, min_values[0] if min_values else -100.0)
                max_values = _bound_vector(ps.get("max_values", max_values), dims, max_values[0] if max_values else 100.0)
                specs.append({
                    "id": fn_id,
                    "label": ps.get("label") or eng.get("name", fn_id),
                    "fn": fn,
                    "min": min_values[0],
                    "max": max_values[0],
                    "min_values": min_values,
                    "max_values": max_values,
                    "dims": dims,
                    "optimum": ps.get("optimum", eng.get("best_known_fitness")),
                    "constraints": eng.get("constraints") or [],
                    "constraint_handler": ps.get("constraint_handler") or req.get("constraint_handler") or "deb",
                    "penalty_coefficient": float(ps.get("penalty_coefficient", req.get("penalty_coeff", 1.0e6)) or 1.0e6),
                    "is_engineering": True,
                })
                continue
            fn = _resolve_fn(fn_id)
            meta = _KNOWN.get(fn_id, {})
            dims = int(ps.get("dimensions") or meta.get("fixed_dims") or default_dims or req.get("dimensions", 10) or 10)
            min_values = _bound_vector(ps.get("min_values", ps.get("min")), dims, meta.get("min", (req.get("min_values") or [-100.0])[0]))
            max_values = _bound_vector(ps.get("max_values", ps.get("max")), dims, meta.get("max", (req.get("max_values") or [100.0])[0]))
            specs.append({
                "id": fn_id,
                "label": ps.get("label") or meta.get("label", fn_id),
                "fn": fn,
                "min": min_values[0],
                "max": max_values[0],
                "min_values": min_values,
                "max_values": max_values,
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
        if _is_engineering_function(fn_id):
            eng = _engineering_benchmark_info(fn_id) or {}
            min_values = [float(v) for v in eng.get("min_values", [])]
            max_values = [float(v) for v in eng.get("max_values", [])]
            dims = len(min_values) or int(default_dims or req.get("dimensions", 10) or 10)
            specs.append({
                "id": fn_id,
                "label": eng.get("name", fn_id),
                "fn": eng.get("objective") or _resolve_fn(fn_id),
                "min": min_values[0] if min_values else -100.0,
                "max": max_values[0] if max_values else 100.0,
                "min_values": min_values,
                "max_values": max_values,
                "dims": dims,
                "optimum": eng.get("best_known_fitness"),
                "constraints": eng.get("constraints") or [],
                "constraint_handler": req.get("constraint_handler") or "deb",
                "penalty_coefficient": float(req.get("penalty_coeff", 1.0e6) or 1.0e6),
                "is_engineering": True,
            })
            continue
        fn = _resolve_fn(fn_id)
        meta = _KNOWN.get(fn_id, {})
        dims = int(meta.get("fixed_dims") or default_dims or req.get("dimensions", 10) or len(req.get("min_values") or []) or 10)
        min_values = _bound_vector(None, dims, meta.get("min", (req.get("min_values") or [-100.0])[0]))
        max_values = _bound_vector(None, dims, meta.get("max", (req.get("max_values") or [100.0])[0]))
        specs.append({
            "id": fn_id,
            "label": meta.get("label", fn_id),
            "fn": fn,
            "min": min_values[0],
            "max": max_values[0],
            "min_values": min_values,
            "max_values": max_values,
            "dims": dims,
        })

    for idx, cf in enumerate(req.get("custom_functions") or [], start=1):
        cid = str(cf.get("id") or f"custom_{idx}")
        dims = int(cf.get("dimensions") or default_dims or req.get("dimensions", 10) or len(req.get("min_values") or []) or 10)
        specs.append({
            "id": cid,
            "label": cf.get("label") or cid,
            "fn": _compile_custom_function(cf.get("code", "")),
            "min": _bound_vector(cf.get("min_values", cf.get("min")), dims, (req.get("min_values") or [-5.12])[0])[0],
            "max": _bound_vector(cf.get("max_values", cf.get("max")), dims, (req.get("max_values") or [5.12])[0])[0],
            "min_values": _bound_vector(cf.get("min_values", cf.get("min")), dims, (req.get("min_values") or [-5.12])[0]),
            "max_values": _bound_vector(cf.get("max_values", cf.get("max")), dims, (req.get("max_values") or [5.12])[0]),
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
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _operator_execution_counts_from_result(result) -> dict[str, float]:
    """Return engine-side operator execution counts for the web EvoMapX panel.

    EvoMapX activity measures contribution-active steps.  Several native
    operators, such as CMA-ES covariance and step-size updates, can execute
    every generation while intentionally receiving zero direct Δf attribution.
    Engines that expose ``operator_counts_total`` let the UI show this execution
    activity without changing EvoMapX's contribution semantics.
    """
    history = getattr(result, "history", None) or []
    if not isinstance(history, list):
        return {}

    # Prefer engine-reported totals from the most recent observation.
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        totals = row.get("operator_counts_total")
        if isinstance(totals, dict) and totals:
            return {str(k): float(v or 0.0) for k, v in totals.items()}

    # Fallback for engines that only expose per-step operator_counts.
    counts: dict[str, float] = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        step_counts = row.get("operator_counts")
        if not isinstance(step_counts, dict):
            continue
        for key, value in step_counts.items():
            counts[str(key)] = counts.get(str(key), 0.0) + float(value or 0.0)
    return counts


def _evomapx_payload(result, level: str = "auto") -> dict:
    """Small JSON-safe EvoMapX summary for the web UI."""
    try:
        report = evomapx_analysis(result, level=level)
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        labels = list(data.get("labels", []) or [])
        activity = data.get("activity", {}) or {}
        cds_raw = data.get("cds_raw", {}) or {}
        execution_counts = _operator_execution_counts_from_result(result)
        # Web-facing scores are computed from positive contribution first.
        # If the run has no positive mass, fall back to absolute contribution
        # activity.  If even that is zero, use engine execution counts and label
        # the panel accordingly instead of pretending they are convergence
        # contributions.
        label_order = []
        for source in (labels, list(cds_raw.keys()), list(activity.keys()), list(execution_counts.keys())):
            for label in source or []:
                label = str(label)
                if label not in label_order:
                    label_order.append(label)
        driver_rows = []
        for label in label_order:
            a = activity.get(label, {}) or {}
            signed = float(a.get("total_contribution", cds_raw.get(label, 0.0)) or 0.0)
            positive = float(a.get("total_positive_contribution", max(0.0, signed)) or 0.0)
            negative = float(a.get("total_negative_contribution", min(0.0, signed)) or 0.0)
            absolute = float(abs(positive) + abs(negative))
            active_steps = float(a.get("active_steps", 0.0) or 0.0)
            execution_count = float(execution_counts.get(label, active_steps) or 0.0)
            driver_rows.append({
                "label": str(label),
                "positive": positive,
                "negative": negative,
                "signed": signed,
                "absolute": absolute,
                "active_steps": active_steps,
                "execution_count": execution_count,
            })
        total_positive = sum(max(0.0, r["positive"]) for r in driver_rows)
        total_absolute = sum(max(0.0, r["absolute"]) for r in driver_rows)
        total_execution = sum(max(0.0, r["execution_count"]) for r in driver_rows)
        if total_positive > 1.0e-12:
            driver_mode = "positive"
            for r in driver_rows:
                r["score"] = max(0.0, r["positive"]) / total_positive
        elif total_absolute > 1.0e-12:
            driver_mode = "activity"
            for r in driver_rows:
                r["score"] = max(0.0, r["absolute"]) / total_absolute
        elif total_execution > 1.0e-12:
            driver_mode = "execution"
            for r in driver_rows:
                r["score"] = max(0.0, r["execution_count"]) / total_execution
        else:
            driver_mode = "none"
            for r in driver_rows:
                r["score"] = 0.0
        driver_rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return _json_safe({
            "objective": data.get("objective"),
            "level": data.get("level"),
            "support_note": "Operator-level EvoMapX is available for all registered algorithms; cooperative runs also expose island and migration attribution. Bars show contribution share when available; the execution count is engine-side operator activity.",
            "labels": labels,
            "steps": data.get("steps", []),
            "cds_normalized": data.get("cds_normalized", {}),
            "cds_raw": cds_raw,
            "driver_scores": driver_rows,
            "driver_mode": driver_mode,
            "operator_execution_counts": execution_counts,
            "activity": activity,
            "summary": data.get("summary", {}),
            "migration_attribution": data.get("migration_attribution", {}),
            "normalized_attribution": data.get("normalized_attribution", {}),
            "text": explain_evomapx(report),
        })
    except Exception as exc:
        return {"error": str(exc), "level": level}


# ── ① SINGLE ─────────────────────────────────────────────────────────────────
def _run_single(req: dict, state: dict) -> None:
    try:
        sub_mode = req.get("sub_mode", "standard")
        eng_info = _engineering_benchmark_info(req.get("target_function"))
        fn       = (eng_info.get("objective") if eng_info else None) or _resolve_fn(req["target_function"], req.get("custom_code", ""))
        auto_constraints = list(eng_info.get("constraints") or []) if eng_info else []
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

        if eng_info and sub_mode != "constrained":
            state.update({"status": "error",
                          "error": "Engineering design benchmarks are constrained problems. Select constrained mode and a constraint handler.",
                          "elapsed": time.time() - state["start_time"]})
            return

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
                constraints = auto_constraints + _parse_constraints(constraints_text)
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

        # Phase 10: all registered algorithms expose operator-level EvoMapX
        # telemetry through native-engine or native-family hooks.
        evomapx_payload = _evomapx_payload(res, level="operator")

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
            "evomapx":              evomapx_payload,
            "result":               {"best_fitness": float(res.best_fitness),
                                     "best_position": [float(v) for v in res.best_position],
                                     "metadata": metadata,
                                     "evomapx": evomapx_payload},
        })

    except Exception:
        state.update({"status": "error",
                      "error": traceback.format_exc(limit=10),
                      "elapsed": time.time() - state["start_time"]})


# ── ② COLLABORATIVE ───────────────────────────────────────────────────────────
def _island_manifest(req: dict) -> list[dict]:
    """Return the user-facing island configuration, including semantic roles.

    The legacy cooperative/orchestrated runners currently consume algorithm,
    label, config and seed.  Role/exchange_role are preserved here for the web
    report, diagnostics, and future role-aware controllers.
    """
    manifest: list[dict] = []
    for idx, isl in enumerate(req.get("islands", []), start=1):
        cfg = dict(isl.get("config", {}) or {})
        manifest.append({
            "index": idx,
            "algorithm": isl.get("algorithm", ""),
            "label": isl.get("label") or isl.get("algorithm", f"island_{idx}"),
            "role": isl.get("role") or "auto",
            "exchange_role": isl.get("exchange_role") or "both",
            "seed": isl.get("seed"),
            "population_size": cfg.get("population_size") or cfg.get("swarm_size") or cfg.get("pack_size"),
        })
    return manifest


def _build_islands(req: dict) -> list[dict]:
    islands = []
    for isl in req["islands"]:
        item = {
            "algorithm": isl["algorithm"],
            "label":     isl.get("label") or isl["algorithm"],
            "config":    isl.get("config", {}),
        }
        if isl.get("seed") is not None:
            item["seed"] = isl.get("seed")
        islands.append(item)
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
                    min_values         = _spec_bounds(spec)[0],
                    max_values         = _spec_bounds(spec)[1],
                    objective          = req.get("objective", "min"),
                    max_steps          = req.get("max_steps", 20),
                    migration_interval = req.get("migration_interval", 2),
                    migration_size     = req.get("migration_size", 4),
                    topology           = req.get("topology", "star"),
                    seed               = (int(base_seed) + i if base_seed is not None else None),
                    verbose            = False,
                    **_constraint_kwargs_from_spec(spec),
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
                    min_values      = _spec_bounds(spec)[0],
                    max_values      = _spec_bounds(spec)[1],
                    objective       = req.get("objective", "min"),
                    max_steps       = req.get("max_steps", 20),
                    seed            = (int(base_seed) + i if base_seed is not None else None),
                    config          = config,
                    verbose         = False,
                    **_constraint_kwargs_from_spec(spec),
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
                           "source_fitness": e.get("source_fitness"),
                           "fit_before": e.get("target_fitness_before"),
                           "fit_after": e.get("target_fitness_after", e.get("best_fitness_after"))})
        else:
            events.append({"step":      getattr(e,"global_step",0),
                           "from":      getattr(e,"source_label",""),
                           "to":        getattr(e,"target_label",""),
                           "migrants":  getattr(e,"migrants",0),
                           "source_fitness": getattr(e,"source_fitness",None),
                           "fit_before": getattr(e,"target_fitness_before",None),
                           "fit_after": getattr(e,"target_fitness_after",getattr(e,"best_fitness_after",None))})

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

    island_manifest = _island_manifest(req)
    evomapx_payload = _evomapx_payload(result, level="island")
    evomapx_operator_payload = _evomapx_payload(result, level="operator")

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
        "island_manifest":    island_manifest,
        "decisions":          decisions,
        "outcomes":           outcomes,
        "evomapx":            evomapx_payload,
        "evomapx_operator":   evomapx_operator_payload,
        "elapsed":            time.time() - state["start_time"],
        "result": {
            "best_fitness":   float(result.best_fitness),
            "best_position":  [float(v) for v in result.best_position],
            "island_summary": island_summary,
            "island_manifest": island_manifest,
            "migration_count": len(events),
            "n_islands":      len(req.get("islands", [])),
            "topology":       req.get("topology", "star"),
            "evomapx":        evomapx_payload,
            "evomapx_operator": evomapx_operator_payload,
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
            bmin, bmax = _spec_bounds(spec)
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
                            **_constraint_kwargs_from_spec(spec),
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



# ── ⑤ BENCHMARK STUDY ────────────────────────────────────────────────────────
def _run_benchmark_study(req: dict, state: dict) -> None:
    try:
        algorithms = req["algorithms"]
        n_trials = int(req.get("n_trials", 5))
        max_steps = req.get("max_steps")
        max_evaluations = req.get("max_evaluations")
        objective = req.get("objective", "min")
        base_seed = req.get("seed")
        alg_configs = req.get("algorithm_configs", {}) or {}
        fn_specs = _function_specs(req, default_dims=int(req.get("dimensions", 10) or 10))
        if not algorithms:
            raise ValueError("BenchmarkStudy requires at least one algorithm candidate.")
        if not fn_specs:
            raise ValueError("BenchmarkStudy requires at least one benchmark problem.")

        candidates = [
            {"name": alg_id, "type": "algorithm", "algorithm": alg_id, "config": dict(alg_configs.get(alg_id, {}) or {})}
            for alg_id in algorithms
        ]
        problems = [_benchmark_problem_from_spec(spec, objective) for spec in fn_specs]
        state["status_text"] = f"BenchmarkStudy: {len(candidates)} candidates × {len(problems)} problems × {n_trials} trials"
        state["step"] = 0

        study = BenchmarkStudy(
            candidates=candidates,
            problems=problems,
            n_trials=n_trials,
            max_steps=int(max_steps) if max_steps not in (None, "") else None,
            max_evaluations=int(max_evaluations) if max_evaluations not in (None, "") else None,
            seed=int(base_seed) if base_seed not in (None, "") else None,
            objective=objective,
            target_tolerance=float(req.get("target_tolerance", 1.0e-8) or 1.0e-8),
            store_convergence=True,
            verbose=False,
        )
        result = study.run()
        df = result.to_dataframe()
        rows = _df_records(df)
        summary = _df_records(result.summary())
        rank_table = _df_records(result.rank_table())
        friedman = _json_safe(result.friedman_test())
        wilcoxon = _json_safe(result.wilcoxon_pairwise())
        best_rank = None
        try:
            if rank_table:
                best_rank = sorted(rank_table, key=lambda r: (r.get("mean_rank") is None, r.get("mean_rank") or 1.0e99))[0]
        except Exception:
            best_rank = None
        sci = {
            "n_records": len(rows),
            "n_candidates": len({r.get("candidate") for r in rows if r.get("candidate") is not None}),
            "n_problems": len({r.get("problem") for r in rows if r.get("problem") is not None}),
            "best_mean_rank_candidate": best_rank.get("candidate") if isinstance(best_rank, dict) else None,
            "best_mean_rank": best_rank.get("mean_rank") if isinstance(best_rank, dict) else None,
        }
        conv = _df_records(result.convergence_dataframe())

        state.update({
            "status": "done",
            "elapsed": time.time() - state["start_time"],
            "step": len(rows),
            "partial_rows": rows,
            "status_text": "BenchmarkStudy complete",
            "result": {
                "rows": rows,
                "summary": summary,
                "rank_table": rank_table,
                "friedman": friedman,
                "wilcoxon": wilcoxon,
                "scientific_summary": sci,
                "convergence": conv,
                "algorithms": algorithms,
                "functions": [spec["id"] for spec in fn_specs],
                "n_trials": n_trials,
                "dimensions": int(req.get("dimensions", 10) or 10),
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
    seed:      int | None     = None
    role:      str            = "auto"
    exchange_role: str        = "both"


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


class BenchmarkStudyReq(BaseModel):
    algorithms: list[str]
    functions:  list[str] = Field(default_factory=list)
    custom_functions: list[dict[str, Any]] = Field(default_factory=list)
    problem_specs: list[dict[str, Any]] = Field(default_factory=list)
    algorithm_configs: dict[str, Any] = Field(default_factory=dict)
    n_trials:   int   = 5
    max_steps:  int | None = 300
    max_evaluations: int | None = None
    target_tolerance: float = 1.0e-8
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


@app.post("/api/jobs/benchmark-study", status_code=202)
def create_benchmark_study(req: BenchmarkStudyReq):
    return _start("benchmark_study", _run_benchmark_study, req.model_dump())


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
        "evomapx":            s.get("evomapx", {}),
        "evomapx_operator":   s.get("evomapx_operator", {}),
        "decisions":          s.get("decisions", []),
        "outcomes":           s.get("outcomes", []),
        # benchmark extras
        "status_text":        s.get("status_text", ""),
        "partial_rows":       s.get("partial_rows", []),
        "request":            s.get("request"),
        # Return the final result payload once the job is complete.  The
        # frontend finalization step uses this object to populate the
        # BenchmarkStudy/BenchmarkRunner tables.  Without it, completed
        # benchmark-study jobs could show "Done" with zero records even though
        # the backend had already stored the study result in state["result"].
        "result":             s.get("result") if s.get("status") == "done" else None,
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
