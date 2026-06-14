"""Registry and EvoMapX audit utilities for pyMetaheuristic.

The audit layer is intentionally read-only.  It checks that the different
representations of an optimizer agree with each other:

* engine classes and ``REGISTRY``;
* table-derived metadata in ``engines/__init__.py``;
* capability flags and capability sets;
* DOI/name/family metadata;
* EvoMapX profile and operator catalog metadata;
* optional runtime smoke checks on a small Sphere benchmark.

Typical usage
-------------

Static audit of the complete registry::

    python -m pymetaheuristic.src.audit

Audit one optimizer and run a smoke test::

    python -m pymetaheuristic.src.audit --algorithm yo --runtime

Programmatic usage::

    from pymetaheuristic.src.audit import audit_registry

    report = audit_registry("yo", runtime=True)
    print(report.to_text())

This module does not modify optimizer state, package metadata, registry tables,
or EvoMapX catalogs.  Runtime checks instantiate optimizers only when explicitly
requested with ``runtime=True`` or ``--runtime``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


_VALID_ALGORITHM_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_FIDELITIES = {"native", "profiled", "family", "macro", "native-family", "family-native"}
_HARD_SEVERITIES = {"FAIL", "ERROR"}


@dataclass(frozen=True)
class AuditIssue:
    """One audit finding."""

    severity: str
    check: str
    message: str
    algorithm_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.upper()
        return data


@dataclass
class AuditReport:
    """Container returned by :func:`audit_registry`."""

    issues: list[AuditIssue] = field(default_factory=list)
    selected_algorithms: list[str] = field(default_factory=list)
    runtime_enabled: bool = False
    strict: bool = False

    def add(
        self,
        severity: str,
        check: str,
        message: str,
        algorithm_id: str | None = None,
        **details: Any,
    ) -> None:
        self.issues.append(
            AuditIssue(
                severity=str(severity).upper(),
                check=str(check),
                message=str(message),
                algorithm_id=None if algorithm_id is None else str(algorithm_id),
                details={k: _json_safe(v) for k, v in details.items() if v is not None},
            )
        )

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(issue.severity.upper() for issue in self.issues)
        return {key: int(counts.get(key, 0)) for key in ("PASS", "WARN", "FAIL", "ERROR")}

    @property
    def ok(self) -> bool:
        counts = self.counts
        if counts["FAIL"] or counts["ERROR"]:
            return False
        if self.strict and counts["WARN"]:
            return False
        return True

    def failures(self) -> list[AuditIssue]:
        return [issue for issue in self.issues if issue.severity.upper() in _HARD_SEVERITIES]

    def warnings(self) -> list[AuditIssue]:
        return [issue for issue in self.issues if issue.severity.upper() == "WARN"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "strict": self.strict,
            "runtime_enabled": self.runtime_enabled,
            "selected_algorithms": list(self.selected_algorithms),
            "counts": self.counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_markdown(self, path: str | Path | None = None, *, include_passed: bool = False) -> str:
        rows = []
        for issue in self.issues:
            if issue.severity == "PASS" and not include_passed:
                continue
            alg = issue.algorithm_id or "—"
            details = ""
            if issue.details:
                details = "<br>".join(f"`{k}`: `{v}`" for k, v in issue.details.items())
            rows.append(
                f"| {issue.severity} | `{alg}` | `{issue.check}` | {issue.message} | {details} |"
            )
        counts = self.counts
        text = "\n".join(
            [
                "# pyMetaheuristic Registry Audit",
                "",
                f"- Status: {'PASS' if self.ok else 'FAIL'}",
                f"- Algorithms audited: {len(self.selected_algorithms)}",
                f"- Runtime checks: {'enabled' if self.runtime_enabled else 'disabled'}",
                f"- Strict mode: {'enabled' if self.strict else 'disabled'}",
                f"- Counts: PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}, ERROR={counts['ERROR']}",
                "",
                "| Severity | Algorithm | Check | Message | Details |",
                "| --- | --- | --- | --- | --- |",
                *(rows or ["| PASS | — | summary | No warnings or failures. |  |"]),
            ]
        )
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_text(self, *, include_passed: bool = False, max_items: int | None = None) -> str:
        counts = self.counts
        lines = [
            "pyMetaheuristic Registry Audit",
            "──────────────────────────────",
            f"Status: {'PASS' if self.ok else 'FAIL'}",
            f"Algorithms audited: {len(self.selected_algorithms)}",
            f"Runtime checks: {'enabled' if self.runtime_enabled else 'disabled'}",
            f"Strict mode: {'enabled' if self.strict else 'disabled'}",
            f"Counts: PASS={counts['PASS']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}  ERROR={counts['ERROR']}",
        ]
        visible = [
            issue for issue in self.issues
            if include_passed or issue.severity.upper() != "PASS"
        ]
        if max_items is not None:
            visible = visible[: int(max_items)]
        if visible:
            lines.append("")
            for issue in visible:
                prefix = f"[{issue.severity}]"
                alg = f" {issue.algorithm_id}" if issue.algorithm_id else ""
                lines.append(f"{prefix}{alg} :: {issue.check}: {issue.message}")
                if issue.details:
                    for key, value in issue.details.items():
                        lines.append(f"       {key}: {value}")
        else:
            lines.append("")
            lines.append("No warnings or failures.")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError(self.to_text(include_passed=False))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_registry(
    algorithms: str | Sequence[str] | None = None,
    *,
    runtime: bool = False,
    evomapx: bool = True,
    strict: bool = False,
    seed: int = 42,
    dimension: int = 3,
    max_steps: int = 5,
    max_evaluations: int = 250,
    store_population_snapshots: bool = False,
) -> AuditReport:
    """Audit all registered engines or a selected subset.

    Parameters
    ----------
    algorithms:
        ``None`` audits all registered algorithms.  A string or sequence audits
        only those IDs or aliases.
    runtime:
        If ``True``, run a short Sphere optimization smoke test for each selected
        algorithm.  This is deliberately opt-in because auditing every optimizer
        can take time.
    evomapx:
        If ``True`` and ``runtime=True``, run an EvoMapX passive comparison by
        executing the same smoke problem with EvoMapX disabled and enabled.
    strict:
        If ``True``, warnings make ``report.ok`` false.
    seed, dimension, max_steps, max_evaluations:
        Runtime smoke-test configuration.
    store_population_snapshots:
        Request population snapshots during runtime smoke tests.  This is off by
        default to keep the audit fast.
    """

    report = AuditReport(runtime_enabled=bool(runtime), strict=bool(strict))

    try:
        from . import engines
    except Exception as exc:  # pragma: no cover - catastrophic import failure
        report.add(
            "ERROR",
            "import.engines",
            "Could not import pymetaheuristic.src.engines.",
            exception=repr(exc),
            traceback=traceback.format_exc(limit=10),
        )
        return report

    selected = _select_algorithms(report, engines, algorithms)
    report.selected_algorithms = selected

    _audit_global_registry(report, engines)
    for algorithm_id in selected:
        _audit_engine_metadata(report, engines, algorithm_id)
        if evomapx:
            _audit_evomapx_static(report, engines, algorithm_id)

    if runtime:
        for algorithm_id in selected:
            _audit_runtime_smoke(
                report,
                algorithm_id=algorithm_id,
                seed=int(seed),
                dimension=int(dimension),
                max_steps=int(max_steps),
                max_evaluations=int(max_evaluations),
                evomapx=bool(evomapx),
                store_population_snapshots=bool(store_population_snapshots),
            )

    return report


def assert_registry_consistent(
    algorithms: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    """Run :func:`audit_registry` and raise ``AssertionError`` on failure."""

    report = audit_registry(algorithms, **kwargs)
    report.raise_if_failed()


# ---------------------------------------------------------------------------
# Static registry checks
# ---------------------------------------------------------------------------


def _select_algorithms(report: AuditReport, engines: Any, algorithms: str | Sequence[str] | None) -> list[str]:
    registry = dict(getattr(engines, "REGISTRY", {}) or {})
    aliases = dict(getattr(engines, "_REGISTRY_ALIASES", {}) or {})

    if algorithms is None:
        return sorted(registry)

    if isinstance(algorithms, str):
        raw = [piece.strip() for piece in algorithms.split(",") if piece.strip()]
    else:
        raw = []
        for item in algorithms:
            raw.extend(str(item).split(","))
        raw = [piece.strip() for piece in raw if piece.strip()]

    selected: list[str] = []
    for requested in raw:
        key = requested.lower()
        canonical = aliases.get(key, key)
        if canonical not in registry:
            report.add(
                "FAIL",
                "selection.algorithm_exists",
                f"Requested algorithm '{requested}' is not registered.",
                requested=requested,
                resolved=canonical,
            )
            continue
        if canonical not in selected:
            selected.append(canonical)
    return selected


def _audit_global_registry(report: AuditReport, engines: Any) -> None:
    registry = dict(getattr(engines, "REGISTRY", {}) or {})
    engine_classes = tuple(getattr(engines, "_ENGINE_CLASSES", ()) or ())
    table_ids = set(getattr(engines, "_TABLE_ALGORITHM_IDS", set()) or set())
    aliases = dict(getattr(engines, "_REGISTRY_ALIASES", {}) or {})

    if not engine_classes:
        report.add("FAIL", "registry.engine_classes_present", "_ENGINE_CLASSES is empty or missing.")
    else:
        report.add("PASS", "registry.engine_classes_present", f"{len(engine_classes)} engine classes found.")

    ids = [str(getattr(cls, "algorithm_id", "")) for cls in engine_classes]
    duplicate_ids = sorted([aid for aid, count in Counter(ids).items() if aid and count > 1])
    if duplicate_ids:
        report.add("FAIL", "registry.unique_algorithm_ids", "Duplicate algorithm_id values found.", duplicates=duplicate_ids)
    else:
        report.add("PASS", "registry.unique_algorithm_ids", "Every engine class algorithm_id is unique.")

    duplicate_classes = [name for name, count in Counter(id(cls) for cls in engine_classes).items() if count > 1]
    if duplicate_classes:
        report.add("FAIL", "registry.unique_engine_classes", "One or more engine classes appear multiple times in _ENGINE_CLASSES.", duplicate_object_count=len(duplicate_classes))
    else:
        report.add("PASS", "registry.unique_engine_classes", "Every engine class object appears once in _ENGINE_CLASSES.")

    expected_registry = {str(getattr(cls, "algorithm_id", "")): cls for cls in engine_classes if getattr(cls, "algorithm_id", None)}
    missing_from_registry = sorted(set(expected_registry) - set(registry))
    extra_in_registry = sorted(set(registry) - set(expected_registry))
    if missing_from_registry or extra_in_registry:
        report.add(
            "FAIL",
            "registry.class_registry_match",
            "REGISTRY does not match _ENGINE_CLASSES.",
            missing_from_registry=missing_from_registry,
            extra_in_registry=extra_in_registry,
        )
    else:
        report.add("PASS", "registry.class_registry_match", "REGISTRY matches _ENGINE_CLASSES.")

    if table_ids:
        missing_engine_for_table = sorted(table_ids - set(registry))
        missing_table_for_engine = sorted(set(registry) - table_ids)
        if missing_engine_for_table or missing_table_for_engine:
            report.add(
                "FAIL",
                "registry.table_ids_match",
                "_TABLE_ALGORITHM_IDS and REGISTRY disagree.",
                table_without_engine=missing_engine_for_table,
                engine_without_table=missing_table_for_engine,
            )
        else:
            report.add("PASS", "registry.table_ids_match", "_TABLE_ALGORITHM_IDS matches REGISTRY.")
    else:
        report.add("WARN", "registry.table_ids_present", "_TABLE_ALGORITHM_IDS is missing or empty.")

    for alias, target in sorted(aliases.items()):
        if target not in registry:
            report.add("FAIL", "registry.alias_target_exists", "Alias points to an unknown algorithm ID.", alias=alias, target=target)
        elif alias in registry:
            report.add("WARN", "registry.alias_shadowing", "Alias shadows a canonical registry ID.", alias=alias, target=target)
        else:
            report.add("PASS", "registry.alias_target_exists", "Alias target exists.", algorithm_id=target, alias=alias)


def _audit_engine_metadata(report: AuditReport, engines: Any, algorithm_id: str) -> None:
    registry = dict(getattr(engines, "REGISTRY", {}) or {})
    cls = registry.get(algorithm_id)
    if cls is None:
        report.add("FAIL", "engine.registered", "Algorithm is not present in REGISTRY.", algorithm_id=algorithm_id)
        return

    table_id = _table_id_for(engines, algorithm_id)
    table_ids = set(getattr(engines, "_TABLE_ALGORITHM_IDS", set()) or set())
    population_based = set(getattr(engines, "_POPULATION_BASED", set()) or set())
    injection_enabled = set(getattr(engines, "_INJECTION_ENABLED", set()) or set())
    restart_enabled = set(getattr(engines, "_RESTART_ENABLED", set()) or set())
    snapshot_fit_enabled = set(getattr(engines, "_SNAPSHOT_FIT_ENABLED", set()) or set())
    algorithm_names = dict(getattr(engines, "_ALGORITHM_NAMES", {}) or {})
    algorithm_families = dict(getattr(engines, "_ALGORITHM_FAMILIES", {}) or {})
    algorithm_dois = dict(getattr(engines, "_ALGORITHM_DOIS", {}) or {})
    engine_classes = tuple(getattr(engines, "_ENGINE_CLASSES", ()) or ())

    cls_id = str(getattr(cls, "algorithm_id", ""))
    if cls_id == algorithm_id:
        report.add("PASS", "engine.algorithm_id_matches_registry", "Engine algorithm_id matches registry key.", algorithm_id)
    else:
        report.add(
            "FAIL",
            "engine.algorithm_id_matches_registry",
            "Engine algorithm_id does not match registry key.",
            algorithm_id=algorithm_id,
            class_algorithm_id=cls_id,
        )

    if _VALID_ALGORITHM_ID_RE.match(algorithm_id):
        report.add("PASS", "engine.algorithm_id_format", "Algorithm ID has a valid canonical format.", algorithm_id)
    else:
        report.add("WARN", "engine.algorithm_id_format", "Algorithm ID is not lowercase snake-case alphanumeric.", algorithm_id)

    count_in_classes = sum(1 for item in engine_classes if item is cls)
    if count_in_classes == 1:
        report.add("PASS", "engine.in_engine_classes_once", "Engine class appears exactly once in _ENGINE_CLASSES.", algorithm_id)
    else:
        report.add("FAIL", "engine.in_engine_classes_once", "Engine class does not appear exactly once in _ENGINE_CLASSES.", algorithm_id, count=count_in_classes)

    if table_id in table_ids:
        report.add("PASS", "engine.in_table_algorithm_ids", "Algorithm appears in _TABLE_ALGORITHM_IDS.", algorithm_id)
    else:
        report.add("FAIL", "engine.in_table_algorithm_ids", "Algorithm is missing from _TABLE_ALGORITHM_IDS.", algorithm_id, table_id=table_id)

    _check_metadata_value(report, algorithm_id, "name", algorithm_names, table_id, getattr(cls, "algorithm_name", None), required=True)
    _check_metadata_value(report, algorithm_id, "family", algorithm_families, table_id, getattr(cls, "family", None), required=True)

    reference = dict(getattr(cls, "_REFERENCE", {}) or {})
    class_doi = reference.get("doi")
    if table_id in algorithm_dois:
        if class_doi == algorithm_dois[table_id]:
            report.add("PASS", "metadata.doi", "DOI metadata matches _ALGORITHM_DOIS.", algorithm_id, doi=class_doi)
        else:
            report.add(
                "FAIL",
                "metadata.doi",
                "Engine DOI metadata does not match _ALGORITHM_DOIS.",
                algorithm_id,
                table_doi=algorithm_dois[table_id],
                engine_doi=class_doi,
            )
    else:
        report.add("WARN", "metadata.doi", "Algorithm has no DOI in _ALGORITHM_DOIS.", algorithm_id)

    caps = getattr(cls, "capabilities", None)
    if caps is None:
        report.add("FAIL", "capabilities.present", "Engine has no CapabilityProfile.", algorithm_id)
        return
    report.add("PASS", "capabilities.present", "Engine has a CapabilityProfile.", algorithm_id)

    # CapabilityProfile uses has_population; older generated registry code may
    # also try to set is_population_based.  The audit checks the truthful field.
    _check_capability_set(
        report,
        algorithm_id,
        "has_population",
        bool(getattr(caps, "has_population", False)),
        table_id in population_based,
        "_POPULATION_BASED",
    )
    _check_capability_set(
        report,
        algorithm_id,
        "supports_candidate_injection",
        bool(getattr(caps, "supports_candidate_injection", False)),
        table_id in injection_enabled,
        "_INJECTION_ENABLED",
    )
    _check_capability_set(
        report,
        algorithm_id,
        "supports_restart",
        bool(getattr(caps, "supports_restart", False)),
        table_id in restart_enabled,
        "_RESTART_ENABLED",
    )
    _check_capability_set(
        report,
        algorithm_id,
        "supports_snapshot_fit",
        bool(getattr(caps, "supports_snapshot_fit", False)),
        table_id in snapshot_fit_enabled,
        "_SNAPSHOT_FIT_ENABLED",
    )


def _check_metadata_value(
    report: AuditReport,
    algorithm_id: str,
    field_name: str,
    table: dict[str, Any],
    table_id: str,
    class_value: Any,
    *,
    required: bool,
) -> None:
    check = f"metadata.{field_name}"
    if table_id not in table:
        severity = "FAIL" if required else "WARN"
        report.add(severity, check, f"Algorithm is missing from _ALGORITHM_{field_name.upper()}S.", algorithm_id, table_id=table_id)
        return
    table_value = table[table_id]
    if class_value is None or str(class_value) == "":
        report.add("FAIL", check, f"Engine class has no algorithm_{field_name} value.", algorithm_id, table_value=table_value)
        return
    if str(class_value).strip().lower() == str(table_value).strip().lower():
        report.add("PASS", check, f"Engine {field_name} matches registry metadata.", algorithm_id, value=class_value)
    else:
        report.add(
            "FAIL",
            check,
            f"Engine {field_name} does not match registry metadata.",
            algorithm_id,
            engine_value=class_value,
            table_value=table_value,
        )


def _check_capability_set(
    report: AuditReport,
    algorithm_id: str,
    capability_name: str,
    engine_value: bool,
    registry_value: bool,
    registry_set_name: str,
) -> None:
    if engine_value == registry_value:
        report.add(
            "PASS",
            f"capabilities.{capability_name}",
            f"CapabilityProfile agrees with {registry_set_name}.",
            algorithm_id,
            value=engine_value,
        )
    else:
        report.add(
            "FAIL",
            f"capabilities.{capability_name}",
            f"CapabilityProfile disagrees with {registry_set_name}.",
            algorithm_id,
            engine_value=engine_value,
            registry_value=registry_value,
        )


# ---------------------------------------------------------------------------
# EvoMapX static checks
# ---------------------------------------------------------------------------


def _audit_evomapx_static(report: AuditReport, engines: Any, algorithm_id: str) -> None:
    cls = dict(getattr(engines, "REGISTRY", {}) or {}).get(algorithm_id)
    family = None if cls is None else getattr(cls, "family", None)

    try:
        from . import evomapx_profiles
        from . import evomapx_operator_catalog
    except Exception as exc:
        report.add("ERROR", "evomapx.import", "Could not import EvoMapX profile/catalog modules.", algorithm_id, exception=repr(exc))
        return

    profiles = dict(getattr(evomapx_profiles, "EVOMAPX_OPERATOR_PROFILES", {}) or {})
    if algorithm_id not in profiles:
        report.add("FAIL", "evomapx.profile_present", "Algorithm has no explicit EVOMAPX_OPERATOR_PROFILES entry.", algorithm_id)
        profile = evomapx_profiles.get_evomapx_profile(algorithm_id, family=family)
    else:
        report.add("PASS", "evomapx.profile_present", "Algorithm has an explicit EvoMapX profile.", algorithm_id)
        profile = profiles[algorithm_id]

    profile_id = str(getattr(profile, "algorithm_id", ""))
    if profile_id == algorithm_id:
        report.add("PASS", "evomapx.profile_algorithm_id", "EvoMapX profile algorithm_id matches registry ID.", algorithm_id)
    else:
        report.add("FAIL", "evomapx.profile_algorithm_id", "EvoMapX profile algorithm_id does not match registry ID.", algorithm_id, profile_algorithm_id=profile_id)

    profile_family = str(getattr(profile, "family", "") or "")
    if family is None or profile_family.lower() == str(family).lower():
        report.add("PASS", "evomapx.profile_family", "EvoMapX profile family matches engine family.", algorithm_id, family=profile_family)
    else:
        report.add("FAIL", "evomapx.profile_family", "EvoMapX profile family does not match engine family.", algorithm_id, profile_family=profile_family, engine_family=family)

    fidelity = str(getattr(profile, "fidelity", "") or "")
    if fidelity in _VALID_FIDELITIES:
        report.add("PASS", "evomapx.profile_fidelity", "EvoMapX fidelity value is recognized.", algorithm_id, fidelity=fidelity)
    else:
        report.add("WARN", "evomapx.profile_fidelity", "EvoMapX fidelity value is not in the recognized set.", algorithm_id, fidelity=fidelity)

    operators = tuple(getattr(profile, "operators", ()) or ())
    if operators and all(isinstance(op, str) and op.strip() for op in operators):
        report.add("PASS", "evomapx.profile_operators", "EvoMapX profile declares non-empty operators.", algorithm_id, operators=list(operators))
    else:
        report.add("FAIL", "evomapx.profile_operators", "EvoMapX profile operators are empty or invalid.", algorithm_id)

    try:
        operators_from_helper = tuple(evomapx_profiles.get_evomapx_operators(algorithm_id, family=family))
    except Exception as exc:
        report.add("ERROR", "evomapx.get_operators", "get_evomapx_operators raised an exception.", algorithm_id, exception=repr(exc))
        operators_from_helper = ()
    if tuple(operators_from_helper) == tuple(operators):
        report.add("PASS", "evomapx.get_operators", "get_evomapx_operators agrees with the profile operator tuple.", algorithm_id)
    else:
        report.add("WARN", "evomapx.get_operators", "get_evomapx_operators differs from the profile operator tuple.", algorithm_id, profile=list(operators), helper=list(operators_from_helper))

    try:
        labels = list(evomapx_operator_catalog.labels_for_algorithm(algorithm_id))
    except Exception as exc:
        report.add("ERROR", "evomapx.catalog_labels", "labels_for_algorithm raised an exception.", algorithm_id, exception=repr(exc))
        labels = []

    if labels:
        report.add("PASS", "evomapx.catalog_labels", "labels_for_algorithm returns non-empty labels.", algorithm_id, labels=labels)
    else:
        report.add("FAIL", "evomapx.catalog_labels", "labels_for_algorithm returns no labels.", algorithm_id)

    unqualified = [label for label in labels if "." not in str(label)]
    if unqualified:
        report.add("WARN", "evomapx.catalog_labels_qualified", "Some EvoMapX labels are not fully qualified with an algorithm prefix.", algorithm_id, labels=unqualified)
    else:
        report.add("PASS", "evomapx.catalog_labels_qualified", "All EvoMapX catalog labels are fully qualified.", algorithm_id)

    # Soft overlap diagnostic.  Profiled/family mappings are often conceptual
    # while catalog labels are concrete runtime labels, so overlap is enforced
    # only for native-style profiles where the implementation is expected to
    # expose algorithm-specific semantic operators.
    if operators and labels and fidelity == "native":
        suffixes = {str(label).split(".", 1)[-1] for label in labels}
        op_tokens = {str(op).replace(" ", "_").replace("/", "_").replace("-", "_").lower() for op in operators}
        suffix_tokens = {s.replace(" ", "_").replace("/", "_").replace("-", "_").lower() for s in suffixes}
        if op_tokens & suffix_tokens:
            report.add("PASS", "evomapx.profile_catalog_overlap", "Native profile operators overlap with catalog label suffixes.", algorithm_id)
        else:
            report.add("WARN", "evomapx.profile_catalog_overlap", "Native profile operators and catalog labels have no exact suffix overlap.", algorithm_id)


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------


def _audit_runtime_smoke(
    report: AuditReport,
    *,
    algorithm_id: str,
    seed: int,
    dimension: int,
    max_steps: int,
    max_evaluations: int,
    evomapx: bool,
    store_population_snapshots: bool,
) -> None:
    try:
        import numpy as np
        from .api import optimize
    except Exception as exc:
        report.add("ERROR", "runtime.import", "Could not import runtime dependencies.", algorithm_id, exception=repr(exc))
        return

    lo = [-5.0] * int(dimension)
    hi = [5.0] * int(dimension)

    def sphere(x: Sequence[float]) -> float:
        arr = np.asarray(x, dtype=float)
        return float(np.sum(arr * arr))

    def run_once(*, evomapx_enabled: bool, run_seed: int):
        return optimize(
            algorithm_id,
            sphere,
            lo,
            hi,
            objective="min",
            max_steps=max_steps,
            max_evaluations=max_evaluations,
            seed=run_seed,
            store_history=True,
            store_population_snapshots=store_population_snapshots,
            evomapx=bool(evomapx_enabled),
        )

    try:
        result = run_once(evomapx_enabled=evomapx, run_seed=seed)
    except Exception as exc:
        report.add(
            "FAIL",
            "runtime.sphere_runs",
            "Algorithm failed on the Sphere smoke test.",
            algorithm_id,
            exception=repr(exc),
            traceback=traceback.format_exc(limit=12),
        )
        return

    report.add("PASS", "runtime.sphere_runs", "Algorithm ran on the Sphere smoke test.", algorithm_id)

    if _is_finite_number(getattr(result, "best_fitness", None)):
        report.add("PASS", "runtime.best_fitness_finite", "Best fitness is finite.", algorithm_id, best_fitness=float(result.best_fitness))
    else:
        report.add("FAIL", "runtime.best_fitness_finite", "Best fitness is not finite.", algorithm_id, best_fitness=getattr(result, "best_fitness", None))

    best_position = getattr(result, "best_position", None)
    if _position_respects_bounds(best_position, lo, hi):
        report.add("PASS", "runtime.bounds", "Best position respects bounds.", algorithm_id)
    else:
        report.add("FAIL", "runtime.bounds", "Best position violates bounds.", algorithm_id, best_position=best_position, min_values=lo, max_values=hi)

    if getattr(result, "algorithm_id", None) == algorithm_id:
        report.add("PASS", "runtime.result_algorithm_id", "Result algorithm_id matches requested algorithm.", algorithm_id)
    else:
        report.add("FAIL", "runtime.result_algorithm_id", "Result algorithm_id does not match requested algorithm.", algorithm_id, result_algorithm_id=getattr(result, "algorithm_id", None))

    if isinstance(getattr(result, "history", None), list):
        report.add("PASS", "runtime.history_format", "Result history is a list.", algorithm_id, history_length=len(result.history))
    else:
        report.add("FAIL", "runtime.history_format", "Result history is not a list.", algorithm_id)

    try:
        repeat = run_once(evomapx_enabled=evomapx, run_seed=seed)
        if _same_result_signature(result, repeat):
            report.add("PASS", "runtime.reproducibility", "Fixed-seed smoke test is reproducible.", algorithm_id)
        else:
            report.add(
                "FAIL",
                "runtime.reproducibility",
                "Fixed-seed smoke test is not reproducible.",
                algorithm_id,
                first=_result_signature(result),
                second=_result_signature(repeat),
            )
    except Exception as exc:
        report.add("FAIL", "runtime.reproducibility", "Repeat fixed-seed run failed.", algorithm_id, exception=repr(exc))

    if evomapx:
        _audit_runtime_evomapx(report, run_once, result, algorithm_id, seed)


def _audit_runtime_evomapx(report: AuditReport, run_once: Any, result: Any, algorithm_id: str, seed: int) -> None:
    try:
        evomapx_report = result.evomapx_analysis(level="operator")
        labels = list(getattr(evomapx_report, "labels", []) or [])
        if labels:
            report.add("PASS", "runtime.evomapx_operator_analysis", "evomapx_analysis(level='operator') returned labels.", algorithm_id, labels=labels)
        else:
            report.add("FAIL", "runtime.evomapx_operator_analysis", "evomapx_analysis(level='operator') returned no labels.", algorithm_id)
    except Exception as exc:
        report.add("FAIL", "runtime.evomapx_operator_analysis", "evomapx_analysis(level='operator') failed.", algorithm_id, exception=repr(exc))

    contributions = _operator_contributions_from_history(getattr(result, "history", []) or [])
    if contributions:
        report.add("PASS", "runtime.operator_contributions", "History contains non-empty operator_contributions.", algorithm_id, operators=sorted(contributions))
    else:
        # Not all older/profiled engines emit native operator_contributions; the
        # passive probe can still build EvoMapX from population/evaluation traces.
        report.add("WARN", "runtime.operator_contributions", "History contains no native operator_contributions.", algorithm_id)

    try:
        without = run_once(evomapx_enabled=False, run_seed=seed)
        with_enabled = run_once(evomapx_enabled=True, run_seed=seed)
        if _same_result_signature(without, with_enabled):
            report.add("PASS", "runtime.evomapx_passive", "EvoMapX enabled/disabled fixed-seed runs are identical for final result, history signature, and evaluation count.", algorithm_id)
        else:
            report.add(
                "FAIL",
                "runtime.evomapx_passive",
                "EvoMapX changed the fixed-seed result, history signature, or evaluation count.",
                algorithm_id,
                without_evomapx=_result_signature(without),
                with_evomapx=_result_signature(with_enabled),
            )
    except Exception as exc:
        report.add("FAIL", "runtime.evomapx_passive", "EvoMapX passive comparison failed to run.", algorithm_id, exception=repr(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_id_for(engines: Any, algorithm_id: str) -> str:
    aliases = dict(getattr(engines, "_REGISTRY_ALIASES", {}) or {})
    return str(aliases.get(algorithm_id, algorithm_id))


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _position_respects_bounds(position: Any, lo: Sequence[float], hi: Sequence[float], tol: float = 1e-8) -> bool:
    try:
        import numpy as np
        arr = np.asarray(position, dtype=float)
        lower = np.asarray(lo, dtype=float)
        upper = np.asarray(hi, dtype=float)
        return arr.shape == lower.shape and bool(np.all(arr >= lower - tol) and np.all(arr <= upper + tol))
    except Exception:
        return False


def _result_signature(result: Any) -> dict[str, Any]:
    history = getattr(result, "history", []) or []
    hist_sig = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if "best_fitness" not in row:
            continue
        hist_sig.append(
            {
                "step": _safe_int(row.get("step")),
                "evaluations": _safe_int(row.get("evaluations")),
                "best_fitness": _safe_float(row.get("best_fitness")),
            }
        )
    return {
        "algorithm_id": getattr(result, "algorithm_id", None),
        "best_fitness": _safe_float(getattr(result, "best_fitness", None)),
        "best_position": [_safe_float(v) for v in (getattr(result, "best_position", []) or [])],
        "steps": _safe_int(getattr(result, "steps", None)),
        "evaluations": _safe_int(getattr(result, "evaluations", None)),
        "history": hist_sig,
    }


def _same_result_signature(left: Any, right: Any, *, rtol: float = 1e-10, atol: float = 1e-10) -> bool:
    try:
        import numpy as np
        lsig = _result_signature(left)
        rsig = _result_signature(right)
        if lsig["algorithm_id"] != rsig["algorithm_id"]:
            return False
        if lsig["steps"] != rsig["steps"] or lsig["evaluations"] != rsig["evaluations"]:
            return False
        if not math.isclose(float(lsig["best_fitness"]), float(rsig["best_fitness"]), rel_tol=rtol, abs_tol=atol):
            return False
        if not np.allclose(np.asarray(lsig["best_position"], dtype=float), np.asarray(rsig["best_position"], dtype=float), rtol=rtol, atol=atol):
            return False
        if len(lsig["history"]) != len(rsig["history"]):
            return False
        for a, b in zip(lsig["history"], rsig["history"]):
            if a["step"] != b["step"] or a["evaluations"] != b["evaluations"]:
                return False
            if not math.isclose(float(a["best_fitness"]), float(b["best_fitness"]), rel_tol=rtol, abs_tol=atol):
                return False
        return True
    except Exception:
        return False


def _operator_contributions_from_history(history: Iterable[Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in history:
        if not isinstance(row, dict):
            continue
        contrib = row.get("operator_contributions")
        if not isinstance(contrib, dict):
            continue
        for key, value in contrib.items():
            try:
                val = float(value)
            except Exception:
                val = 0.0
            if math.isfinite(val):
                totals[str(key)] += val
    return dict(totals)


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            val = float(value)
            return val if math.isfinite(val) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit pyMetaheuristic engine registry, capabilities, and EvoMapX metadata.")
    parser.add_argument(
        "--algorithm",
        "-a",
        action="append",
        default=None,
        help="Algorithm ID or comma-separated IDs to audit. Repeatable. Omit to audit all registered engines.",
    )
    parser.add_argument("--runtime", action="store_true", help="Run short Sphere smoke tests for the selected algorithms.")
    parser.add_argument("--no-evomapx", action="store_true", help="Skip EvoMapX static checks and runtime passive checks.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as release-blocking.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by runtime smoke tests.")
    parser.add_argument("--dimension", type=int, default=3, help="Sphere benchmark dimension for runtime smoke tests.")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum engine steps for runtime smoke tests.")
    parser.add_argument("--max-evaluations", type=int, default=250, help="Maximum objective evaluations for runtime smoke tests.")
    parser.add_argument("--snapshots", action="store_true", help="Request population snapshots during runtime smoke tests.")
    parser.add_argument("--show-passed", action="store_true", help="Print passed checks as well as warnings/failures.")
    parser.add_argument("--json", dest="json_path", default=None, help="Write JSON audit report to this path.")
    parser.add_argument("--markdown", dest="markdown_path", default=None, help="Write Markdown audit report to this path.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of displayed text findings.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    algorithms = args.algorithm if args.algorithm else None
    report = audit_registry(
        algorithms=algorithms,
        runtime=bool(args.runtime),
        evomapx=not bool(args.no_evomapx),
        strict=bool(args.strict),
        seed=int(args.seed),
        dimension=int(args.dimension),
        max_steps=int(args.max_steps),
        max_evaluations=int(args.max_evaluations),
        store_population_snapshots=bool(args.snapshots),
    )
    if args.json_path:
        report.to_json(args.json_path)
    if args.markdown_path:
        report.to_markdown(args.markdown_path, include_passed=args.show_passed)
    print(report.to_text(include_passed=args.show_passed, max_items=args.max_items))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
