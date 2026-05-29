"""Passive EvoMapX probes for faithful pyMetaheuristic engines.

The probe layer is intentionally observational: it does not change candidate
positions, fitness values, random numbers, selection rules, or evaluation
budgets.  It only copies already-computed states and objective evaluations.

Levels
------
1. Population snapshots: deep-copy population/state snapshots.
2. Operator deltas: signed parent→child Δf grouped by the operator label
   assigned to the accepted population transition.
3. Lineage: passive parent -> child links inferred from nearest previous
   population member unless an engine provides exact ancestry.
4. Full activity: acceptance, displacement, diversity, candidate-delta, and
   operator activity metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import os
import sys
import numpy as np


_SKIP_FUNCTIONS = {
    "record_evaluation", "_source_label", "evaluate", "evaluate_details",
    "_evaluate_population", "_pop_from_positions", "_callback_payload",
    "_run_callbacks", "run", "wrapped_initialize", "hooked_uniform",
    "_greedy_update", "_greedy_single", "_greedy_apply", "_trial_fit",
    "_evaluate_trial", "_evaluate_candidate",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr.copy()


def _fitness_better_delta(before: float | None, after: float | None, objective: str) -> float:
    if before is None or after is None:
        return 0.0
    try:
        b, a = float(before), float(after)
    except Exception:
        return 0.0
    if not (math.isfinite(b) and math.isfinite(a)):
        return 0.0
    return (a - b) if str(objective).lower() == "max" else (b - a)





def _expanded_operator_parts(algorithm_id: str, operator: Any, value: float) -> list[tuple[str, float, float, int]]:
    """Return ``(label, value_share, share, split_size)`` tuples.

    The passive probe may observe a complete macro-transition whose label packs
    multiple operators.  Decomposition is done only on already-observed deltas;
    no objective calls or RNG draws are introduced.
    """
    op = str(operator or f"{algorithm_id}.step")
    try:
        from .evomapx_operator_catalog import expand_compound_operator_label
        labels = expand_compound_operator_label(str(algorithm_id or ""), op)
    except Exception:
        labels = [op]
    labels = [str(x) for x in labels if x not in {None, ""}]
    if not labels:
        labels = [op]
    split_size = max(1, len(labels))
    share = 1.0 / float(split_size)
    return [(label, float(value) * share, share, split_size) for label in labels]


def _best_value(fitness: np.ndarray | None, objective: str) -> float | None:
    if fitness is None or fitness.size == 0:
        return None
    fit = np.asarray(fitness, dtype=float)
    fit = fit[np.isfinite(fit)]
    if fit.size == 0:
        return None
    return float(np.max(fit) if str(objective).lower() == "max" else np.min(fit))


def _diversity(positions: np.ndarray | None, lo: np.ndarray, hi: np.ndarray) -> float | None:
    if positions is None:
        return None
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return None
    denom = float(np.linalg.norm(hi - lo)) or 1.0
    centroid = pos.mean(axis=0)
    return float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)


@dataclass
class _StepSnapshot:
    step: int
    evaluations: int
    best_fitness: float | None
    positions: np.ndarray | None
    fitness: np.ndarray | None
    ids: list[str] = field(default_factory=list)


class EvoMapXProbe:
    """A passive observer attached to one engine instance."""

    def __init__(self, engine: Any, level: int = 4, enabled: bool = True, keep_events: bool = False) -> None:
        self.engine = engine
        self.algorithm_id = str(getattr(engine, "algorithm_id", "algorithm") or "algorithm")
        self.objective = str(getattr(getattr(engine, "problem", None), "objective", "min") or "min")
        self.level = int(max(0, min(4, level)))
        self.enabled = bool(enabled and self.level > 0)
        self.keep_events = bool(keep_events)
        self.snapshots: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.raw_events: list[dict[str, Any]] = []
        self._active_step: int | None = None
        self._step_evals: list[dict[str, Any]] = []
        self._before: _StepSnapshot | None = None
        self._last_lineage: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}

    @classmethod
    def from_engine(cls, engine: Any) -> "EvoMapXProbe":
        params = dict(getattr(getattr(engine, "config", None), "params", {}) or {})
        # Default ON because the package's web UI and EvoMapX analysis expect
        # operator telemetry.  Users can disable with evomapx=False or level=0.
        enabled = bool(params.get("evomapx", params.get("enable_evomapx", True)))
        raw_level = params.get("evomapx_level", params.get("evomapx_probe_level", 4))
        try:
            level = int(raw_level)
        except Exception:
            level = 4
        if not enabled:
            level = 0
        return cls(
            engine=engine,
            level=level,
            enabled=enabled,
            keep_events=bool(params.get("evomapx_keep_events", False)),
        )

    def _problem_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        problem = getattr(self.engine, "problem", None)
        lo = np.asarray(getattr(problem, "min_values", []), dtype=float)
        hi = np.asarray(getattr(problem, "max_values", []), dtype=float)
        return lo, hi

    def _extract_population(self, state: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        payload = getattr(state, "payload", {}) or {}
        dim = int(getattr(getattr(self.engine, "problem", None), "dimension", 0) or 0)
        # Common population matrix convention: [x_1, ..., x_d, fitness]
        for key in ("population", "sources", "positions", "memories"):
            value = payload.get(key)
            arr = _safe_array(value)
            if arr is not None and arr.ndim == 2:
                if dim > 0 and arr.shape[1] >= dim + 1:
                    return arr[:, :dim].copy(), arr[:, -1].copy()
                if arr.shape[1] > 1:
                    return arr[:, :-1].copy(), arr[:, -1].copy()
        # Single-trajectory convention.
        cur = payload.get("current")
        cur_fit = payload.get("current_fit")
        if cur is not None and cur_fit is not None:
            pos = _safe_array(cur)
            if pos is not None:
                return pos.reshape(1, -1), np.asarray([float(cur_fit)], dtype=float)
        # Last resort: best-so-far.
        best = getattr(state, "best_position", None)
        best_fit = getattr(state, "best_fitness", None)
        if best is not None and best_fit is not None:
            pos = _safe_array(best)
            if pos is not None:
                return pos.reshape(1, -1), np.asarray([float(best_fit)], dtype=float)
        return None, None

    def _make_ids(self, step: int, n: int) -> list[str]:
        return [f"{self.algorithm_id}:s{int(step)}:i{int(i)}" for i in range(max(0, int(n)))]

    def _capture(self, state: Any) -> _StepSnapshot:
        pos, fit = self._extract_population(state)
        step = int(getattr(state, "step", 0) or 0)
        ids = self._make_ids(step, 0 if pos is None else int(pos.shape[0]))
        return _StepSnapshot(
            step=step,
            evaluations=int(getattr(state, "evaluations", 0) or 0),
            best_fitness=None if getattr(state, "best_fitness", None) is None else float(getattr(state, "best_fitness")),
            positions=pos,
            fitness=fit,
            ids=ids,
        )

    def _source_label(self) -> tuple[str, str]:
        # Called from record_evaluation -> ProblemSpec.evaluate -> actual engine helper.
        # The resolver maps exact source call-sites to explicit labels generated
        # from the faithful engine files. This keeps the algorithms untouched
        # while avoiding generic labels such as ``wca._step_impl`` whenever the
        # source exposes a more precise internal section.
        try:
            frame = sys._getframe(3)
        except Exception:
            return f"{self.algorithm_id}.unknown", "unknown"
        try:
            from .evomapx_operator_catalog import resolve_operator_label
        except Exception:
            resolve_operator_label = None
        fallback = None
        for _ in range(16):
            if frame is None:
                break
            name = frame.f_code.co_name
            filename = frame.f_code.co_filename or ""
            base = os.path.basename(filename)
            line = int(getattr(frame, "f_lineno", 0) or 0)
            if name not in _SKIP_FUNCTIONS and not base.startswith("protocol"):
                label = None
                if resolve_operator_label is not None:
                    try:
                        label = resolve_operator_label(self.algorithm_id, base, name, line)
                    except Exception:
                        label = None
                if label:
                    return label, f"{base}::{name}:{line}"
                if fallback is None:
                    fallback = (f"{self.algorithm_id}.{(name.strip('_') or 'step')}_evaluation_l{line}", f"{base}::{name}:{line}")
            frame = frame.f_back
        if fallback is not None:
            return fallback
        return f"{self.algorithm_id}.step", "step"

    def begin_run(self) -> None:
        if not self.enabled:
            return
        self.snapshots.clear(); self.records.clear(); self.raw_events.clear()
        self._step_evals.clear(); self._before = None; self._last_lineage = []

    def after_initialize(self, state: Any) -> None:
        if not self.enabled or self.level < 1:
            return
        snap = self._capture(state)
        self._append_snapshot(snap, phase="initial")

    def before_step(self, state: Any) -> None:
        if not self.enabled:
            return
        self._before = self._capture(state)
        self._active_step = int(getattr(state, "step", 0) or 0) + 1
        self._step_evals = []

    def record_evaluation(self, position: Any, fitness: float, raw_fitness: float | None = None, details: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.level < 2 or self._active_step is None:
            return
        pos = _safe_array(position)
        label, source = self._source_label()
        event = {
            "step": int(self._active_step),
            "algorithm": self.algorithm_id,
            "operator": label,
            "source": source,
            "fitness": float(fitness),
            "raw_fitness": None if raw_fitness is None else float(raw_fitness),
            "position": None if pos is None else pos.tolist(),
        }
        self._step_evals.append(event)
        if self.keep_events:
            self.raw_events.append(dict(event))

    def _append_snapshot(self, snap: _StepSnapshot, phase: str) -> None:
        if self.level < 1 or snap.positions is None:
            return
        pop = []
        for i in range(snap.positions.shape[0]):
            item = {
                "id": snap.ids[i] if i < len(snap.ids) else f"{self.algorithm_id}:s{snap.step}:i{i}",
                "position": snap.positions[i].astype(float).tolist(),
                "fitness": None if snap.fitness is None else float(snap.fitness[i]),
                "operator": "initialization" if phase == "initial" else None,
                "parent_ids": [],
                "metadata": {"phase": phase},
            }
            pop.append(item)
        self.snapshots.append({"step": int(snap.step), "phase": phase, "population": pop})

    def _operator_scores(self, before_best: float | None) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
        counts: dict[str, int] = {}
        raw_scores: dict[str, float] = {}
        candidate_sum: dict[str, float] = {}
        for ev in self._step_evals:
            op = str(ev.get("operator") or f"{self.algorithm_id}.step")
            delta = _fitness_better_delta(before_best, ev.get("fitness"), self.objective)
            for split_op, split_delta, _share, _split_size in _expanded_operator_parts(self.algorithm_id, op, float(delta)):
                counts[split_op] = counts.get(split_op, 0) + 1
                raw_scores[split_op] = raw_scores.get(split_op, 0.0) + float(split_delta)
                candidate_sum[split_op] = candidate_sum.get(split_op, 0.0) + float(split_delta)
        return raw_scores, counts, candidate_sum

    def _explicit_label_for_index(self, operator_labels: Any, index: int) -> str | None:
        """Return an engine-provided passive operator label for one candidate.

        Engines may optionally place ``operator_labels`` or
        ``evomapx_operator_labels`` in ``state.payload``. This is observational
        metadata only: the base runner does not use it to alter the population.
        """
        if operator_labels is None:
            return None
        try:
            if isinstance(operator_labels, dict):
                value = operator_labels.get(index, operator_labels.get(str(index)))
            else:
                value = operator_labels[index] if index < len(operator_labels) else None
        except Exception:
            return None
        if value in {None, ""}:
            return None
        return str(value)

    def _event_label_for_candidate(self, position: Any, fitness: Any, used_events: set[int]) -> str | None:
        """Match an accepted population member to the evaluation event that produced it.

        This keeps the probe passive: it uses already-recorded candidate
        positions/fitnesses and never re-evaluates the objective. Matching is
        deliberately conservative; if no near-exact event is found, callers fall
        back to the dominant source-level label for the step.
        """
        pos = _safe_array(position)
        if pos is None or not self._step_evals:
            return None
        try:
            fit = float(fitness)
        except Exception:
            fit = None
        best_j = None
        best_dist = float("inf")
        for j, ev in enumerate(self._step_evals):
            if j in used_events:
                continue
            ev_pos = _safe_array(ev.get("position"))
            if ev_pos is None or ev_pos.shape != pos.shape:
                continue
            dist = float(np.linalg.norm(ev_pos - pos))
            if dist < best_dist:
                if fit is not None and ev.get("fitness") is not None:
                    try:
                        if abs(float(ev.get("fitness")) - fit) > 1.0e-7 * max(1.0, abs(fit)):
                            continue
                    except Exception:
                        pass
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= 1.0e-8:
            used_events.add(best_j)
            return str(self._step_evals[best_j].get("operator") or f"{self.algorithm_id}.step")
        return None

    def _match_lineage(self, before: _StepSnapshot | None, after: _StepSnapshot, dominant_operator: str, operator_labels: Any = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        metrics = {"mean_displacement": 0.0, "max_displacement": 0.0, "changed_count": 0.0}
        if self.level < 3 or after.positions is None:
            return [], metrics
        line: list[dict[str, Any]] = []
        if before is None or before.positions is None or before.positions.size == 0:
            for i in range(after.positions.shape[0]):
                child_fit = None if after.fitness is None else float(after.fitness[i])
                line.append({
                    "id": after.ids[i], "parent_ids": [], "operator": "initialization",
                    "index": int(i), "step": int(after.step), "distance": 0.0,
                    "parent_index": None, "parent_fitness": None, "child_fitness": child_fit,
                    "lineage_delta": 0.0,
                })
            return line, metrics
        prev = np.asarray(before.positions, dtype=float)
        curr = np.asarray(after.positions, dtype=float)
        displacements = []
        changed = 0
        used_events: set[int] = set()
        for i, cvec in enumerate(curr):
            d = np.linalg.norm(prev - cvec[None, :], axis=1)
            parent_idx = int(np.argmin(d))
            dist = float(d[parent_idx])
            displacements.append(dist)
            parent_id = before.ids[parent_idx] if parent_idx < len(before.ids) else f"{self.algorithm_id}:s{before.step}:i{parent_idx}"
            parent_fit = None if before.fitness is None else float(before.fitness[parent_idx])
            child_fit = None if after.fitness is None else float(after.fitness[i])
            delta = _fitness_better_delta(parent_fit, child_fit, self.objective)
            if dist <= 1.0e-12:
                op = "carryover"
            else:
                changed += 1
                op = (
                    self._explicit_label_for_index(operator_labels, i)
                    or self._event_label_for_candidate(cvec, child_fit, used_events)
                    or dominant_operator
                )
            line.append({
                "id": after.ids[i] if i < len(after.ids) else f"{self.algorithm_id}:s{after.step}:i{i}",
                "parent_ids": [parent_id],
                "operator": op,
                "index": int(i),
                "step": int(after.step),
                "distance": dist,
                "parent_index": int(parent_idx),
                "parent_fitness": parent_fit,
                "child_fitness": child_fit,
                "lineage_delta": float(delta),
            })
        if displacements:
            metrics["mean_displacement"] = float(np.mean(displacements))
            metrics["max_displacement"] = float(np.max(displacements))
            metrics["changed_count"] = float(changed)
        return line, metrics

    def _lineage_scores(self, lineage: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
        """Aggregate signed parent→child Δf by candidate operator label."""
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        candidate_sum: dict[str, float] = {}
        for lin in lineage or []:
            if not isinstance(lin, dict):
                continue
            op = str(lin.get("operator") or f"{self.algorithm_id}.step")
            # Keep carryover in lineage snapshots, but omit it from OAM/CDS unless
            # it actually has a nonzero delta (normally it does not).
            delta = float(lin.get("lineage_delta") or 0.0)
            if op == "initialization":
                continue
            if op == "carryover" and abs(delta) <= 1.0e-14:
                continue
            for split_op, split_delta, _share, _split_size in _expanded_operator_parts(self.algorithm_id, op, delta):
                if split_op == "initialization":
                    continue
                if split_op == "carryover" and abs(split_delta) <= 1.0e-14:
                    continue
                scores[split_op] = scores.get(split_op, 0.0) + float(split_delta)
                candidate_sum[split_op] = candidate_sum.get(split_op, 0.0) + float(split_delta)
                counts[split_op] = counts.get(split_op, 0) + 1
        return scores, counts, candidate_sum

    def after_step(self, state: Any, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return observation
        obs = dict(observation or {})
        after = self._capture(state)
        before = self._before
        before_best = None if before is None else before.best_fitness
        macro_delta = _fitness_better_delta(before_best, after.best_fitness, self.objective)

        # Evaluation-event telemetry is still useful for activity/counts and as a
        # fallback when lineage is unavailable. It is not used as the primary CDS
        # source when parent→child lineage can be computed.
        event_scores, event_counts, event_candidate_sum = self._operator_scores(before_best) if self.level >= 2 else ({}, {}, {})
        if self.level >= 2 and not event_counts:
            op = f"{self.algorithm_id}.step"
            event_counts = {op: 1}
            event_scores = {op: float(macro_delta)}
            event_candidate_sum = {op: float(macro_delta)}

        # Choose a fallback label for changed candidates whose exact event/branch
        # cannot be matched. This preserves the previous source-location behavior.
        dominant = None
        if event_scores:
            dominant = max(event_scores, key=lambda k: abs(float(event_scores[k])))
        if dominant is None:
            dominant = f"{self.algorithm_id}.step"

        operator_labels = None
        try:
            payload = getattr(state, "payload", {}) or {}
            operator_labels = payload.get("operator_labels", payload.get("evomapx_operator_labels"))
        except Exception:
            operator_labels = None

        lineage, movement = self._match_lineage(before, after, dominant, operator_labels=operator_labels)
        self._last_lineage = lineage

        if self.level >= 2:
            lineage_scores, lineage_counts, lineage_candidate_sum = self._lineage_scores(lineage)
            use_lineage = bool(lineage_counts)
            raw_scores = lineage_scores if use_lineage else event_scores
            counts = lineage_counts if use_lineage else event_counts
            candidate_sum = lineage_candidate_sum if use_lineage else event_candidate_sum

            # Observation-level contributions are sums for the step. The OAM/CDS
            # builder can later aggregate them as mean or sum.
            contributions = {str(op): float(val) for op, val in (raw_scores or {}).items()}
            obs["operator_contributions"] = contributions
            obs["operator_counts"] = {str(k): int(v) for k, v in (counts or {}).items()}
            obs["evomapx_delta_f"] = "signed_parent_child" if use_lineage else "signed_event_fallback"
            obs["evomapx_attribution_available"] = True
            obs["evomapx_fidelity_runtime"] = "passive_probe_lineage_delta" if use_lineage else "passive_probe_event_fallback"
            obs["evomapx_probe_level"] = int(self.level)
            obs["evomapx_macro_delta"] = float(macro_delta)
            obs["evomapx_operator_candidate_delta_sum"] = {str(k): float(v) for k, v in (candidate_sum or {}).items()}

            # Store one record per candidate-level parent→child transition when
            # lineage exists. This is the closest package-wide implementation of
            # the paper's OAM quantity Δf = f(parent) - f(child).
            if use_lineage:
                for lin in lineage:
                    if not isinstance(lin, dict):
                        continue
                    op = str(lin.get("operator") or f"{self.algorithm_id}.step")
                    delta = float(lin.get("lineage_delta") or 0.0)
                    if op == "initialization":
                        continue
                    if op == "carryover" and abs(delta) <= 1.0e-14:
                        continue
                    for split_op, split_delta, split_share, split_size in _expanded_operator_parts(self.algorithm_id, op, delta):
                        if split_op == "initialization":
                            continue
                        if split_op == "carryover" and abs(split_delta) <= 1.0e-14:
                            continue
                        self.records.append({
                            "step": int(after.step),
                            "operator": split_op,
                            "algorithm": self.algorithm_id,
                            "raw_improvement": float(split_delta),
                            "positive_improvement": float(max(0.0, split_delta)),
                            "n_applications": 1,
                            "source": "evomapx_lineage_delta",
                            "metadata": {
                                "macro_delta": float(macro_delta),
                                "parent_id": (lin.get("parent_ids") or [None])[0],
                                "child_id": lin.get("id"),
                                "parent_index": lin.get("parent_index"),
                                "index": lin.get("index"),
                                "parent_fitness": lin.get("parent_fitness"),
                                "child_fitness": lin.get("child_fitness"),
                                "distance": lin.get("distance"),
                                "original_operator": op,
                                "compound_split_size": int(split_size),
                                "compound_split_fraction": float(split_share),
                            },
                        })
            else:
                for op, val in contributions.items():
                    for split_op, split_delta, split_share, split_size in _expanded_operator_parts(self.algorithm_id, op, float(val)):
                        self.records.append({
                            "step": int(after.step),
                            "operator": split_op,
                            "algorithm": self.algorithm_id,
                            "raw_improvement": float(split_delta),
                            "positive_improvement": float(max(0.0, split_delta)),
                            "n_applications": int(counts.get(op, counts.get(split_op, 1))),
                            "source": "evomapx_event_fallback",
                            "metadata": {
                                "macro_delta": float(macro_delta),
                                "candidate_delta_sum": float(candidate_sum.get(op, candidate_sum.get(split_op, 0.0))),
                                "original_operator": op,
                                "compound_split_size": int(split_size),
                                "compound_split_fraction": float(split_share),
                            },
                        })
        else:
            contributions = {}
            counts = {}

        if self.level >= 3:
            obs["evomapx_lineage"] = lineage
            try:
                state.payload["lineage"] = [dict(x) for x in lineage]
            except Exception:
                pass

        if self.level >= 4:
            lo, hi = self._problem_bounds()
            div_before = _diversity(None if before is None else before.positions, lo, hi)
            div_after = _diversity(after.positions, lo, hi)
            accepted = int(movement.get("changed_count", 0.0))
            total_trials = max(1, sum(int(v) for v in (counts or {}).values()) or (0 if after.positions is None else after.positions.shape[0]))
            dominant_activity = None
            if contributions:
                positive = {k: max(0.0, float(v)) for k, v in contributions.items()}
                if sum(positive.values()) > 0:
                    dominant_activity = max(positive, key=positive.get)
                else:
                    dominant_activity = max(contributions, key=lambda k: abs(float(contributions[k])))
            obs["evomapx_activity"] = {
                "diversity_before": div_before,
                "diversity_after": div_after,
                "diversity_delta": None if div_before is None or div_after is None else float(div_after - div_before),
                "mean_displacement": float(movement.get("mean_displacement", 0.0)),
                "max_displacement": float(movement.get("max_displacement", 0.0)),
                "changed_count": accepted,
                "candidate_evaluations_observed": int(sum(int(v) for v in (event_counts or {}).values())),
                "lineage_transitions_observed": int(sum(int(v) for v in (counts or {}).values())),
                "acceptance_rate_inferred": float(accepted / max(1, int(sum(int(v) for v in (event_counts or {}).values()) or total_trials))),
                "dominant_operator": dominant_activity or dominant,
            }

        if self.level >= 1:
            self._append_snapshot(after, phase="step")
        self._before = None
        self._active_step = None
        self._step_evals = []
        return obs

    def enrich_population_snapshot(self, state: Any, population: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or self.level < 3 or not population:
            return population
        lineage_by_index = {int(x.get("index", -1)): x for x in self._last_lineage if isinstance(x, dict)}
        enriched = []
        step = int(getattr(state, "step", 0) or 0)
        for i, cand in enumerate(population):
            item = dict(cand)
            lin = lineage_by_index.get(i)
            item.setdefault("id", f"{self.algorithm_id}:s{step}:i{i}")
            if lin:
                item["id"] = lin.get("id", item["id"])
                item["parent_ids"] = list(lin.get("parent_ids", []))
                item["operator"] = lin.get("operator")
                item.setdefault("metadata", {})
                item["metadata"] = {
                    **dict(item.get("metadata", {}) or {}),
                    "distance_from_parent": lin.get("distance"),
                    "parent_fitness": lin.get("parent_fitness"),
                    "child_fitness": lin.get("child_fitness"),
                    "lineage_delta": lin.get("lineage_delta"),
                }
            enriched.append(item)
        return enriched

    def finalize_result(self, result: Any) -> Any:
        if not self.enabled:
            return result
        meta = getattr(result, "metadata", None)
        if not isinstance(meta, dict):
            return result
        meta["evomapx_probe"] = {
            "enabled": True,
            "level": int(self.level),
            "algorithm": self.algorithm_id,
            "records": int(len(self.records)),
            "snapshots": int(len(self.snapshots)),
            "raw_events_stored": int(len(self.raw_events)),
            "mode": "passive_non_intrusive",
        }
        meta["evomapx_records"] = [_jsonable(r) for r in self.records]
        if self.keep_events:
            meta["evomapx_raw_events"] = [_jsonable(e) for e in self.raw_events]
        # Keep compact Level-1 snapshots out of the primary result unless the
        # caller explicitly requests them via standard population_snapshots; make
        # them available for audits without duplicating huge arrays by default.
        meta["evomapx_probe_snapshot_count"] = int(len(self.snapshots))
        return result


__all__ = ["EvoMapXProbe"]
