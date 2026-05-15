"""Core benchmark-study runner for algorithms and island systems."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from ..api import optimize
from ..islands import IslandSystem
from .problems import BenchmarkProblem, ProblemSuite
from .records import BenchmarkResult, ExperimentRecord, build_manifest


def _safe_name(value: Any) -> str:
    text = str(value)
    return text if text else "candidate"


class BenchmarkStudy:
    """Run reproducible benchmark studies over algorithms and island systems.

    The study stores one long-format record per candidate/problem/trial/budget.
    Candidates may be ordinary algorithms or :class:`IslandSystem` objects.
    """

    def __init__(
        self,
        candidates: list[Any] | None = None,
        *,
        algorithms: list[Any] | None = None,
        systems: list[Any] | None = None,
        problems: Any = None,
        n_trials: int = 30,
        max_steps: int | None = None,
        max_evaluations: int | None = None,
        budgets: list[int] | tuple[int, ...] | None = None,
        target_tolerance: float = 1.0e-8,
        seed: int | None = None,
        objective: str = "min",
        study_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        store_convergence: bool = True,
        verbose: bool = False,
    ) -> None:
        self.study_id = study_id or f"benchmark_{uuid.uuid4().hex[:10]}"
        self.objective = objective
        self.n_trials = int(n_trials)
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1.")
        self.max_steps = max_steps
        self.max_evaluations = max_evaluations
        if budgets is None:
            budgets = [max_evaluations] if max_evaluations is not None else [None]
        self.budgets = [None if b is None else int(b) for b in list(budgets)]
        self.target_tolerance = float(target_tolerance)
        self.seed = seed
        self.metadata = dict(metadata or {})
        self.store_convergence = bool(store_convergence)
        self.verbose = bool(verbose)
        all_candidates: list[Any] = []
        if candidates is not None:
            all_candidates.extend(candidates)
        if algorithms is not None:
            all_candidates.extend(self._algorithm_candidate(x) for x in algorithms)
        if systems is not None:
            all_candidates.extend(self._system_candidate(x) for x in systems)
        if not all_candidates:
            raise ValueError("At least one candidate, algorithm, or system is required.")
        self.candidates = [self._normalize_candidate(candidate, index=i) for i, candidate in enumerate(all_candidates)]
        self.problems = ProblemSuite.from_any(problems, objective=objective)
        if len(self.problems) == 0:
            raise ValueError("At least one benchmark problem is required.")

    @staticmethod
    def _algorithm_candidate(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            return {"type": "algorithm", "algorithm": value, "name": value, "config": {}}
        if isinstance(value, dict):
            data = dict(value)
            if "type" not in data:
                data["type"] = "algorithm"
            if "algorithm" not in data:
                if "name" in data:
                    data["algorithm"] = data["name"]
                else:
                    raise ValueError("Algorithm dictionaries require 'algorithm' or 'name'.")
            data.setdefault("name", data["algorithm"])
            data.setdefault("config", {})
            return data
        raise TypeError("Algorithm candidates must be strings or dictionaries.")

    @staticmethod
    def _system_candidate(value: Any) -> dict[str, Any]:
        if isinstance(value, IslandSystem):
            mode = value.orchestration.mode if getattr(value, "orchestration", None) is not None else "cooperative"
            return {"type": "island_system", "system": value, "name": value.metadata.get("name", f"island_{mode}"), "mode": mode}
        if isinstance(value, dict):
            data = dict(value)
            data.setdefault("type", "island_system")
            if "system" not in data:
                raise ValueError("System dictionaries require a 'system' entry.")
            data.setdefault("mode", data.get("controller_mode", "cooperative"))
            data.setdefault("name", data.get("label", f"island_{data['mode']}"))
            return data
        raise TypeError("System candidates must be IslandSystem instances or dictionaries.")

    def _normalize_candidate(self, candidate: Any, *, index: int) -> dict[str, Any]:
        if isinstance(candidate, IslandSystem):
            candidate = self._system_candidate(candidate)
        elif isinstance(candidate, str):
            candidate = self._algorithm_candidate(candidate)
        elif not isinstance(candidate, dict):
            raise TypeError("Candidates must be strings, dicts, or IslandSystem objects.")
        data = dict(candidate)
        ctype = str(data.get("type", "algorithm")).lower()
        if ctype in {"algorithm", "optimizer"}:
            algorithm = data.get("algorithm", data.get("name"))
            if not algorithm:
                raise ValueError("Algorithm candidate requires an algorithm id.")
            name = str(data.get("label", data.get("candidate", data.get("name", algorithm))))
            return {
                "name": name,
                "type": "algorithm",
                "algorithm": str(algorithm),
                "config": dict(data.get("config") or {}),
                "metadata": dict(data.get("metadata") or {}),
            }
        if ctype in {"island", "island_system", "system"}:
            system = data.get("system")
            if not isinstance(system, IslandSystem):
                raise TypeError("Island-system candidates require system=IslandSystem(...).")
            mode = str(data.get("mode", data.get("controller_mode", "cooperative"))).lower()
            name = str(data.get("label", data.get("candidate", data.get("name", f"system_{index}"))))
            return {
                "name": name,
                "type": "island_system",
                "system": system,
                "mode": mode,
                "metadata": dict(data.get("metadata") or {}),
            }
        raise ValueError("Candidate type must be 'algorithm' or 'island_system'.")

    def _seed_for(self, candidate_index: int, problem_index: int, trial: int, budget_index: int) -> int | None:
        if self.seed is None:
            return None
        return int(self.seed) + 1000003 * int(trial) + 10007 * int(problem_index) + 101 * int(candidate_index) + int(budget_index)

    def run(self, n_jobs: int = 1) -> BenchmarkResult:
        """Execute the benchmark study.

        ``n_jobs`` is accepted for API stability.  The initial implementation
        runs serially because most user objectives and IslandSystem instances
        are not safely pickleable.  This keeps the result deterministic and
        notebook-friendly.
        """
        records: list[ExperimentRecord] = []
        total = len(self.candidates) * len(self.problems) * self.n_trials * len(self.budgets)
        counter = 0
        for b_idx, budget in enumerate(self.budgets):
            for p_idx, problem in enumerate(self.problems):
                for t_idx in range(self.n_trials):
                    for c_idx, candidate in enumerate(self.candidates):
                        counter += 1
                        if self.verbose:
                            print(f"[{counter}/{total}] {candidate['name']} on {problem.name} trial={t_idx} budget={budget}")
                        seed = self._seed_for(c_idx, p_idx, t_idx, b_idx)
                        record = self._run_one(candidate, problem, trial=t_idx, seed=seed, budget=budget)
                        records.append(record)
        manifest = build_manifest(
            study_id=self.study_id,
            n_trials=self.n_trials,
            seed=self.seed,
            max_steps=self.max_steps,
            max_evaluations=self.max_evaluations,
            budgets=self.budgets,
            target_tolerance=self.target_tolerance,
            objective=self.objective,
            candidates=[self._candidate_manifest(c) for c in self.candidates],
            problems=self.problems.to_dict(),
            metadata=dict(self.metadata or {}),
        )
        return BenchmarkResult(records=records, manifest=manifest)

    def _candidate_manifest(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if candidate["type"] == "algorithm":
            return {
                "name": candidate["name"],
                "type": candidate["type"],
                "algorithm": candidate["algorithm"],
                "config": dict(candidate.get("config") or {}),
                "metadata": dict(candidate.get("metadata") or {}),
            }
        system = candidate.get("system")
        return {
            "name": candidate["name"],
            "type": candidate["type"],
            "mode": candidate.get("mode"),
            "system": system.to_dict() if hasattr(system, "to_dict") else str(system),
            "metadata": dict(candidate.get("metadata") or {}),
        }

    def _run_one(self, candidate: dict[str, Any], problem: BenchmarkProblem, *, trial: int, seed: int | None, budget: int | None) -> ExperimentRecord:
        start = time.perf_counter()
        if candidate["type"] == "algorithm":
            result = optimize(
                candidate["algorithm"],
                target_function=problem.evaluate,
                min_values=problem.min_values,
                max_values=problem.max_values,
                objective=problem.objective,
                max_steps=self.max_steps,
                max_evaluations=budget if budget is not None else self.max_evaluations,
                seed=seed,
                store_history=self.store_convergence,
                config=dict(candidate.get("config") or {}),
            )
            runtime = time.perf_counter() - start
            return self._record_from_result(candidate, problem, result, trial=trial, seed=seed, budget=budget, runtime=runtime)

        system = copy.deepcopy(candidate["system"])
        if seed is not None:
            system.seed = seed
        if self.max_steps is not None:
            system.max_steps = int(self.max_steps)
        if budget is not None:
            system.max_evaluations = int(budget)
        elif self.max_evaluations is not None:
            system.max_evaluations = int(self.max_evaluations)
        system.objective = problem.objective
        result = system.optimize(
            target_function=problem.evaluate,
            min_values=problem.min_values,
            max_values=problem.max_values,
            mode=candidate.get("mode", "cooperative"),
        )
        runtime = time.perf_counter() - start
        return self._record_from_result(candidate, problem, result, trial=trial, seed=seed, budget=budget, runtime=runtime, system=system)

    def _extract_convergence(self, result: Any, problem: BenchmarkProblem) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        history = getattr(result, "history", None) or []
        for item in history:
            if not isinstance(item, dict):
                continue
            fit = item.get("best_fitness", item.get("global_best_fitness"))
            if fit is None:
                continue
            fit = float(fit)
            points.append({
                "step": item.get("step"),
                "evaluations": item.get("evaluations"),
                "best_fitness": fit,
                "error_to_optimum": problem.error(fit),
            })
        # Cooperative/orchestrated histories are often island-level dicts.
        if not points and isinstance(history, list):
            for idx, item in enumerate(history):
                if not isinstance(item, dict):
                    continue
                best_fit = item.get("global_best_fitness", item.get("best_fitness"))
                if best_fit is None:
                    # Try island snapshots.
                    vals = []
                    for value in item.values():
                        if isinstance(value, dict) and "best_fitness" in value:
                            vals.append(float(value["best_fitness"]))
                    if vals:
                        best_fit = min(vals) if problem.objective == "min" else max(vals)
                if best_fit is not None:
                    best_fit = float(best_fit)
                    points.append({
                        "step": item.get("step", idx),
                        "evaluations": item.get("evaluations"),
                        "best_fitness": best_fit,
                        "error_to_optimum": problem.error(best_fit),
                    })
        return points

    def _evaluations_to_target(self, convergence: list[dict[str, Any]], result: Any, problem: BenchmarkProblem) -> int | None:
        if problem.optimum is None:
            return None
        for point in convergence:
            fit = point.get("best_fitness")
            if fit is not None and problem.reached_target(float(fit), tolerance=self.target_tolerance):
                ev = point.get("evaluations")
                return int(ev) if ev is not None else None
        if problem.reached_target(float(getattr(result, "best_fitness")), tolerance=self.target_tolerance):
            ev = getattr(result, "evaluations", None)
            return int(ev) if ev is not None else None
        return None

    def _record_from_result(
        self,
        candidate: dict[str, Any],
        problem: BenchmarkProblem,
        result: Any,
        *,
        trial: int,
        seed: int | None,
        budget: int | None,
        runtime: float,
        system: IslandSystem | None = None,
    ) -> ExperimentRecord:
        best_fitness = float(getattr(result, "best_fitness"))
        error = problem.error(best_fitness)
        convergence = self._extract_convergence(result, problem) if self.store_convergence else []
        evaluations_to_target = self._evaluations_to_target(convergence, result, problem)
        success = None if problem.optimum is None else bool(problem.reached_target(best_fitness, tolerance=self.target_tolerance))
        metadata = dict(candidate.get("metadata") or {})
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            for key in ("n_improvements", "mean_diversity", "final_diversity"):
                if key in result.metadata:
                    metadata[key] = result.metadata.get(key)
        return ExperimentRecord(
            study_id=self.study_id,
            candidate=candidate["name"],
            candidate_type=candidate["type"],
            problem=problem.name,
            dimension=problem.dimension,
            objective=problem.objective,
            trial=int(trial),
            seed=int(seed) if seed is not None else -1,
            budget=budget,
            max_steps=self.max_steps,
            evaluations=getattr(result, "evaluations", None),
            steps=getattr(result, "steps", None),
            best_fitness=best_fitness,
            error_to_optimum=error,
            evaluations_to_target=evaluations_to_target,
            target_tolerance=self.target_tolerance,
            success=success,
            runtime_seconds=float(runtime),
            termination_reason=getattr(result, "termination_reason", None),
            controller_mode=getattr(result, "controller_mode", candidate.get("mode") if candidate["type"] == "island_system" else None),
            topology=getattr(system.topology, "name", None) if system is not None else None,
            migration_interval=getattr(system.migration, "interval", None) if system is not None else None,
            migration_size=getattr(system.migration, "size", None) if system is not None else None,
            n_events=len(getattr(result, "events", []) or []),
            n_checkpoints=len(getattr(result, "checkpoints", []) or []),
            n_decisions=len(getattr(result, "decisions", []) or []),
            metadata=metadata,
            convergence=convergence,
        )


__all__ = ["BenchmarkStudy"]
