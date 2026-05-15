"""Benchmark records and result containers."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExperimentRecord:
    """One candidate/problem/trial/budget outcome."""

    study_id: str
    candidate: str
    candidate_type: str
    problem: str
    dimension: int
    objective: str
    trial: int
    seed: int
    budget: int | None
    max_steps: int | None
    evaluations: int | None
    steps: int | None
    best_fitness: float
    error_to_optimum: float | None = None
    evaluations_to_target: int | None = None
    target_tolerance: float | None = None
    success: bool | None = None
    runtime_seconds: float | None = None
    termination_reason: str | None = None
    controller_mode: str | None = None
    topology: str | None = None
    migration_interval: int | None = None
    migration_size: int | None = None
    n_events: int | None = None
    n_checkpoints: int | None = None
    n_decisions: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    convergence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkResult:
    """Analysis object returned by :class:`BenchmarkStudy.run`."""

    def __init__(self, records: list[ExperimentRecord], manifest: dict[str, Any] | None = None) -> None:
        self.records = list(records)
        self.manifest = dict(manifest or {})

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([record.to_dict() for record in self.records])

    def convergence_dataframe(self):
        import pandas as pd
        rows: list[dict[str, Any]] = []
        for record in self.records:
            for point in record.convergence or []:
                rows.append({
                    "study_id": record.study_id,
                    "candidate": record.candidate,
                    "candidate_type": record.candidate_type,
                    "problem": record.problem,
                    "dimension": record.dimension,
                    "trial": record.trial,
                    "seed": record.seed,
                    "budget": record.budget,
                    "objective": record.objective,
                    "evaluations": point.get("evaluations"),
                    "step": point.get("step"),
                    "best_fitness": point.get("best_fitness"),
                    "error_to_optimum": point.get("error_to_optimum"),
                })
        return pd.DataFrame(rows)

    def summary(self):
        from .statistics import summary_table
        return summary_table(self)

    def rank_table(self, metric: str | None = None, higher_is_better: bool | None = None):
        from .statistics import rank_table
        return rank_table(self, metric=metric, higher_is_better=higher_is_better)

    def friedman_test(self, metric: str | None = None):
        from .statistics import friedman_test
        return friedman_test(self, metric=metric)

    def wilcoxon_pairwise(self, metric: str | None = None):
        from .statistics import wilcoxon_pairwise
        return wilcoxon_pairwise(self, metric=metric)

    def statistical_tests(self, metric: str | None = None) -> dict[str, Any]:
        return {
            "friedman": self.friedman_test(metric=metric),
            "wilcoxon_pairwise": self.wilcoxon_pairwise(metric=metric),
            "rank_table": self.rank_table(metric=metric),
        }

    def scientific_summary(self, metric: str | None = None) -> dict[str, Any]:
        df = self.summary()
        ranks = self.rank_table(metric=metric)
        best_candidate = None
        best_mean_rank = None
        if not ranks.empty:
            best = ranks.sort_values("mean_rank", ascending=True).iloc[0]
            best_candidate = best["candidate"]
            best_mean_rank = float(best["mean_rank"])
        return {
            "n_records": len(self.records),
            "n_candidates": int(df["candidate"].nunique()) if not df.empty else 0,
            "n_problems": int(df["problem"].nunique()) if not df.empty else 0,
            "best_mean_rank_candidate": best_candidate,
            "best_mean_rank": best_mean_rank,
            "summary": df,
            "rank_table": ranks,
        }

    def plot_convergence(self, *args, **kwargs):
        from .plots import plot_convergence
        return plot_convergence(self, *args, **kwargs)

    def plot_ecdf(self, *args, **kwargs):
        from .plots import plot_ecdf
        return plot_ecdf(self, *args, **kwargs)

    def plot_performance_profile(self, *args, **kwargs):
        from .plots import plot_performance_profile
        return plot_performance_profile(self, *args, **kwargs)

    def plot_rank_heatmap(self, *args, **kwargs):
        from .plots import plot_rank_heatmap
        return plot_rank_heatmap(self, *args, **kwargs)

    def save(self, filepath: str | Path) -> str:
        path = Path(filepath)
        payload = {
            "manifest": self.manifest,
            "records": [record.to_dict() for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, filepath: str | Path) -> "BenchmarkResult":
        payload = json.loads(Path(filepath).read_text(encoding="utf-8"))
        records = [ExperimentRecord(**item) for item in payload.get("records", [])]
        return cls(records=records, manifest=payload.get("manifest", {}))



def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)



def build_manifest(**kwargs) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        **kwargs,
    }



def load_benchmark(filepath: str | Path) -> BenchmarkResult:
    return BenchmarkResult.load(filepath)


__all__ = ["ExperimentRecord", "BenchmarkResult", "build_manifest", "load_benchmark"]
