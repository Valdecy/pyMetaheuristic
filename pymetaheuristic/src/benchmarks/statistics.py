"""Statistical summaries for benchmark studies."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _default_metric(df) -> tuple[str, bool]:
    if "error_to_optimum" in df.columns and df["error_to_optimum"].notna().any():
        return "error_to_optimum", False
    return "best_fitness", False


def _metric_and_direction(result, metric: str | None = None, higher_is_better: bool | None = None):
    df = result.to_dataframe()
    if metric is None:
        metric, default_higher = _default_metric(df)
    else:
        default_higher = False
    if higher_is_better is None:
        if metric in {"success", "success_rate"}:
            higher_is_better = True
        elif metric in {"runtime_seconds", "evaluations_to_target", "error_to_optimum", "best_fitness"}:
            higher_is_better = False
        else:
            higher_is_better = default_higher
    return metric, bool(higher_is_better)


def _score_column(df, metric: str) -> str:
    if metric not in df.columns:
        raise KeyError(f"Metric {metric!r} is not available. Columns: {', '.join(df.columns)}")
    return metric


def summary_table(result):
    """Return candidate-level descriptive statistics."""
    import pandas as pd

    df = result.to_dataframe()
    if df.empty:
        return pd.DataFrame()
    metric, _ = _metric_and_direction(result)
    _score_column(df, metric)
    grouped = df.groupby("candidate", dropna=False)
    summary = grouped.agg(
        n=(metric, "count"),
        mean=(metric, "mean"),
        median=(metric, "median"),
        std=(metric, "std"),
        min=(metric, "min"),
        max=(metric, "max"),
        mean_best_fitness=("best_fitness", "mean"),
        median_best_fitness=("best_fitness", "median"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_evaluations=("evaluations", "mean"),
    ).reset_index()
    if "success" in df.columns:
        success = grouped["success"].mean().rename("success_rate").reset_index()
        summary = summary.merge(success, on="candidate", how="left")
    return summary.sort_values("median", ascending=True).reset_index(drop=True)


def _instance_table(result, metric: str | None = None, higher_is_better: bool | None = None):
    import pandas as pd

    df = result.to_dataframe()
    if df.empty:
        return pd.DataFrame(), "", False
    metric, higher = _metric_and_direction(result, metric, higher_is_better)
    _score_column(df, metric)
    keys = ["problem", "dimension", "budget", "trial"]
    table = df.pivot_table(index=keys, columns="candidate", values=metric, aggfunc="mean")
    table = table.dropna(axis=0, how="any")
    return table, metric, higher


def rank_table(result, metric: str | None = None, higher_is_better: bool | None = None):
    """Return mean-rank table across problem/trial/budget instances."""
    import pandas as pd

    table, metric, higher = _instance_table(result, metric, higher_is_better)
    if table.empty:
        return pd.DataFrame(columns=["candidate", "mean_rank", "median_rank", "wins", "n_instances", "metric"])
    ranks = table.rank(axis=1, method="average", ascending=not higher)
    wins = ranks.eq(1.0).sum(axis=0)
    out = pd.DataFrame({
        "candidate": ranks.columns,
        "mean_rank": ranks.mean(axis=0).values,
        "median_rank": ranks.median(axis=0).values,
        "best_rank_count": wins.values,
        "n_instances": [int(ranks.shape[0])] * ranks.shape[1],
        "metric": [metric] * ranks.shape[1],
        "higher_is_better": [higher] * ranks.shape[1],
    })
    return out.sort_values(["mean_rank", "candidate"], ascending=[True, True]).reset_index(drop=True)


def friedman_test(result, metric: str | None = None) -> dict[str, Any]:
    """Friedman rank test over paired candidate scores.

    Uses SciPy when available; otherwise returns Iman-Davenport-like basic rank
    information without a p-value.
    """
    table, metric, higher = _instance_table(result, metric, None)
    if table.empty or table.shape[1] < 2 or table.shape[0] < 2:
        return {
            "metric": metric,
            "statistic": None,
            "p_value": None,
            "n_instances": int(table.shape[0]) if not table.empty else 0,
            "n_candidates": int(table.shape[1]) if not table.empty else 0,
            "message": "Need at least two candidates and two paired instances.",
        }
    arrays = [table[col].to_numpy(dtype=float) for col in table.columns]
    try:
        from scipy.stats import friedmanchisquare
        stat, p = friedmanchisquare(*arrays)
        return {
            "metric": metric,
            "statistic": float(stat),
            "p_value": float(p),
            "n_instances": int(table.shape[0]),
            "n_candidates": int(table.shape[1]),
            "candidates": list(table.columns),
            "higher_is_better": bool(higher),
        }
    except Exception as exc:  # pragma: no cover - fallback for scipy-less envs
        return {
            "metric": metric,
            "statistic": None,
            "p_value": None,
            "n_instances": int(table.shape[0]),
            "n_candidates": int(table.shape[1]),
            "candidates": list(table.columns),
            "higher_is_better": bool(higher),
            "message": f"SciPy unavailable or failed: {exc}",
        }


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * float(p))
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / (len(x) * len(y)))


def wilcoxon_pairwise(result, metric: str | None = None) -> list[dict[str, Any]]:
    """Pairwise Wilcoxon signed-rank tests with Holm correction."""
    table, metric, higher = _instance_table(result, metric, None)
    if table.empty or table.shape[1] < 2:
        return []
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for a, b in combinations(table.columns, 2):
        x = table[a].to_numpy(dtype=float)
        y = table[b].to_numpy(dtype=float)
        diff = x - y
        try:
            from scipy.stats import wilcoxon
            if np.allclose(diff, 0.0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            statistic = float(stat)
            p_value = float(p)
        except Exception as exc:  # pragma: no cover
            statistic = None
            p_value = 1.0
        mean_a = float(np.mean(x))
        mean_b = float(np.mean(y))
        if higher:
            winner = a if mean_a > mean_b else b if mean_b > mean_a else "tie"
        else:
            winner = a if mean_a < mean_b else b if mean_b < mean_a else "tie"
        row = {
            "candidate_a": a,
            "candidate_b": b,
            "metric": metric,
            "higher_is_better": bool(higher),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "winner_by_mean": winner,
            "statistic": statistic,
            "p_value": p_value,
            "cliffs_delta_a_minus_b": cliffs_delta(x, y),
            "n_instances": int(len(x)),
        }
        rows.append(row)
        p_values.append(p_value)
    adjusted = _holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["holm_p_value"] = float(p_adj)
        row["significant_0_05"] = bool(p_adj < 0.05)
    return rows


__all__ = [
    "summary_table",
    "rank_table",
    "friedman_test",
    "wilcoxon_pairwise",
    "cliffs_delta",
]
