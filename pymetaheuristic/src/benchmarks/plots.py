"""Matplotlib visualizations for benchmark results."""

from __future__ import annotations

import numpy as np


def _get_ax(ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    return ax


def _default_metric(df):
    if "error_to_optimum" in df.columns and df["error_to_optimum"].notna().any():
        return "error_to_optimum"
    return "best_fitness"


def plot_convergence(result, ax=None, metric: str | None = None, problem: str | None = None, show_iqr: bool = True):
    """Plot median convergence curves by candidate."""
    import pandas as pd

    ax = _get_ax(ax)
    df = result.convergence_dataframe()
    if df.empty:
        ax.set_title("No convergence history available")
        return ax
    if problem is not None:
        df = df[df["problem"] == problem]
    if df.empty:
        ax.set_title("No convergence history for selected problem")
        return ax
    metric = metric or _default_metric(df)
    if metric not in df.columns or not df[metric].notna().any():
        metric = "best_fitness"
    # Bin by observed evaluation counts to avoid requiring identical traces.
    for candidate, g in df.groupby("candidate"):
        rows = []
        for evals, h in g.groupby("evaluations"):
            vals = h[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append({
                "evaluations": float(evals),
                "median": float(np.median(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
            })
        curve = pd.DataFrame(rows).sort_values("evaluations")
        if curve.empty:
            continue
        ax.plot(curve["evaluations"], curve["median"], label=str(candidate))
        if show_iqr:
            ax.fill_between(curve["evaluations"].to_numpy(), curve["q25"].to_numpy(), curve["q75"].to_numpy(), alpha=0.15)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(metric)
    ax.set_title("Benchmark convergence")
    ax.legend()
    return ax


def plot_ecdf(result, ax=None, target_tolerance: float | None = None):
    """Plot ECDF of evaluations needed to reach the target."""
    ax = _get_ax(ax)
    df = result.to_dataframe()
    if df.empty or "evaluations_to_target" not in df.columns:
        ax.set_title("No target-hitting data available")
        return ax
    if target_tolerance is not None and "target_tolerance" in df.columns:
        df = df[np.isclose(df["target_tolerance"].astype(float), float(target_tolerance))]
    budget_max = float(np.nanmax(df["budget"].fillna(df["evaluations"].max()).to_numpy(dtype=float))) if len(df) else 1.0
    xs = np.linspace(0.0, max(1.0, budget_max), 100)
    for candidate, g in df.groupby("candidate"):
        hits = g["evaluations_to_target"].dropna().to_numpy(dtype=float)
        if len(g) == 0:
            continue
        ys = [float(np.mean(hits <= x)) if len(hits) else 0.0 for x in xs]
        ax.plot(xs, ys, label=str(candidate))
    ax.set_xlabel("Evaluations to target")
    ax.set_ylabel("Solved fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ECDF of target hits")
    ax.legend()
    return ax


def plot_performance_profile(result, ax=None, metric: str | None = None, higher_is_better: bool | None = None, tau_max: float = 10.0):
    """Plot a Dolan-Moré style performance profile."""
    import pandas as pd

    ax = _get_ax(ax)
    df = result.to_dataframe()
    if df.empty:
        ax.set_title("No benchmark records")
        return ax
    metric = metric or _default_metric(df)
    if higher_is_better is None:
        higher_is_better = metric in {"success", "success_rate"}
    keys = ["problem", "dimension", "budget", "trial"]
    table = df.pivot_table(index=keys, columns="candidate", values=metric, aggfunc="mean").dropna(axis=0, how="any")
    if table.empty:
        ax.set_title("No paired records for performance profile")
        return ax
    values = table.to_numpy(dtype=float)
    eps = 1.0e-300
    if higher_is_better:
        best = np.nanmax(values, axis=1)
        ratios = np.where(values > 0, best[:, None] / np.maximum(values, eps), np.inf)
    else:
        shifted = values.copy()
        min_val = np.nanmin(shifted)
        if min_val <= 0.0:
            shifted = shifted - min_val + 1.0e-12
        best = np.nanmin(shifted, axis=1)
        ratios = shifted / np.maximum(best[:, None], eps)
    taus = np.linspace(1.0, tau_max, 150)
    for j, candidate in enumerate(table.columns):
        r = ratios[:, j]
        y = [float(np.mean(r <= tau)) for tau in taus]
        ax.plot(taus, y, label=str(candidate))
    ax.set_xlabel("Performance ratio τ")
    ax.set_ylabel("Fraction of instances")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Performance profile: {metric}")
    ax.legend()
    return ax


def plot_rank_heatmap(result, ax=None, metric: str | None = None, higher_is_better: bool | None = None):
    """Plot median candidate ranks by problem."""
    ax = _get_ax(ax)
    df = result.to_dataframe()
    if df.empty:
        ax.set_title("No benchmark records")
        return ax
    metric = metric or _default_metric(df)
    if higher_is_better is None:
        higher_is_better = metric in {"success", "success_rate"}
    keys = ["problem", "dimension", "budget", "trial"]
    table = df.pivot_table(index=keys, columns="candidate", values=metric, aggfunc="mean").dropna(axis=0, how="any")
    if table.empty:
        ax.set_title("No paired records for rank heatmap")
        return ax
    ranks = table.rank(axis=1, method="average", ascending=not higher_is_better).reset_index()
    problem_labels = ranks.apply(lambda row: f"{row['problem']} D{int(row['dimension'])}", axis=1)
    ranks["problem_label"] = problem_labels
    heat = ranks.groupby("problem_label")[list(table.columns)].median()
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(list(heat.index))
    ax.set_title(f"Median ranks by problem: {metric}")
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Problem")
    try:
        import matplotlib.pyplot as plt
        plt.colorbar(image, ax=ax, label="Median rank")
    except Exception:
        pass
    return ax


__all__ = ["plot_convergence", "plot_ecdf", "plot_performance_profile", "plot_rank_heatmap"]
