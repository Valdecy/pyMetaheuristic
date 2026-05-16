"""Plotly visualizations for benchmark results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

_DARK_BG   = "#0d1117"
_PANEL_BG  = "#161b22"
_GRID_CLR  = "#21262d"
_TEXT_CLR  = "#e6edf3"
_ACCENT    = "#58a6ff"

_DISCRETE_PALETTE = [
    "#3BC9DB", "#F0A500", "#3FB950", "#F85149",
    "#C084FC", "#FB923C", "#34D399", "#60A5FA",
    "#F472B6", "#A3E635", "#FBBF24", "#818CF8",
]
_COLORSCALE = [
    [0.00, "#0d1b4b"], [0.20, "#0c4a6e"], [0.40, "#0e7490"],
    [0.55, "#059669"], [0.70, "#d97706"], [0.85, "#dc2626"], [1.00, "#7f1d1d"],
]
_FONT = dict(family="Inter, Arial, sans-serif", color=_TEXT_CLR, size=13)
_LAYOUT_BASE = dict(
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_PANEL_BG,
    font=_FONT,
    margin=dict(l=65, r=45, t=75, b=65),
    hoverlabel=dict(bgcolor=_PANEL_BG, font_color=_TEXT_CLR, bordercolor=_GRID_CLR),
)
_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=_GRID_CLR,
    zeroline=False,
    linecolor=_GRID_CLR,
    tickfont=dict(color=_TEXT_CLR, size=11),
    title_font=dict(color=_TEXT_CLR, size=13),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) != 6:
        return str(hex_color)
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha):.3g})"


def _save_plotly(fig: go.Figure, filepath, scale: int = 2) -> None:
    if filepath is None:
        return
    from pathlib import Path
    p = Path(str(filepath))
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext in {"", ".html"}:
        if ext == "":
            p = p.with_suffix(".html")
        fig.write_html(str(p))
    elif ext in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"}:
        fig.write_image(str(p), scale=scale)
    else:
        fig.write_html(str(p.with_suffix(".html")))


def _show(fig: go.Figure, show: bool = False, renderer: str = "browser") -> go.Figure:
    if show:
        if renderer:
            pio.renderers.default = renderer
        fig.show()
    return fig


def _base_layout(fig: go.Figure, title: str, width: int = 950, height: int = 560) -> go.Figure:
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=17), x=0.04),
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=width,
        height=height,
    )
    return fig


def _message(title: str, text: str, filepath=None, show: bool = False, renderer: str = "browser") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=text, showarrow=False,
                       font=dict(color=_TEXT_CLR, size=15))
    fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, x=0.04), xaxis=dict(visible=False), yaxis=dict(visible=False), width=950, height=560)
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def _default_metric(df):
    if "error_to_optimum" in df.columns and df["error_to_optimum"].notna().any():
        return "error_to_optimum"
    return "best_fitness"


def plot_convergence(
    result,
    metric: str | None = None,
    problem: str | None = None,
    show_iqr: bool = True,
    filepath=None,
    show: bool = False,
    renderer: str = "browser",
):
    """Plot median convergence curves by candidate. Returns a Plotly figure."""
    df = result.convergence_dataframe()
    if df.empty:
        return _message("Benchmark convergence", "No convergence history available", filepath, show, renderer)
    if problem is not None:
        df = df[df["problem"] == problem]
    if df.empty:
        return _message("Benchmark convergence", "No convergence history for selected problem", filepath, show, renderer)
    metric = metric or _default_metric(df)
    if metric not in df.columns or not df[metric].notna().any():
        metric = "best_fitness"
    fig = go.Figure()
    for i, (candidate, g) in enumerate(df.groupby("candidate")):
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
        color = _DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)]
        if show_iqr:
            fig.add_trace(go.Scatter(
                x=list(curve["evaluations"]) + list(curve["evaluations"])[::-1],
                y=list(curve["q75"]) + list(curve["q25"])[::-1],
                fill="toself", fillcolor=_hex_to_rgba(color, 0.18), line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False, name=f"{candidate} IQR",
            ))
        fig.add_trace(go.Scatter(
            x=curve["evaluations"], y=curve["median"], mode="lines", name=str(candidate),
            line=dict(color=color, width=2.4),
            hovertemplate="candidate=%{fullData.name}<br>evaluations=%{x}<br>median=%{y:.6g}<extra></extra>",
        ))
    if not fig.data:
        return _message("Benchmark convergence", "No plottable convergence records", filepath, show, renderer)
    _base_layout(fig, "Benchmark convergence")
    fig.update_xaxes(title_text="Evaluations")
    fig.update_yaxes(title_text=metric)
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_ecdf(
    result,
    target_tolerance: float | None = None,
    filepath=None,
    show: bool = False,
    renderer: str = "browser",
):
    """Plot ECDF of evaluations needed to reach the target. Returns a Plotly figure."""
    df = result.to_dataframe()
    if df.empty or "evaluations_to_target" not in df.columns:
        return _message("ECDF of target hits", "No target-hitting data available", filepath, show, renderer)
    if target_tolerance is not None and "target_tolerance" in df.columns:
        df = df[np.isclose(df["target_tolerance"].astype(float), float(target_tolerance))]
    budget_max = float(np.nanmax(df["budget"].fillna(df["evaluations"].max()).to_numpy(dtype=float))) if len(df) else 1.0
    xs = np.linspace(0.0, max(1.0, budget_max), 140)
    fig = go.Figure()
    for i, (candidate, g) in enumerate(df.groupby("candidate")):
        hits = g["evaluations_to_target"].dropna().to_numpy(dtype=float)
        if len(g) == 0:
            continue
        ys = [float(np.mean(hits <= x)) if len(hits) else 0.0 for x in xs]
        color = _DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(candidate), line=dict(color=color, width=2.4), hovertemplate="evaluations=%{x:.0f}<br>solved fraction=%{y:.3f}<extra></extra>"))
    if not fig.data:
        return _message("ECDF of target hits", "No target hits available", filepath, show, renderer)
    _base_layout(fig, "ECDF of target hits")
    fig.update_xaxes(title_text="Evaluations to target")
    fig.update_yaxes(title_text="Solved fraction", range=[0, 1.05])
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_performance_profile(
    result,
    metric: str | None = None,
    higher_is_better: bool | None = None,
    tau_max: float = 10.0,
    filepath=None,
    show: bool = False,
    renderer: str = "browser",
):
    """Plot a Dolan-Moré style performance profile. Returns a Plotly figure."""
    df = result.to_dataframe()
    if df.empty:
        return _message("Performance profile", "No benchmark records", filepath, show, renderer)
    metric = metric or _default_metric(df)
    if higher_is_better is None:
        higher_is_better = metric in {"success", "success_rate"}
    keys = ["problem", "dimension", "budget", "trial"]
    table = df.pivot_table(index=keys, columns="candidate", values=metric, aggfunc="mean").dropna(axis=0, how="any")
    if table.empty:
        return _message("Performance profile", "No paired records for performance profile", filepath, show, renderer)
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
    taus = np.linspace(1.0, tau_max, 180)
    fig = go.Figure()
    for j, candidate in enumerate(table.columns):
        r = ratios[:, j]
        y = [float(np.mean(r <= tau)) for tau in taus]
        color = _DISCRETE_PALETTE[j % len(_DISCRETE_PALETTE)]
        fig.add_trace(go.Scatter(x=taus, y=y, mode="lines", name=str(candidate), line=dict(color=color, width=2.4), hovertemplate="τ=%{x:.3f}<br>fraction=%{y:.3f}<extra></extra>"))
    _base_layout(fig, f"Performance profile: {metric}")
    fig.update_xaxes(title_text="Performance ratio τ")
    fig.update_yaxes(title_text="Fraction of instances", range=[0, 1.05])
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


def plot_rank_heatmap(
    result,
    metric: str | None = None,
    higher_is_better: bool | None = None,
    filepath=None,
    show: bool = False,
    renderer: str = "browser",
):
    """Plot median candidate ranks by problem. Returns a Plotly figure."""
    df = result.to_dataframe()
    if df.empty:
        return _message("Median ranks by problem", "No benchmark records", filepath, show, renderer)
    metric = metric or _default_metric(df)
    if higher_is_better is None:
        higher_is_better = metric in {"success", "success_rate"}
    keys = ["problem", "dimension", "budget", "trial"]
    table = df.pivot_table(index=keys, columns="candidate", values=metric, aggfunc="mean").dropna(axis=0, how="any")
    if table.empty:
        return _message("Median ranks by problem", "No paired records for rank heatmap", filepath, show, renderer)
    ranks = table.rank(axis=1, method="average", ascending=not higher_is_better).reset_index()
    ranks["problem_label"] = ranks.apply(lambda row: f"{row['problem']} D{int(row['dimension'])}", axis=1)
    heat = ranks.groupby("problem_label")[list(table.columns)].median()
    fig = go.Figure(go.Heatmap(
        z=heat.to_numpy(dtype=float), x=list(heat.columns), y=list(heat.index), colorscale=_COLORSCALE,
        colorbar=dict(title=dict(text="Median rank", font=dict(color=_TEXT_CLR)), tickfont=dict(color=_TEXT_CLR)),
        hovertemplate="problem=%{y}<br>candidate=%{x}<br>median rank=%{z:.3f}<extra></extra>",
    ))
    _base_layout(fig, f"Median ranks by problem: {metric}", width=max(850, 120 * len(heat.columns)), height=max(480, 70 * len(heat.index) + 180))
    fig.update_xaxes(title_text="Candidate")
    fig.update_yaxes(title_text="Problem")
    _save_plotly(fig, filepath)
    return _show(fig, show=show, renderer=renderer)


__all__ = ["plot_convergence", "plot_ecdf", "plot_performance_profile", "plot_rank_heatmap"]
