"""
pyMetaheuristic src — Plotly History Visualization Utilities
=============================================================

Plotly-based convergence, diversity, runtime, and explore/exploit charts for
``OptimizationResult`` objects.  All functions return a
``plotly.graph_objects.Figure`` and accept ``show``/``renderer`` for notebook,
Colab, browser, or static export workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .telemetry import convergence_data


# ── Shared style ─────────────────────────────────────────────────────────

_DARK_BG    = "#0d1117"
_PANEL_BG   = "#161b22"
_GRID_CLR   = "#21262d"
_TEXT_CLR   = "#e6edf3"
_ACCENT     = "#58a6ff"
_AMBER      = "#e3b341"
_GREEN      = "#3fb950"
_RED        = "#f85149"
_ORANGE     = "#d29922"

_DISCRETE_PALETTE = [
    "#3BC9DB", "#F0A500", "#3FB950", "#F85149",
    "#C084FC", "#FB923C", "#34D399", "#60A5FA",
    "#F472B6", "#A3E635", "#FBBF24", "#818CF8",
]

_FONT = dict(family="Inter, Arial, sans-serif", color=_TEXT_CLR, size=13)
_LAYOUT_BASE = dict(
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_PANEL_BG,
    font=_FONT,
    margin=dict(l=60, r=40, t=70, b=60),
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
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha):.3g})"


def _save_plotly(fig: go.Figure, filepath: str | Path | None, scale: int = 2) -> None:
    if filepath is None:
        return
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


def _show_plotly_if_needed(fig: go.Figure, show: bool = False, renderer: str = "browser") -> go.Figure:
    if show:
        if renderer:
            pio.renderers.default = renderer
        fig.show()
    return fig


def _base_layout(fig: go.Figure, title: str, width: int = 900, height: int = 520) -> go.Figure:
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=width,
        height=height,
    )
    return fig


def _message_figure(title: str, message: str, filepath=None, show: bool = False, renderer: str = "browser") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=message, showarrow=False,
                       font=dict(color=_TEXT_CLR, size=15))
    fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, x=0.04), xaxis=dict(visible=False), yaxis=dict(visible=False), width=900, height=520)
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


def _history(result) -> list[dict]:
    return list(getattr(result, "history", []) or [])


# ===========================================================================
# 1. Global-best convergence chart
# ===========================================================================

def plot_global_best_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    color: str = _ACCENT,
    x_axis: str = "steps",
    renderer: str = "browser",
) -> go.Figure:
    """Plot the global-best fitness curve. Returns a Plotly figure."""
    alg_id = getattr(result, "algorithm_id", "algorithm")
    x, y = convergence_data(result, x_axis=x_axis)
    xlabel = "Evaluations" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "Step"
    if len(x) == 0:
        return _message_figure(title or f"Global-Best Convergence — {alg_id}", "No history stored in result.", filepath, show, renderer)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y), mode="lines", name="Global best",
        line=dict(color=color, width=2.4, shape="hv" if xlabel == "Evaluations" else "linear"),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hovertemplate=f"{xlabel}=%{{x}}<br>fitness=%{{y:.6g}}<extra></extra>",
    ))
    _base_layout(fig, title or f"Global-Best Convergence — {alg_id}")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Fitness")
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


# ===========================================================================
# 2. Diversity chart
# ===========================================================================

def plot_diversity_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    color: str = _GREEN,
    renderer: str = "browser",
) -> go.Figure:
    """Plot population diversity over steps. Returns a Plotly figure."""
    history = _history(result)
    alg_id = getattr(result, "algorithm_id", "algorithm")
    rows = [(i + 1, h.get("diversity")) for i, h in enumerate(history) if h.get("diversity") is not None]
    if not rows:
        return _message_figure(title or f"Population Diversity — {alg_id}", "No diversity data in history. Run with store_history=True.", filepath, show, renderer)
    x, y = zip(*rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y), mode="lines", name="Diversity",
        line=dict(color=color, width=2.2), fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.22),
        hovertemplate="step=%{x}<br>diversity=%{y:.6g}<extra></extra>",
    ))
    _base_layout(fig, title or f"Population Diversity — {alg_id}")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Diversity")
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


# ===========================================================================
# 3. Exploration vs Exploitation chart
# ===========================================================================

def plot_explore_exploit_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    renderer: str = "browser",
) -> go.Figure:
    """Stacked Plotly area chart of exploration and exploitation fractions."""
    history = _history(result)
    alg_id = getattr(result, "algorithm_id", "algorithm")
    rows = []
    for i, h in enumerate(history):
        if "exploration" in h or "exploitation" in h:
            rows.append((i + 1, float(h.get("exploration", 1.0)), float(h.get("exploitation", 0.0))))
    if not rows:
        return _message_figure(title or f"Exploration vs Exploitation — {alg_id}", "No exploration/exploitation data. Run with store_history=True.", filepath, show, renderer)
    x = [r[0] for r in rows]
    explore = [r[1] for r in rows]
    exploit = [r[2] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=explore, mode="lines", stackgroup="one", name="Exploration", line=dict(color=_AMBER, width=1.5), hovertemplate="step=%{x}<br>exploration=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=exploit, mode="lines", stackgroup="one", name="Exploitation", line=dict(color=_ACCENT, width=1.5), hovertemplate="step=%{x}<br>exploitation=%{y:.3f}<extra></extra>"))
    _base_layout(fig, title or f"Exploration vs Exploitation — {alg_id}")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Fraction", range=[0, 1])
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


# ===========================================================================
# 4. Runtime-per-step chart
# ===========================================================================

def plot_runtime_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    color: str = _ORANGE,
    renderer: str = "browser",
) -> go.Figure:
    """Plot per-step runtime deltas using Plotly bars."""
    history = _history(result)
    alg_id = getattr(result, "algorithm_id", "algorithm")
    times = [h.get("elapsed_time") for h in history if h.get("elapsed_time") is not None]
    if not times:
        return _message_figure(title or f"Runtime per Step — {alg_id}", "No elapsed_time data in history. Run with store_history=True.", filepath, show, renderer)
    deltas = [float(times[0])] + [float(times[i]) - float(times[i - 1]) for i in range(1, len(times))]
    x = list(range(1, len(deltas) + 1))
    fig = go.Figure(go.Bar(x=x, y=deltas, name="Runtime", marker=dict(color=color, opacity=0.85), hovertemplate="step=%{x}<br>seconds=%{y:.6g}<extra></extra>"))
    _base_layout(fig, title or f"Runtime per Step — {alg_id}")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Seconds")
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


# ===========================================================================
# 5. All-in-one dashboard
# ===========================================================================

def plot_run_dashboard(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    renderer: str = "browser",
) -> go.Figure:
    """2×2 Plotly dashboard: convergence, diversity, explore/exploit, runtime."""
    history = _history(result)
    alg_id = getattr(result, "algorithm_id", "algorithm")
    dash_title = title or f"Run Dashboard — {alg_id}"
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Global-Best Convergence", "Population Diversity", "Exploration vs Exploitation", "Runtime per Step"),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    x_conv, y_conv = convergence_data(result, x_axis="steps")
    if len(x_conv) > 0:
        fig.add_trace(go.Scatter(x=list(x_conv), y=list(y_conv), mode="lines", name="Global best", line=dict(color=_ACCENT, width=2)), row=1, col=1)
    divs = [(i + 1, h.get("diversity")) for i, h in enumerate(history) if h.get("diversity") is not None]
    if divs:
        x, y = zip(*divs)
        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name="Diversity", fill="tozeroy", fillcolor=_hex_to_rgba(_GREEN, 0.20), line=dict(color=_GREEN, width=2)), row=1, col=2)
    rows = [(i + 1, float(h.get("exploration", 1.0)), float(h.get("exploitation", 0.0))) for i, h in enumerate(history) if "exploration" in h or "exploitation" in h]
    if rows:
        x = [r[0] for r in rows]
        fig.add_trace(go.Scatter(x=x, y=[r[1] for r in rows], mode="lines", stackgroup="one", name="Exploration", line=dict(color=_AMBER, width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=[r[2] for r in rows], mode="lines", stackgroup="one", name="Exploitation", line=dict(color=_ACCENT, width=1.5)), row=2, col=1)
    times = [h.get("elapsed_time") for h in history if h.get("elapsed_time") is not None]
    if times:
        deltas = [float(times[0])] + [float(times[i]) - float(times[i - 1]) for i in range(1, len(times))]
        fig.add_trace(go.Bar(x=list(range(1, len(deltas) + 1)), y=deltas, name="Runtime", marker=dict(color=_ORANGE, opacity=0.85)), row=2, col=2)
    fig.update_layout(**_LAYOUT_BASE, title=dict(text=dash_title, x=0.04, font=dict(color=_TEXT_CLR, size=17)), width=1100, height=760, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)))
    for axis_name in fig.layout:
        if str(axis_name).startswith("xaxis"):
            fig.layout[axis_name].update(**_AXIS_STYLE, title_text="Step")
        elif str(axis_name).startswith("yaxis"):
            fig.layout[axis_name].update(**_AXIS_STYLE)
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


# ===========================================================================
# 6. Multi-algorithm diversity comparison
# ===========================================================================

def plot_diversity_comparison(
    results: dict[str, Any],
    filepath: str | Path | None = None,
    title: str = "Diversity Comparison",
    show: bool = False,
    renderer: str = "browser",
) -> go.Figure:
    """Overlay diversity curves from multiple runs / algorithms."""
    if not results:
        return _message_figure(title, "No results provided.", filepath, show, renderer)
    fig = go.Figure()
    for i, (label, result) in enumerate(results.items()):
        history = _history(result)
        rows = [(j + 1, h.get("diversity")) for j, h in enumerate(history) if h.get("diversity") is not None]
        if not rows:
            continue
        x, y = zip(*rows)
        color = _DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)]
        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=str(label), line=dict(color=color, width=2.2), hovertemplate="step=%{x}<br>diversity=%{y:.6g}<extra></extra>"))
    if not fig.data:
        return _message_figure(title, "No diversity data available. Run with store_history=True.", filepath, show, renderer)
    _base_layout(fig, title)
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Diversity")
    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


__all__ = [
    "plot_global_best_chart",
    "plot_diversity_chart",
    "plot_explore_exploit_chart",
    "plot_runtime_chart",
    "plot_run_dashboard",
    "plot_diversity_comparison",
]
