############################################################################
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Graphs — unified, publication-quality function-landscape and
#           optimisation-history visualisations.
#
# Supported dimensionalities
#   1D  : line plot   (x vs f(x))
#   2D  : contour + filled-heatmap
#   3D  : interactive surface + projected contour base
#   4D+ : parallel-coordinates (true ND) + optional PCA projection
#
# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic
############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .telemetry import convergence_data


############################################################################
# Palette & style constants
############################################################################

_DARK_BG   = "#0d1117"
_PANEL_BG  = "#161b22"
_GRID_CLR  = "#21262d"
_TEXT_CLR  = "#e6edf3"
_ACCENT    = "#58a6ff"
_SOL_CLR   = "#f78166"

_COLORSCALE = [
    [0.00, "#0d1b4b"],
    [0.20, "#0c4a6e"],
    [0.40, "#0e7490"],
    [0.55, "#059669"],
    [0.70, "#d97706"],
    [0.85, "#dc2626"],
    [1.00, "#7f1d1d"],
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

_MPL_STYLE = {
    "figure.facecolor":  _DARK_BG,
    "axes.facecolor":    _PANEL_BG,
    "axes.edgecolor":    _GRID_CLR,
    "axes.labelcolor":   _TEXT_CLR,
    "axes.grid":         True,
    "grid.color":        _GRID_CLR,
    "grid.linewidth":    0.6,
    "text.color":        _TEXT_CLR,
    "xtick.color":       _TEXT_CLR,
    "ytick.color":       _TEXT_CLR,
    "lines.linewidth":   2.2,
    "savefig.facecolor": _DARK_BG,
}

_MPL_CMAP = LinearSegmentedColormap.from_list(
    "pym", ["#0d1b4b", "#0e7490", "#059669", "#d97706", "#dc2626", "#7f1d1d"]
)

# Discrete palette for benchmark bar/box/convergence charts
_DISCRETE_PALETTE = [
    "#3BC9DB", "#F0A500", "#3FB950", "#F85149",
    "#C084FC", "#FB923C", "#34D399", "#60A5FA",
    "#F472B6", "#A3E635", "#FBBF24", "#818CF8",
]


__all__ = [
    "plot_function",
    "plot_function_1d",
    "plot_function_2d",
    "plot_function_3d",
    "plot_function_nd",
    "plot_convergence",
    "compare_convergence",
    "plot_population_snapshot",
    "plot_benchmark_summary",
    "plot_benchmark_barplots",
    "plot_benchmark_boxplots",
    "plot_benchmark_rank_heatmap",
    "plot_benchmark_runtime",
    "plot_benchmark_convergence",
    "plot_function_contour",
    "plot_function_surface",
    "plot_island_dynamics",
    "plot_collaboration_network",
]


############################################################################
# Internal helpers
############################################################################

def _as_path(filepath) -> Optional[Path]:
    return None if filepath is None else Path(filepath)


def _save_plotly(fig: go.Figure, filepath, scale: int = 2):
    p = _as_path(filepath)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext in (".html", ""):
        fig.write_html(str(p))
    elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"):
        fig.write_image(str(p), scale=scale)
    else:
        fig.write_html(str(p.with_suffix(".html")))


def _maybe_save_mpl(fig, filepath, dpi: int = 160):
    p = _as_path(filepath)
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    return fig


def _history_series(result, key: str = "best_fitness"):
    history = list(getattr(result, "history", []) or [])
    if not history:
        return np.array([]), np.array([])
    x = np.array([row.get("step", i + 1) for i, row in enumerate(history)], dtype=float)
    y = np.array([row.get(key, np.nan) for row in history], dtype=float)
    return x, y


def _grid_2d(min_vals, max_vals, n: int = 250):
    x = np.linspace(float(min_vals[0]), float(max_vals[0]), n)
    y = np.linspace(float(min_vals[1]), float(max_vals[1]), n)
    return np.meshgrid(x, y)


def _eval_surface(fn: Callable, X, Y) -> np.ndarray:
    Z = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = float(fn([float(X[i, j]), float(Y[i, j])]))
    return Z


def _pca_2d(data: np.ndarray) -> np.ndarray:
    if data.shape[1] <= 2:
        return data[:, :2]
    mu = data.mean(axis=0)
    C = np.cov((data - mu).T)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    return (data - mu) @ vecs[:, idx[:2]]


def _sol_array(solutions) -> Optional[np.ndarray]:
    if solutions is None:
        return None
    arr = np.atleast_2d(np.array(solutions, dtype=float))
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def _apply_dark_mpl():
    plt.rcParams.update(_MPL_STYLE)



def _empty_plotly_figure(
    title: str,
    message: str,
    filepath=None,
    show: bool = False,
    renderer: str = "browser",
) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font=dict(color=_TEXT_CLR, size=16),
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=900, height=520,
    )
    _save_plotly(fig, filepath)
    if show:
        pio.renderers.default = renderer
        fig.show()
    return fig


def _resolve_snapshot(result, snapshot: Union[str, int] = "last"):
    snapshots = list(getattr(result, "population_snapshots", []) or [])
    if not snapshots:
        return snapshots, None
    if snapshot == "last":
        idx = -1
    else:
        idx = int(snapshot)
    snap = snapshots[idx]
    return snapshots, snap


def _snapshot_arrays(snap) -> tuple[np.ndarray, np.ndarray]:
    population = list((snap or {}).get("population", []) or [])
    if not population:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float)

    positions = np.array([p.get("position", []) for p in population], dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    fitness = np.array([p.get("fitness", np.nan) for p in population], dtype=float)
    return positions, fitness


def _normalise_dims(ndim: int, dims=None, max_dims: Optional[int] = None) -> tuple[int, ...]:
    if ndim <= 0:
        return tuple()

    target = ndim if max_dims is None else min(ndim, max_dims)

    if dims is None:
        return tuple(range(target))

    if isinstance(dims, int):
        dims = (dims,)

    cleaned = []
    used = set()
    for d in dims:
        try:
            d = int(d)
        except (TypeError, ValueError):
            continue
        if -ndim <= d < ndim:
            d = d % ndim
            if d not in used:
                cleaned.append(d)
                used.add(d)
        if len(cleaned) >= target:
            break

    if len(cleaned) < target:
        for d in range(ndim):
            if d not in used:
                cleaned.append(d)
                used.add(d)
            if len(cleaned) >= target:
                break

    return tuple(cleaned[:target])


def _result_best_array(result) -> Optional[np.ndarray]:
    best = getattr(result, "best_position", None)
    if best is None:
        return None
    arr = np.asarray(best, dtype=float).ravel()
    return arr if arr.size > 0 else None


def _snapshot_best_index(fitness: np.ndarray, result=None) -> Optional[int]:
    if fitness.size == 0:
        return None
    finite = np.isfinite(fitness)
    if not np.any(finite):
        return None

    objective = str(getattr(result, "metadata", {}).get("objective", "min")).lower()
    fit = fitness.astype(float).copy()
    if objective == "max":
        fit[~finite] = -np.inf
        return int(np.argmax(fit))

    fit[~finite] = np.inf
    return int(np.argmin(fit))


def _show_plotly_if_needed(fig: go.Figure, show: bool = False, renderer: str = "browser") -> go.Figure:
    if show:
        pio.renderers.default = renderer
        fig.show()
    return fig


############################################################################
# 1-D
############################################################################



def _resolve_problem_inputs(target_function, min_values, max_values):
    if hasattr(target_function, "evaluate") and hasattr(target_function, "lower") and hasattr(target_function, "upper"):
        problem = target_function
        fn = problem if callable(problem) else problem.evaluate
        title_extra = getattr(problem, "name", type(problem).__name__)
        return fn, list(problem.lower), list(problem.upper), problem, title_extra
    return target_function, min_values, max_values, None, None

def plot_function_1d(
    target_function: Callable,
    min_values: Sequence[float],
    max_values: Sequence[float],
    solutions=None,
    title: str = "Function Landscape - 1D",
    grid_points: int = 600,
    filepath=None,
    renderer: str = "browser",
) -> go.Figure:
    """Interactive 1-D line plot with gradient fill and optional solution overlay."""
    x  = np.linspace(float(min_values[0]), float(max_values[0]), grid_points)
    fx = np.array([float(target_function([xi])) for xi in x])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=fx,
        mode="lines",
        line=dict(color=_ACCENT, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.12)",
        hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
        name="f(x)",
    ))

    sol = _sol_array(solutions)
    if sol is not None:
        sx  = sol[:, 0]
        sfx = np.array([float(target_function([xi])) for xi in sx])
        fig.add_trace(go.Scatter(
            x=sx, y=sfx,
            mode="markers",
            marker=dict(symbol="circle", size=14, color=_SOL_CLR,
                        line=dict(width=2, color="white")),
            hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
            name="Solution",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(**_AXIS_STYLE, title="x1"),
        yaxis=dict(**_AXIS_STYLE, title="f(x)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
    )

    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig


############################################################################
# 2-D
############################################################################

def plot_function_2d(
    target_function: Callable,
    min_values: Sequence[float],
    max_values: Sequence[float],
    solutions=None,
    title: str = "Function Landscape - 2D",
    grid_points: int = 250,
    contour_levels: int = 40,
    filepath=None,
    renderer: str = "browser",
) -> go.Figure:
    """Filled contour + heatmap side-by-side with optional solution overlay."""
    X, Y = _grid_2d(min_values, max_values, grid_points)
    Z    = _eval_surface(target_function, X, Y)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Contour Map", "Heatmap"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Contour(
        x=X[0], y=Y[:, 0], z=Z,
        colorscale=_COLORSCALE,
        ncontours=contour_levels,
        contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
        colorbar=dict(x=0.44, thickness=12, tickfont=dict(color=_TEXT_CLR)),
        hovertemplate="x1=%{x:.3f}  x2=%{y:.3f}  f=%{z:.4f}<extra></extra>",
        name="f",
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=X[0], y=Y[:, 0], z=Z,
        colorscale=_COLORSCALE,
        colorbar=dict(x=1.01, thickness=12, tickfont=dict(color=_TEXT_CLR)),
        hovertemplate="x1=%{x:.3f}  x2=%{y:.3f}  f=%{z:.4f}<extra></extra>",
        name="f",
    ), row=1, col=2)

    sol = _sol_array(solutions)
    if sol is not None and sol.shape[1] >= 2:
        sx  = sol[:, 0]
        sy  = sol[:, 1]
        sfz = np.array([float(target_function(s.tolist())) for s in sol])
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=sx, y=sy,
                mode="markers",
                marker=dict(symbol="star", size=16, color=_SOL_CLR,
                            line=dict(width=1.5, color="white")),
                hovertemplate="x1=%{x:.4f}<br>x2=%{y:.4f}<br>f=%{customdata:.4f}<extra></extra>",
                customdata=sfz,
                name="Solution",
                showlegend=(col == 1),
            ), row=1, col=col)

    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(**_AXIS_STYLE)
    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT_CLR, size=13)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=1100, height=500,
    )

    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig


############################################################################
# 3-D
############################################################################

def plot_function_3d(
    target_function: Callable,
    min_values: Sequence[float],
    max_values: Sequence[float],
    solutions=None,
    title: str = "Function Landscape - 3D",
    grid_points: int = 120,
    filepath=None,
    renderer: str = "browser",
) -> go.Figure:
    """Shaded 3-D surface with projected contour base and solution spike."""
    X, Y = _grid_2d(min_values, max_values, grid_points)
    Z    = _eval_surface(target_function, X, Y)
    zmin = float(Z.min())

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=_COLORSCALE,
        lighting=dict(ambient=0.6, diffuse=0.85, specular=0.25,
                      roughness=0.5, fresnel=0.2),
        lightposition=dict(x=1, y=1, z=2),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white",
                   project_z=True, width=1),
        ),
        colorbar=dict(thickness=14, tickfont=dict(color=_TEXT_CLR, size=11),
                      outlinewidth=0, bgcolor="rgba(0,0,0,0)"),
        hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>f=%{z:.4f}<extra></extra>",
        name="f(x1,x2)",
        opacity=0.92,
    ))

    sol = _sol_array(solutions)
    if sol is not None and sol.shape[1] >= 2:
        sx  = sol[:, 0]
        sy  = sol[:, 1]
        sfz = np.array([float(target_function(s.tolist())) for s in sol])
        spike_x, spike_y, spike_z = [], [], []
        for xi, yi, zi in zip(sx, sy, sfz):
            spike_x += [xi, xi, None]
            spike_y += [yi, yi, None]
            spike_z += [zmin, zi, None]
        fig.add_trace(go.Scatter3d(
            x=spike_x, y=spike_y, z=spike_z,
            mode="lines",
            line=dict(color=_SOL_CLR, width=4, dash="dot"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sfz,
            mode="markers",
            marker=dict(size=10, color=_SOL_CLR, symbol="diamond",
                        line=dict(width=2, color="white")),
            hovertemplate="x1=%{x:.4f}<br>x2=%{y:.4f}<br>f=%{z:.4f}<extra></extra>",
            name="Solution",
        ))

    scene = dict(
        bgcolor=_DARK_BG,
        xaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR,
                   showbackground=True, tickfont=dict(color=_TEXT_CLR),
                   title=dict(text="x1", font=dict(color=_TEXT_CLR))),
        yaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR,
                   showbackground=True, tickfont=dict(color=_TEXT_CLR),
                   title=dict(text="x2", font=dict(color=_TEXT_CLR))),
        zaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR,
                   showbackground=True, tickfont=dict(color=_TEXT_CLR),
                   title=dict(text="f(x)", font=dict(color=_TEXT_CLR))),
        camera=dict(eye=dict(x=1.55, y=1.55, z=0.85)),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        scene=scene,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=900, height=650,
    )

    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig


############################################################################
# N-D (>=4)
############################################################################

def plot_function_nd(
    target_function: Callable,
    min_values: Sequence[float],
    max_values: Sequence[float],
    solutions=None,
    title: str = "Function Landscape - ND",
    n_samples: int = 4000,
    show_pca: bool = True,
    filepath=None,
    renderer: str = "browser",
) -> go.Figure:
    """Parallel-coordinates + optional 2-D PCA scatter for >=4 variables."""
    ndim = len(min_values)
    rng  = np.random.default_rng(42)
    pts  = rng.uniform(
        low=np.array(min_values, dtype=float),
        high=np.array(max_values, dtype=float),
        size=(n_samples, ndim),
    )
    fvals = np.array([float(target_function(p.tolist())) for p in pts])

    dims = [
        dict(range=[float(min_values[i]), float(max_values[i])],
             label=f"x{i+1}", values=pts[:, i].tolist())
        for i in range(ndim)
    ]
    dims.append(dict(range=[float(fvals.min()), float(fvals.max())],
                     label="f(x)", values=fvals.tolist()))

    rows   = 1 + int(show_pca)
    specs  = [[{"type": "parcoords"}]] + ([[{"type": "scatter"}]] if show_pca else [])
    titles = ["Parallel-Coordinates (Colour = f(x))"]
    if show_pca:
        titles.append(f"PCA Projection (2 of {ndim} Dims)")

    fig = make_subplots(
        rows=rows, cols=1,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.14,
        row_heights=[0.52, 0.48] if show_pca else [1.0],
    )

    fig.add_trace(go.Parcoords(
        line=dict(color=fvals, colorscale=_COLORSCALE, showscale=True,
                  colorbar=dict(thickness=14, tickfont=dict(color=_TEXT_CLR),
                                title=dict(text="f(x)", font=dict(color=_TEXT_CLR)))),
        dimensions=dims,
        labelfont=dict(color=_TEXT_CLR, size=12),
        tickfont=dict(color=_TEXT_CLR, size=10),
        rangefont=dict(color=_TEXT_CLR, size=10),
    ), row=1, col=1)

    if show_pca:
        pca2 = _pca_2d(pts)
        fig.add_trace(go.Scatter(
            x=pca2[:, 0], y=pca2[:, 1],
            mode="markers",
            marker=dict(color=fvals, colorscale=_COLORSCALE, size=5,
                        opacity=0.75, showscale=False, line=dict(width=0)),
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
            name="Samples",
        ), row=2, col=1)

        sol = _sol_array(solutions)
        if sol is not None:
            all_pts = np.vstack([pts, sol])
            all_pca = _pca_2d(all_pts)
            sol_pca = all_pca[n_samples:]
            sfz     = np.array([float(target_function(s.tolist())) for s in sol])
            fig.add_trace(go.Scatter(
                x=sol_pca[:, 0], y=sol_pca[:, 1],
                mode="markers",
                marker=dict(symbol="star", size=18, color=_SOL_CLR,
                            line=dict(width=1.5, color="white")),
                hovertemplate="PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>f=%{customdata:.4f}<extra></extra>",
                customdata=sfz,
                name="Solution",
            ), row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font = dict(color=_TEXT_CLR, size=13)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(**_AXIS_STYLE)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
        width=1050, height=780 if show_pca else 440,
    )

    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig


############################################################################
# Unified dispatcher
############################################################################

def plot_function(
    target_function: Callable,
    min_values: Sequence[float] | None = None,
    max_values: Sequence[float] | None = None,
    solutions=None,
    title: Optional[str] = None,
    grid_points: int = 200,
    filepath=None,
    renderer: str = "browser",
    **kwargs,
) -> go.Figure:
    """Auto-select the best visualisation for any dimensionality.

    Parameters
    ----------
    target_function : callable
        f(variables_values) -> float  (same signature as built-in test functions).
    min_values, max_values : sequence of float
        Bounds per dimension.
    solutions : array-like, optional
        One or more known solutions as rows  (n_solutions, n_dims).
    title : str, optional
        Plot title (auto-generated when omitted).
    grid_points : int
        Grid resolution for surface / contour grids.
    filepath : str or Path, optional
        Save to file when provided  (HTML, PNG, SVG, PDF, …).
    renderer : str
        Plotly renderer for interactive display (``'browser'``,
        ``'notebook'``, ``'png'``, …).
    **kwargs
        Forwarded to the specific plotting function.
    """
    ndim = len(min_values)
    _t   = title or f"f(x) - {ndim}-D landscape"

    if ndim == 1:
        return plot_function_1d(
            target_function, min_values, max_values,
            solutions=solutions, title=_t, grid_points=grid_points,
            filepath=filepath, renderer=renderer, **kwargs,
        )
    elif ndim == 2:
        return plot_function_2d(
            target_function, min_values, max_values,
            solutions=solutions, title=_t, grid_points=grid_points,
            filepath=filepath, renderer=renderer, **kwargs,
        )
    elif ndim == 3:
        return plot_function_3d(
            target_function, min_values, max_values,
            solutions=solutions, title=_t,
            grid_points=min(grid_points, 120),
            filepath=filepath, renderer=renderer, **kwargs,
        )
    else:
        return plot_function_nd(
            target_function, min_values, max_values,
            solutions=solutions, title=_t,
            filepath=filepath, renderer=renderer, **kwargs,
        )


############################################################################
# History / benchmark  (matplotlib, kept for backward-compat)
############################################################################

def plot_convergence(result, filepath=None, title=None, show=False, x_axis: str = "steps"):
    """Convergence curve for a single optimisation run."""
    _apply_dark_mpl()
    x, y = convergence_data(result, x_axis=x_axis)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(x) == 0:
        ax.text(0.5, 0.5, "No history stored in result",
                ha="center", va="center", color=_TEXT_CLR)
        ax.set_axis_off()
    else:
        draw_style = "steps-post" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "default"
        ax.plot(x, y, color=_ACCENT, linewidth=2.2, drawstyle=draw_style)
        ax.fill_between(x, y, min(y), alpha=0.10, color=_ACCENT, step="post" if draw_style == "steps-post" else None)
        ax.set_xlabel("Evaluations" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "Step")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, color=_GRID_CLR, linewidth=0.6)
    algo = getattr(result, "algorithm_id", "run")
    ax.set_title(title or f"Convergence - {algo}", pad=12)
    fig.tight_layout()
    _maybe_save_mpl(fig, filepath)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compare_convergence(
    results: Iterable,
    labels=None,
    filepath=None,
    title="Convergence comparison",
    show=False,
    x_axis: str = "steps",
):
    """Overlay convergence curves for multiple runs."""
    _apply_dark_mpl()
    results = list(results)
    if labels is None:
        labels = [getattr(r, "algorithm_id", f"run_{i+1}") for i, r in enumerate(results)]
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(results), 1)))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for result, label, colour in zip(results, labels, palette):
        x, y = convergence_data(result, x_axis=x_axis)
        if len(x) > 0:
            style = "steps-post" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "default"
            ax.plot(x, y, linewidth=2.2, label=label, color=colour, drawstyle=style)
    ax.set_xlabel("Evaluations" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "Step")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title, pad=12)
    ax.grid(True, color=_GRID_CLR, linewidth=0.6)
    if results:
        ax.legend(frameon=False)
    fig.tight_layout()
    _maybe_save_mpl(fig, filepath)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_population_snapshot(
    result,
    snapshot: Union[str, int] = "last",
    dims=None,
    filepath=None,
    title="Population Snapshot",
    show=False,
    backend: str = "plotly",
    renderer: str = "browser",
    marker_size: int = 9,
    opacity: float = 0.85,
):
    """Adaptive population visualisation.

    Behaviour
    ---------
    * 1 variable  -> Plotly 1-D scatter  (x vs fitness)
    * 2 variables -> Plotly 2-D scatter
    * 3 variables -> Plotly 3-D scatter
    * 4+ vars     -> Plotly parallel coordinates

    Notes
    -----
    * ``dims`` is honoured for 2-D / 3-D projections.
    * For 4+ variables the full decision vector is shown automatically.
    * Set ``backend='matplotlib'`` to keep the legacy 2-D scatter behaviour.
    """
    snapshots, snap = _resolve_snapshot(result, snapshot)

    if backend.lower() == "matplotlib":
        _apply_dark_mpl()
        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        if not snapshots or snap is None:
            ax.text(0.5, 0.5, "No snapshots stored", ha="center", va="center")
            ax.set_axis_off()
        else:
            positions, fitness = _snapshot_arrays(snap)
            if positions.size == 0:
                ax.text(0.5, 0.5, "Empty snapshot", ha="center", va="center")
                ax.set_axis_off()
            elif positions.shape[1] < 2:
                ax.scatter(positions[:, 0], fitness, c=fitness, cmap=_MPL_CMAP, s=45, edgecolors="none", alpha=0.85)
                ax.set_xlabel("x1")
                ax.set_ylabel("Fitness")
                ax.set_title(f"{title}  (step={snap.get('step', '?')})", pad=12)
            else:
                i, j = _normalise_dims(positions.shape[1], dims=dims, max_dims=2)
                sc = ax.scatter(positions[:, i], positions[:, j], c=fitness, cmap=_MPL_CMAP, s=45, edgecolors="none", alpha=0.85)
                cb = fig.colorbar(sc, ax=ax, shrink=0.85)
                cb.set_label("Fitness")
                ax.set_xlabel(f"x{i+1}")
                ax.set_ylabel(f"x{j+1}")
                ax.set_title(f"{title}  (step={snap.get('step', '?')})", pad=12)
        fig.tight_layout()
        _maybe_save_mpl(fig, filepath)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    if not snapshots or snap is None:
        return _empty_plotly_figure(title, "No snapshots stored", filepath=filepath, show=show, renderer=renderer)

    positions, fitness = _snapshot_arrays(snap)
    if positions.size == 0:
        return _empty_plotly_figure(title, "Empty snapshot", filepath=filepath, show=show, renderer=renderer)

    ndim = int(positions.shape[1])
    step = snap.get("step", "?")
    best_position = _result_best_array(result)
    best_fitness = getattr(result, "best_fitness", None)
    best_idx = _snapshot_best_index(fitness, result=result)

    if ndim == 1:
        fig = go.Figure()
        x = positions[:, 0]
        fig.add_trace(go.Scatter(
            x=x, y=fitness,
            mode="markers",
            marker=dict(
                size=marker_size, color=fitness, colorscale=_COLORSCALE,
                opacity=opacity, showscale=True,
                colorbar=dict(title="Fitness", tickfont=dict(color=_TEXT_CLR)),
                line=dict(width=0),
            ),
            customdata=np.arange(len(x)),
            hovertemplate="agent=%{customdata}<br>x1=%{x:.4f}<br>fitness=%{y:.6g}<extra></extra>",
            name="Population",
        ))

        if best_position is not None and best_position.size >= 1 and best_fitness is not None:
            fig.add_trace(go.Scatter(
                x=[best_position[0]], y=[best_fitness],
                mode="markers",
                marker=dict(symbol="star", size=max(marker_size + 7, 14), color=_SOL_CLR, line=dict(width=1.5, color="white")),
                hovertemplate="best x1=%{x:.4f}<br>best fitness=%{y:.6g}<extra></extra>",
                name="Best known",
            ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(text=f"{title} - 1D  (step={step})", font=dict(color=_TEXT_CLR, size=16), x=0.04),
            xaxis=dict(**_AXIS_STYLE, title="x1"),
            yaxis=dict(**_AXIS_STYLE, title="Fitness"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
            width=900, height=520,
        )
        _save_plotly(fig, filepath)
        return _show_plotly_if_needed(fig, show=show, renderer=renderer)

    if ndim == 2:
        i, j = _normalise_dims(ndim, dims=dims, max_dims=2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=positions[:, i], y=positions[:, j],
            mode="markers",
            marker=dict(
                size=marker_size, color=fitness, colorscale=_COLORSCALE, opacity=opacity,
                showscale=True, colorbar=dict(title="Fitness", tickfont=dict(color=_TEXT_CLR)),
                line=dict(width=0),
            ),
            customdata=np.column_stack([fitness, np.arange(len(fitness))]),
            hovertemplate=(
                f"agent=%{{customdata[1]}}<br>x{i+1}=%{{x:.4f}}<br>x{j+1}=%{{y:.4f}}"
                "<br>fitness=%{customdata[0]:.6g}<extra></extra>"
            ),
            name="Population",
        ))

        if best_position is not None and best_position.size >= ndim and best_fitness is not None:
            fig.add_trace(go.Scatter(
                x=[best_position[i]], y=[best_position[j]],
                mode="markers",
                marker=dict(symbol="star", size=max(marker_size + 8, 15), color=_SOL_CLR, line=dict(width=1.5, color="white")),
                hovertemplate=(
                    f"best x{i+1}=%{{x:.4f}}<br>best x{j+1}=%{{y:.4f}}"
                    f"<br>best fitness={float(best_fitness):.6g}<extra></extra>"
                ),
                name="Best known",
            ))
        elif best_idx is not None:
            fig.add_trace(go.Scatter(
                x=[positions[best_idx, i]], y=[positions[best_idx, j]],
                mode="markers",
                marker=dict(symbol="star", size=max(marker_size + 8, 15), color=_SOL_CLR, line=dict(width=1.5, color="white")),
                hovertemplate="Best in snapshot<extra></extra>",
                name="Best in snapshot",
            ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(text=f"{title} - 2D  (step={step})", font=dict(color=_TEXT_CLR, size=16), x=0.04),
            xaxis=dict(**_AXIS_STYLE, title=f"x{i+1}"),
            yaxis=dict(**_AXIS_STYLE, title=f"x{j+1}"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
            width=900, height=620,
        )
        _save_plotly(fig, filepath)
        return _show_plotly_if_needed(fig, show=show, renderer=renderer)

    if ndim == 3:
        i, j, k = _normalise_dims(ndim, dims=dims, max_dims=3)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=positions[:, i], y=positions[:, j], z=positions[:, k],
            mode="markers",
            marker=dict(
                size=max(marker_size - 1, 4), color=fitness, colorscale=_COLORSCALE, opacity=opacity,
                showscale=True, colorbar=dict(title="Fitness", tickfont=dict(color=_TEXT_CLR)),
                line=dict(width=0),
            ),
            customdata=np.column_stack([fitness, np.arange(len(fitness))]),
            hovertemplate=(
                f"agent=%{{customdata[1]}}<br>x{i+1}=%{{x:.4f}}<br>x{j+1}=%{{y:.4f}}<br>x{k+1}=%{{z:.4f}}"
                "<br>fitness=%{customdata[0]:.6g}<extra></extra>"
            ),
            name="Population",
        ))

        if best_position is not None and best_position.size >= ndim and best_fitness is not None:
            fig.add_trace(go.Scatter3d(
                x=[best_position[i]], y=[best_position[j]], z=[best_position[k]],
                mode="markers",
                marker=dict(symbol="diamond", size=max(marker_size + 2, 8), color=_SOL_CLR, line=dict(width=2, color="white")),
                hovertemplate=(
                    f"best x{i+1}=%{{x:.4f}}<br>best x{j+1}=%{{y:.4f}}<br>best x{k+1}=%{{z:.4f}}"
                    f"<br>best fitness={float(best_fitness):.6g}<extra></extra>"
                ),
                name="Best known",
            ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(text=f"{title} - 3D  (step={step})", font=dict(color=_TEXT_CLR, size=16), x=0.04),
            scene=dict(
                bgcolor=_DARK_BG,
                xaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR, showbackground=True, tickfont=dict(color=_TEXT_CLR), title=dict(text=f"x{i+1}", font=dict(color=_TEXT_CLR))),
                yaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR, showbackground=True, tickfont=dict(color=_TEXT_CLR), title=dict(text=f"x{j+1}", font=dict(color=_TEXT_CLR))),
                zaxis=dict(backgroundcolor=_PANEL_BG, gridcolor=_GRID_CLR, showbackground=True, tickfont=dict(color=_TEXT_CLR), title=dict(text=f"x{k+1}", font=dict(color=_TEXT_CLR))),
                camera=dict(eye=dict(x=1.55, y=1.45, z=0.85)),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
            width=920, height=700,
        )
        _save_plotly(fig, filepath)
        return _show_plotly_if_needed(fig, show=show, renderer=renderer)

    dims_pc = [
        dict(
            range=[float(np.nanmin(positions[:, idx])), float(np.nanmax(positions[:, idx]))],
            label=f"x{idx+1}",
            values=positions[:, idx].tolist(),
        )
        for idx in range(ndim)
    ]
    dims_pc.append(dict(
        range=[float(np.nanmin(fitness)), float(np.nanmax(fitness))],
        label="fitness",
        values=fitness.tolist(),
    ))

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=fitness, colorscale=_COLORSCALE, showscale=True,
            colorbar=dict(title="Fitness", tickfont=dict(color=_TEXT_CLR)),
        ),
        dimensions=dims_pc,
        labelfont=dict(color=_TEXT_CLR, size=12),
        tickfont=dict(color=_TEXT_CLR, size=10),
        rangefont=dict(color=_TEXT_CLR, size=10),
    ))

    layout_nd = dict(_LAYOUT_BASE)
    layout_nd["margin"] = dict(l=80, r=60, t=80, b=40)
    fig.update_layout(
        **layout_nd,
        title=dict(text=f"{title} - {ndim}D  (step={step})", font=dict(color=_TEXT_CLR, size=16), x=0.04),
        width=max(900, 160 * (ndim + 1)), height=520,
    )

    _save_plotly(fig, filepath)
    return _show_plotly_if_needed(fig, show=show, renderer=renderer)


def plot_benchmark_summary(
    results: dict,
    metric: str = "best_fitness",
    filepath=None,
    title: str = "Benchmark Summary",
    show: bool = False,
):
    """Bar chart comparing a scalar metric across algorithms."""
    if not results:
        raise ValueError("results must be a non-empty dict of label -> OptimizationResult")
    _apply_dark_mpl()
    labels  = list(results.keys())
    values  = [getattr(results[k], metric) for k in labels]
    palette = plt.cm.plasma(np.linspace(0.2, 0.8, len(labels)))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.95), 4.8))
    ax.bar(labels, values, color=palette, edgecolor="none", width=0.55)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, pad=12)
    ax.tick_params(axis="x", rotation=40)
    ax.grid(True, axis="y", color=_GRID_CLR, linewidth=0.6)
    fig.tight_layout()
    _maybe_save_mpl(fig, filepath)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


############################################################################
# BenchmarkRunner graphs — Plotly (DataFrame-based)
############################################################################

def _bench_filepath(filepath, problem_name: str, suffix: str) -> Optional[Path]:
    """Derive a per-problem save path from a base filepath."""
    if filepath is None:
        return None
    p = Path(str(filepath))
    slug = problem_name.lower().replace(" ", "_").replace("/", "_")
    return p.parent / f"{p.stem}_{suffix}_{slug}{p.suffix or '.html'}"


def plot_benchmark_barplots(
    summary_df: "pd.DataFrame",
    filepath=None,
    renderer: str = "browser",
    show: bool = False,
) -> "dict[str, go.Figure]":
    """Bar chart of mean best-fitness per algorithm, one figure per problem.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``BenchmarkRunner.summary()``.  Must contain columns
        ``problem``, ``algorithm``, ``mean_fitness``.
    filepath : str or Path, optional
        Base path for saving.  Each figure is saved as
        ``{stem}_barplot_{problem}{ext}``.
    renderer : str
        Plotly renderer used when ``show=True``.
    show : bool
        Call ``fig.show()`` for every figure produced.

    Returns
    -------
    dict[str, go.Figure]
        Keyed by problem name.
    """
    figs: dict[str, go.Figure] = {}
    for problem_name, sub in summary_df.groupby("problem"):
        sub = sub.sort_values("mean_fitness", ascending=True).reset_index(drop=True)
        n = len(sub)
        colours = [_DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)] for i in range(n)]

        bar_colours = colours.copy()
        bar_line_colours = [_PANEL_BG] * n
        bar_line_widths = [0.8] * n
        # highlight the winner with an amber border
        bar_line_colours[0] = "#F0A500"
        bar_line_widths[0] = 2.5

        fig = go.Figure(go.Bar(
            x=sub["algorithm"],
            y=sub["mean_fitness"],
            marker=dict(
                color=colours,
                line=dict(color=bar_line_colours, width=bar_line_widths),
            ),
            text=[f"{v:.4g}" for v in sub["mean_fitness"]],
            textposition="outside",
            textfont=dict(color=_TEXT_CLR, size=9),
            hovertemplate="<b>%{x}</b><br>mean fitness = %{y:.6g}<extra></extra>",
        ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(
                text=f"Mean Best Fitness  ·  {problem_name}",
                font=dict(color=_TEXT_CLR, size=16), x=0.04,
            ),
            xaxis=dict(**_AXIS_STYLE, title="Algorithm", tickangle=-40),
            yaxis=dict(**_AXIS_STYLE, title="Mean best fitness  (↓ lower is better)"),
            bargap=0.35,
            width=max(700, n * 90 + 160),
            height=480,
        )

        _save_plotly(fig, _bench_filepath(filepath, str(problem_name), "barplot"))
        if show:
            pio.renderers.default = renderer
            fig.show()
        figs[str(problem_name)] = fig
    return figs


def plot_benchmark_boxplots(
    valid_df: "pd.DataFrame",
    filepath=None,
    renderer: str = "browser",
    show: bool = False,
) -> "dict[str, go.Figure]":
    """Box plot of per-trial best-fitness distribution per algorithm per problem.

    Parameters
    ----------
    valid_df : pd.DataFrame
        Rows of ``BenchmarkRunner.run()`` that have no error.  Must contain
        columns ``problem``, ``algorithm``, ``best_fitness``.
    filepath : str or Path, optional
        Base path for saving.
    renderer : str
        Plotly renderer.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    dict[str, go.Figure]
    """
    figs: dict[str, go.Figure] = {}
    for problem_name, sub in valid_df.groupby("problem"):
        alg_order = (
            sub.groupby("algorithm")["best_fitness"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        n = len(alg_order)
        fig = go.Figure()
        for idx, alg in enumerate(alg_order):
            colour = _DISCRETE_PALETTE[idx % len(_DISCRETE_PALETTE)]
            vals = sub.loc[sub["algorithm"] == alg, "best_fitness"].tolist()
            fig.add_trace(go.Box(
                y=vals,
                name=alg,
                marker_color=colour,
                line_color=colour,
                fillcolor=colour + "22",
                boxmean="sd",
                hovertemplate=(
                    f"<b>{alg}</b><br>"
                    "median = %{median:.6g}<br>"
                    "mean ± sd<extra></extra>"
                ),
            ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(
                text=f"Trial Distribution  ·  {problem_name}",
                font=dict(color=_TEXT_CLR, size=16), x=0.04,
            ),
            xaxis=dict(**_AXIS_STYLE, title="Algorithm", tickangle=-40),
            yaxis=dict(**_AXIS_STYLE, title="Best fitness  (↓ lower is better)"),
            boxmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_CLR)),
            width=max(700, n * 90 + 160),
            height=480,
        )

        _save_plotly(fig, _bench_filepath(filepath, str(problem_name), "boxplot"))
        if show:
            pio.renderers.default = renderer
            fig.show()
        figs[str(problem_name)] = fig
    return figs


def plot_benchmark_rank_heatmap(
    rank_table: "pd.DataFrame",
    filepath=None,
    renderer: str = "browser",
    show: bool = False,
) -> go.Figure:
    """Heatmap of algorithm ranks across benchmark problems.

    Parameters
    ----------
    rank_table : pd.DataFrame
        ``pivot`` of ranks; index = algorithm, columns = problems.
        An ``average_rank`` column is silently dropped before plotting.
    filepath : str or Path, optional
        Save path.
    renderer : str
        Plotly renderer.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    plot_data = rank_table.drop(columns=["average_rank"], errors="ignore").copy()
    nrows, ncols = plot_data.shape
    zvals = plot_data.values.astype(float)
    vmax = float(np.nanmax(zvals)) if zvals.size else 1.0

    rank_colorscale = [
        [0.0,  "#0D2137"],
        [0.25, "#0E4D6A"],
        [0.5,  "#0F7BAE"],
        [0.75, "#3BC9DB"],
        [1.0,  "#A5F3FC"],
    ]

    annotations = []
    for i in range(nrows):
        for j in range(ncols):
            val = zvals[i, j]
            norm = val / vmax if vmax else 0
            txt_colour = _TEXT_CLR if norm > 0.45 else _DARK_BG
            annotations.append(dict(
                x=j, y=i,
                text=f"{val:.1f}",
                xref="x", yref="y",
                showarrow=False,
                font=dict(color=txt_colour, size=10, family="IBM Plex Mono, monospace"),
            ))

    fig = go.Figure(go.Heatmap(
        z=zvals,
        x=list(plot_data.columns),
        y=list(plot_data.index),
        colorscale=rank_colorscale,
        zmin=1,
        zmax=vmax,
        colorbar=dict(
            title=dict(text="Rank", font=dict(color=_TEXT_CLR, size=11)),
            tickfont=dict(color=_TEXT_CLR, size=9),
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
            thickness=14,
        ),
        hovertemplate="<b>%{y}</b>  ·  %{x}<br>rank = %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        annotations=annotations,
        title=dict(
            text="Algorithm Rank by Benchmark Problem  ·  lower is better",
            font=dict(color=_TEXT_CLR, size=16), x=0.04,
        ),
        xaxis=dict(
            **_AXIS_STYLE,
            title="Problem",
            tickangle=-35,
            side="bottom",
        ),
        yaxis=dict(
            **_AXIS_STYLE,
            title="Algorithm",
            autorange="reversed",
        ),
        width=max(600, ncols * 130 + 200),
        height=max(350, nrows * 42 + 150),
    )

    _save_plotly(fig, filepath)
    if show:
        pio.renderers.default = renderer
        fig.show()
    return fig


def plot_benchmark_runtime(
    raw_df: "pd.DataFrame",
    filepath=None,
    renderer: str = "browser",
    show: bool = False,
) -> go.Figure:
    """Bar chart of mean runtime per algorithm across all successful runs.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Full output of ``BenchmarkRunner.run()``.  Must contain columns
        ``algorithm``, ``error``, ``elapsed_s``.
    filepath : str or Path, optional
        Save path.
    renderer : str
        Plotly renderer.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    runtime = (
        raw_df[raw_df["error"].isna()]
        .groupby("algorithm", as_index=False)["elapsed_s"]
        .mean()
        .sort_values("elapsed_s", ascending=True)
        .reset_index(drop=True)
    )
    n = len(runtime)
    colours = [_DISCRETE_PALETTE[i % len(_DISCRETE_PALETTE)] for i in range(n)]
    # highlight fastest with green border
    line_colours = [_PANEL_BG] * n
    line_widths = [0.8] * n
    if n:
        line_colours[0] = "#3FB950"
        line_widths[0] = 2.5

    fig = go.Figure(go.Bar(
        x=runtime["algorithm"],
        y=runtime["elapsed_s"],
        marker=dict(
            color=colours,
            line=dict(color=line_colours, width=line_widths),
        ),
        text=[f"{v:.3g}s" for v in runtime["elapsed_s"]],
        textposition="outside",
        textfont=dict(color=_TEXT_CLR, size=9),
        hovertemplate="<b>%{x}</b><br>mean runtime = %{y:.4g} s<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="Average Runtime per Run",
            font=dict(color=_TEXT_CLR, size=16), x=0.04,
        ),
        xaxis=dict(**_AXIS_STYLE, title="Algorithm", tickangle=-40),
        yaxis=dict(**_AXIS_STYLE, title="Elapsed seconds"),
        bargap=0.35,
        width=max(700, n * 90 + 160),
        height=480,
    )

    _save_plotly(fig, filepath)
    if show:
        pio.renderers.default = renderer
        fig.show()
    return fig


def plot_benchmark_convergence(
    algorithms: "list[str]",
    problems: "list[dict]",
    termination,
    seed: int = 2026,
    filepath=None,
    renderer: str = "browser",
    show: bool = False,
) -> "dict[str, go.Figure]":
    """Convergence curves for every algorithm on every benchmark problem.

    Runs each algorithm once (``store_history=True``) and overlays all
    curves on a per-problem figure.

    Parameters
    ----------
    algorithms : list[str]
        Algorithm IDs understood by ``pymetaheuristic.optimize``.
    problems : list[dict]
        Each dict must have keys ``name``, ``target_function``,
        ``min_values``, ``max_values`` and optionally ``objective``
        (default ``"min"``).
    termination : Termination
        ``pymetaheuristic.Termination`` instance.
    seed : int
        Random seed passed to every run.
    filepath : str or Path, optional
        Base path for saving.  Each figure is saved as
        ``{stem}_convergence_{problem}{ext}``.
    renderer : str
        Plotly renderer.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    dict[str, go.Figure]
        Keyed by problem name.
    """
    from .api import optimize  # local import to avoid circularity

    figs: dict[str, go.Figure] = {}
    for problem in problems:
        problem_name = problem["name"]
        fig = go.Figure()

        plotted = 0
        for idx, alg in enumerate(algorithms):
            colour = _DISCRETE_PALETTE[idx % len(_DISCRETE_PALETTE)]
            try:
                result = optimize(
                    algorithm=alg,
                    target_function=problem["target_function"],
                    min_values=problem["min_values"],
                    max_values=problem["max_values"],
                    objective=problem.get("objective", "min"),
                    termination=termination,
                    seed=seed,
                    store_history=True,
                    verbose=False,
                )
                xs, ys = _history_series(result, key="best_fitness")
                if xs.size == 0:
                    continue

                # glow effect: thick transparent trace behind a sharp line
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(color=colour, width=8),
                    opacity=0.08,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=alg,
                    line=dict(color=colour, width=1.8),
                    hovertemplate=(
                        f"<b>{alg}</b><br>"
                        "step = %{x}<br>"
                        "best fitness = %{y:.6g}<extra></extra>"
                    ),
                ))
                plotted += 1

            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(
                    f"plot_benchmark_convergence: algorithm={alg!r}, "
                    f"problem={problem_name!r} failed: {exc}",
                    stacklevel=2,
                )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(
                text=f"Convergence Curves  ·  {problem_name}",
                font=dict(color=_TEXT_CLR, size=16), x=0.04,
            ),
            xaxis=dict(**_AXIS_STYLE, title="Step"),
            yaxis=dict(**_AXIS_STYLE, title="Best fitness  (↓ lower is better)", type="log"),
            legend=dict(
                bgcolor="rgba(0,0,0,0.25)",
                font=dict(color=_TEXT_CLR, size=9),
                bordercolor=_GRID_CLR,
                borderwidth=1,
                ncols=max(1, plotted // 10),
            ),
            width=1000,
            height=520,
        )

        _save_plotly(fig, _bench_filepath(filepath, str(problem_name), "convergence"))
        if show:
            pio.renderers.default = renderer
            fig.show()
        figs[str(problem_name)] = fig
    return figs


############################################################################
# Legacy aliases (backward-compat)
############################################################################

def plot_function_contour(
    target_function, min_values, max_values,
    best_position=None, filepath=None,
    title="Function contour", grid_points=200, levels=30, show=False,
):
    """Legacy wrapper -> plot_function_2d."""
    sol = [best_position] if best_position is not None else None
    return plot_function_2d(
        target_function, min_values, max_values,
        solutions=sol, title=title, grid_points=grid_points,
        filepath=filepath, renderer="browser", contour_levels=levels,
    )


def plot_function_surface(
    target_function, min_values, max_values,
    best_position=None, filepath=None,
    title="Function surface", grid_points=120, show=False,
):
    """Legacy wrapper -> plot_function_3d."""
    sol = [best_position] if best_position is not None else None
    return plot_function_3d(
        target_function, min_values, max_values,
        solutions=sol, title=title, grid_points=grid_points,
        filepath=filepath, renderer="browser",
    )


############################################################################
# Collaboration / island dynamics
############################################################################

def plot_island_dynamics(result, metric: str = "best_fitness", title: str = "Island dynamics", filepath=None, renderer: str = "browser") -> go.Figure:
    telemetry = getattr(result, "island_telemetry", {}) or {}
    fig = go.Figure()
    for label, records in telemetry.items():
        if not records:
            continue
        x = [getattr(r, 'global_step', i + 1) for i, r in enumerate(records)]
        y = [getattr(r, metric, None) for r in records]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=label, hovertemplate='island=%{text}<br>step=%{x}<br>value=%{y}<extra></extra>', text=[label] * len(x)))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(**_AXIS_STYLE, title='Global step'),
        yaxis=dict(**_AXIS_STYLE, title=metric),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=_TEXT_CLR)),
    )
    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig


def plot_collaboration_network(result, title: str = "Collaboration dynamics", filepath=None, renderer: str = "browser") -> go.Figure:
    events = list(getattr(result, 'events', []) or [])
    meta = getattr(result, 'metadata', {}) or {}
    nodes = list(meta.get('islands', []))
    if not nodes:
        nodes = sorted({getattr(ev, 'source_label', None) for ev in events} | {getattr(ev, 'target_label', None) for ev in events})
        nodes = [n for n in nodes if n is not None]
    if not nodes:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, x=0.04))
        return fig
    n = len(nodes)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = {label: (float(np.cos(t)), float(np.sin(t))) for label, t in zip(nodes, theta)}
    edge_weight = {}
    for ev in events:
        src = getattr(ev, 'source_label', None) if not isinstance(ev, dict) else ev.get('source_label')
        tgt = getattr(ev, 'target_label', None) if not isinstance(ev, dict) else ev.get('target_label')
        if src is None or tgt is None:
            continue
        edge_weight[(src, tgt)] = edge_weight.get((src, tgt), 0) + 1
    fig = go.Figure()
    for (src, tgt), w in edge_weight.items():
        x0, y0 = coords[src]
        x1, y1 = coords[tgt]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=1 + 0.7 * w, color=_ACCENT), opacity=0.55, hovertemplate=f'{src} → {tgt}<br>migrations={w}<extra></extra>', showlegend=False))
    fig.add_trace(go.Scatter(
        x=[coords[n][0] for n in nodes],
        y=[coords[n][1] for n in nodes],
        mode='markers+text',
        text=nodes,
        textposition='top center',
        marker=dict(size=18, color=_SOL_CLR, line=dict(width=2, color='white')),
        hovertemplate='island=%{text}<extra></extra>',
        showlegend=False,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(color=_TEXT_CLR, size=16), x=0.04),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    _save_plotly(fig, filepath)
    if filepath is None:
        pio.renderers.default = renderer
        fig.show()
    return fig
