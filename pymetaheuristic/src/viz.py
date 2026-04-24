"""
pyMetaheuristic src — History Visualization Utilities
=====================================================
Feature 3: Convergence, diversity, runtime, and explore/exploit charts.

These are matplotlib-based chart functions that operate on the history
stored in an ``OptimizationResult`` object.  matplotlib is an optional
dependency; if it is not installed, a clear ImportError is raised at
call time rather than at import time.

All functions return the ``matplotlib.figure.Figure`` object so the
caller can further customise, embed, or save it.

Usage
-----
::

    import pymetaheuristic
    from pymetaheuristic.src.viz import (
        plot_diversity_chart,
        plot_explore_exploit_chart,
        plot_runtime_chart,
        plot_global_best_chart,
    )

    result = pymetaheuristic.optimize("pso", fn, lb, ub, max_steps=200,
                                      store_history=True)

    fig = plot_global_best_chart(result, show=True)
    fig = plot_diversity_chart(result, show=True)
    fig = plot_explore_exploit_chart(result, show=True)
    fig = plot_runtime_chart(result, show=True)

    # Or the all-in-one dashboard:
    from pymetaheuristic.src.viz import plot_run_dashboard
    fig = plot_run_dashboard(result, show=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .telemetry import convergence_data


# ── Lazy matplotlib import ─────────────────────────────────────────────────

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend by default
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for pymetaheuristic.src.viz. "
            "Install it with:  pip install matplotlib"
        )


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


def _apply_dark_style(ax, plt, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.figure.set_facecolor(_DARK_BG)
    ax.set_title(title, color=_TEXT_CLR, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=_TEXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=_TEXT_CLR, fontsize=9)
    ax.tick_params(colors=_TEXT_CLR, labelsize=8)
    ax.spines["bottom"].set_color(_GRID_CLR)
    ax.spines["top"].set_color(_GRID_CLR)
    ax.spines["left"].set_color(_GRID_CLR)
    ax.spines["right"].set_color(_GRID_CLR)
    ax.grid(True, color=_GRID_CLR, linestyle="--", linewidth=0.5)


def _save_or_show(fig, filepath, show: bool, plt) -> None:
    if filepath is not None:
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)


def _extract(history: list[dict], key: str) -> list[float | None]:
    return [h.get(key) for h in history]


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
) -> Any:
    """
    Plot the global-best fitness curve over steps.

    Parameters
    ----------
    result   : OptimizationResult from optimize() / create_optimizer().run()
    filepath : Optional path to save the figure (.png / .pdf / .svg).
    title    : Optional plot title.
    show     : If True, call plt.show() immediately.
    color    : Line colour.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _mpl()
    history = getattr(result, "history", []) or []
    alg_id  = getattr(result, "algorithm_id", "algorithm")

    x, y = convergence_data(result, x_axis=x_axis)

    fig, ax = plt.subplots(figsize=(8, 4))
    xlabel = "Evaluations" if str(x_axis).lower() in {"evaluations", "evals", "evaluation"} else "Step"
    _apply_dark_style(ax, plt, title or f"Global-Best Convergence — {alg_id}",
                      xlabel, "Fitness")
    ax.plot(x, y, color=color, linewidth=1.5, label="Global best", drawstyle="steps-post" if xlabel == "Evaluations" else "default")
    ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig


# ===========================================================================
# 2. Diversity chart
# ===========================================================================

def plot_diversity_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    color: str = _GREEN,
) -> Any:
    """
    Plot population diversity over steps.

    Diversity is the mean normalised distance of population members from
    their centroid (computed automatically during the run when
    ``store_history=True``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _mpl()
    history = getattr(result, "history", []) or []
    alg_id  = getattr(result, "algorithm_id", "algorithm")

    divs = [h.get("diversity") for h in history]
    divs = [v for v in divs if v is not None]
    x    = list(range(1, len(divs) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(ax, plt, title or f"Population Diversity — {alg_id}",
                      "Step", "Diversity (normalised)")
    if divs:
        ax.fill_between(x, divs, alpha=0.25, color=color)
        ax.plot(x, divs, color=color, linewidth=1.5, label="Diversity")
        ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No diversity data in history.\nRun with store_history=True.",
                ha="center", va="center", transform=ax.transAxes, color=_TEXT_CLR, fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig


# ===========================================================================
# 3. Exploration vs Exploitation chart
# ===========================================================================

def plot_explore_exploit_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> Any:
    """
    Stacked area chart showing the proportion of steps that produced an
    improvement (exploitation) vs. those that did not (exploration).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _mpl()
    history = getattr(result, "history", []) or []
    alg_id  = getattr(result, "algorithm_id", "algorithm")

    exploit = [h.get("exploitation", 0.0) for h in history if "exploitation" in h]
    explore = [h.get("exploration", 1.0)  for h in history if "exploration"  in h]
    x       = list(range(1, len(exploit) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(ax, plt, title or f"Exploration vs Exploitation — {alg_id}",
                      "Step", "Fraction")
    if exploit:
        ax.stackplot(x, explore, exploit,
                     labels=["Exploration", "Exploitation"],
                     colors=[_AMBER, _ACCENT], alpha=0.80)
        ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No exploration/exploitation data.\nRun with store_history=True.",
                ha="center", va="center", transform=ax.transAxes, color=_TEXT_CLR, fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig


# ===========================================================================
# 4. Runtime-per-step chart
# ===========================================================================

def plot_runtime_chart(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    color: str = _ORANGE,
) -> Any:
    """
    Bar chart of cumulative elapsed time recorded in the history.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _mpl()
    history = getattr(result, "history", []) or []
    alg_id  = getattr(result, "algorithm_id", "algorithm")

    times = [h.get("elapsed_time") for h in history]
    times = [v for v in times if v is not None]
    # Convert cumulative → per-step deltas
    deltas = [times[0]] + [times[i] - times[i-1] for i in range(1, len(times))]
    x = list(range(1, len(deltas) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(ax, plt, title or f"Runtime per Step — {alg_id}",
                      "Step", "Seconds")
    if deltas:
        ax.bar(x, deltas, color=color, alpha=0.80, width=0.8)
    else:
        ax.text(0.5, 0.5, "No elapsed_time data in history.\nRun with store_history=True.",
                ha="center", va="center", transform=ax.transAxes, color=_TEXT_CLR, fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig


# ===========================================================================
# 5. All-in-one dashboard
# ===========================================================================

def plot_run_dashboard(
    result,
    filepath: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> Any:
    """
    2×2 dashboard combining convergence, diversity, explore/exploit, and
    runtime charts in a single figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _mpl()
    history = getattr(result, "history", []) or []
    alg_id  = getattr(result, "algorithm_id", "algorithm")
    dash_title = title or f"Run Dashboard — {alg_id}"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.set_facecolor(_DARK_BG)
    fig.suptitle(dash_title, color=_TEXT_CLR, fontsize=13, y=1.01)

    # ── panel helpers ────────────────────────────────────────────────────
    def _fill(ax, x, y, label, color):
        ax.fill_between(x, y, alpha=0.20, color=color)
        ax.plot(x, y, color=color, linewidth=1.5, label=label)
        ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR,
                  labelcolor=_TEXT_CLR, fontsize=7)

    # 1. Convergence
    ax = axes[0, 0]
    _apply_dark_style(ax, plt, "Global-Best Convergence", "Step", "Fitness")
    y = [h.get("global_best_fitness") or h.get("best_fitness") for h in history]
    y = [v for v in y if v is not None]
    if y: _fill(ax, range(1, len(y)+1), y, "Global best", _ACCENT)

    # 2. Diversity
    ax = axes[0, 1]
    _apply_dark_style(ax, plt, "Population Diversity", "Step", "Diversity")
    divs = [h.get("diversity") for h in history if h.get("diversity") is not None]
    if divs: _fill(ax, range(1, len(divs)+1), divs, "Diversity", _GREEN)

    # 3. Explore / Exploit
    ax = axes[1, 0]
    _apply_dark_style(ax, plt, "Exploration vs Exploitation", "Step", "Fraction")
    exploit = [h.get("exploitation", 0.0) for h in history if "exploitation" in h]
    explore = [h.get("exploration",  1.0) for h in history if "exploration"  in h]
    x = list(range(1, len(exploit) + 1))
    if exploit:
        ax.stackplot(x, explore, exploit,
                     labels=["Exploration", "Exploitation"],
                     colors=[_AMBER, _ACCENT], alpha=0.80)
        ax.set_ylim(0, 1)
        ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR,
                  labelcolor=_TEXT_CLR, fontsize=7)

    # 4. Runtime per step
    ax = axes[1, 1]
    _apply_dark_style(ax, plt, "Runtime per Step", "Step", "Seconds")
    times = [h.get("elapsed_time") for h in history if h.get("elapsed_time") is not None]
    if times:
        deltas = [times[0]] + [times[i] - times[i-1] for i in range(1, len(times))]
        ax.bar(range(1, len(deltas)+1), deltas, color=_ORANGE, alpha=0.80, width=0.8)

    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig


# ===========================================================================
# 6. Multi-algorithm diversity comparison
# ===========================================================================

def plot_diversity_comparison(
    results: dict[str, Any],
    filepath: str | Path | None = None,
    title: str = "Diversity Comparison",
    show: bool = False,
) -> Any:
    """
    Overlay diversity curves from multiple runs / algorithms.

    Parameters
    ----------
    results : dict mapping label → OptimizationResult
    """
    plt = _mpl()
    import matplotlib.cm as cm
    import numpy as np

    colors = [_ACCENT, _GREEN, _AMBER, _RED, _ORANGE,
              "#bc8cff", "#79c0ff", "#56d364"]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(ax, plt, title, "Step", "Diversity (normalised)")

    for i, (label, result) in enumerate(results.items()):
        history = getattr(result, "history", []) or []
        divs    = [h.get("diversity") for h in history if h.get("diversity") is not None]
        if divs:
            color = colors[i % len(colors)]
            ax.plot(range(1, len(divs)+1), divs,
                    color=color, linewidth=1.5, label=label)

    ax.legend(facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor=_TEXT_CLR, fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, filepath, show, plt)
    return fig
