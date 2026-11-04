"""Matplotlib helpers for the notebooks, using one consistent palette."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .data import PITCH_NAMES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10, "axes.grid": False,
})


def pitch_colors(classes) -> dict:
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(sorted(classes))}


def name(c):
    return PITCH_NAMES.get(c, c)


def hbar(series: pd.Series, title: str, colors=None, fmt="{:.0%}", ax=None, xlabel=None):
    ax = ax or plt.subplots(figsize=(6, 0.35 * len(series) + 0.8))[1]
    s = series[::-1]
    ax.barh([name(i) if isinstance(i, str) else i for i in s.index], s.values,
            color=[(colors or {}).get(i, SERIES[0]) for i in s.index], height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + fmt.format(v), va="center", color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def heatmap(df: pd.DataFrame, title: str, fmt="{:.0%}", diverging=False, ax=None, annotate=True):
    ax = ax or plt.subplots(figsize=(0.9 * df.shape[1] + 2.5, 0.38 * df.shape[0] + 1))[1]
    vals = df.values.astype(float)
    if diverging:
        m = np.nanmax(np.abs(vals)) or 1
        im = ax.imshow(vals, cmap=DIVERGE, norm=TwoSlopeNorm(0, -m, m), aspect="auto")
    else:
        im = ax.imshow(vals, cmap=BLUES, aspect="auto", vmin=0)
    ax.set_xticks(range(df.shape[1]), [name(c) if isinstance(c, str) else c for c in df.columns], rotation=0)
    ax.set_yticks(range(df.shape[0]), [name(r) if isinstance(r, str) else r for r in df.index])
    if annotate and df.size <= 200:
        hi = np.nanmax(np.abs(vals)) or 1
        for (i, j), v in np.ndenumerate(vals):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * hi else INK)
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return ax


def compare_models(table: pd.DataFrame, title="Model comparison"):
    """table: index = model, columns accuracy, log_loss (and optional family)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(table) + 1.2), sharey=True)
    fam = table.get("family", pd.Series("", index=table.index))
    col = fam.map({"supervised": SERIES[0], "unsupervised": SERIES[1]}).fillna("#898781")
    for ax, metric, f in zip(axes, ["accuracy", "log_loss"], ["{:.0%}", "{:.3f}"]):
        s = table[metric][::-1]
        ax.barh(s.index, s.values, color=col[::-1], height=0.6)
        for y, v in enumerate(s.values):
            ax.text(v, y, " " + f.format(v), va="center", color=INK2)
        ax.set_title(("Accuracy (higher is better)" if metric == "accuracy" else "Log loss (lower is better)"), loc="left")
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=SERIES[0], label="Supervised"), Patch(color=SERIES[1], label="Unsupervised"),
                            Patch(color="#898781", label="Baseline")], loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def skeleton(ax, pose_bins: np.ndarray, joints, bones, bins=(0, 3, 5, 7), title=""):
    """Draw mean skeletons (N_BINS, joints, 2) side by side."""
    for k, b in enumerate(bins):
        P = pose_bins[b]
        for a, c in bones:
            ia, ic = joints.index(a), joints.index(c)
            throw = a.startswith("r_") or c.startswith("r_")
            ax.plot([P[ia, 0] + 3 * k, P[ic, 0] + 3 * k], [P[ia, 1], P[ic, 1]],
                    color=SERIES[0] if throw else "#c3c2b7", lw=2.5, solid_capstyle="round")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)
