"""The figures for the lambda_2 study (T1.9), drawn from its rows.  Used by exp_lambda2.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from aerial.lab.stability import predicted_rate

SERIES = {10: "#2a78d6", 20: "#eb6834"}  # fleet size; validated with --pairs all
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MARKERS = dict(s=46, edgecolors=SURFACE, linewidths=1.0, zorder=3)
RATE_TICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
LAMBDA_TICKS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
TIME_TICKS = (5, 10, 20, 50, 100, 200)
# Where each graph's name sits relative to its cluster, in points, for the two no-loss panels.
RATE_LABELS = {"line": (0, 9), "ring": (0, 9), "grid": (-20, 7), "star": (14, -13), "complete": (0, 9),
               "ER p=0.2": (34, -11), "ER p=0.4": (0, 12), "ER p=0.6": (0, -16)}
TIME_LABELS = {"line": (24, -3), "ring": (22, 6), "grid": (24, 2), "star": (0, -16), "complete": (0, 10),
               "ER p=0.2": (-32, -12), "ER p=0.4": (0, 11), "ER p=0.6": (0, -16)}


def _style(ax, xlabel, ylabel, xticks, yticks):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    for axis, ticks in ((ax.xaxis, xticks), (ax.yaxis, yticks)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plain)
        axis.set_minor_locator(NullLocator())
    ax.tick_params(colors=MUTED, which="both", length=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=11)


def _legend(ax, loc):
    ax.legend(frameon=False, loc=loc, labelcolor=MUTED)


def _scatter(ax, rows, x_key, y_key):
    for n, colour in SERIES.items():
        chosen = [r for r in rows if r["n"] == n and np.isfinite(r[y_key])]
        if chosen:
            ax.scatter([r[x_key] for r in chosen], [r[y_key] for r in chosen], color=colour, label=f"N = {n}", **MARKERS)


def _label_graphs(ax, rows, y_key, offsets, n=None):
    """Name each graph once, beside its cluster for the largest fleet; the legend and rows carry the rest."""
    n = max(r["n"] for r in rows) if n is None else n
    for graph, offset in offsets.items():
        chosen = [r for r in rows if r["graph"] == graph and r["n"] == n and np.isfinite(r[y_key])]
        if chosen:
            at = (np.mean([r["lambda2"] for r in chosen]), np.mean([r[y_key] for r in chosen]))
            ax.annotate(graph, at, xytext=offset, textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)


def _prediction(ax, rows, k_p, k_v, dt):
    values = [r["lambda2"] for r in rows]
    lam = np.logspace(np.log10(min(values)) - 0.1, np.log10(max(values)) + 0.1, 300)
    ax.plot(lam, [predicted_rate(k_p, k_v, dt, x) for x in lam], color=INK, linewidth=1.6, zorder=2,
            label=r"predicted from $\lambda_2$ alone")


def _identity(ax, limits):
    ax.plot(limits, limits, color=INK, linewidth=1.6, zorder=2, label="measured = predicted")
    ax.set_xlim(limits)
    ax.set_ylim(limits)


def _rate_limits(rows, key):
    chosen = [r for r in rows if np.isfinite(r["rate"])]
    return (min(min(r["rate"], r[key]) for r in chosen) * 0.8, max(max(r["rate"], r[key]) for r in chosen) * 1.25)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_convergence(rows, folder, k_p, k_v, dt):
    """Decay rate and time to converge against lambda_2, with no packet loss."""
    rows = [r for r in rows if r["drop_prob"] == 0.0]
    fig, (rate, time) = plt.subplots(1, 2, figsize=(11, 4.6))
    _style(rate, r"algebraic connectivity $\lambda_2(L)$", "decay rate of position spread (1/s)", LAMBDA_TICKS, RATE_TICKS)
    _prediction(rate, rows, k_p, k_v, dt)
    rate.axhline(k_p / k_v, color=MUTED, linewidth=0.8, zorder=1)
    rate.annotate(r"overdamped limit $k_p / k_v$", (rate.get_xlim()[0], k_p / k_v), xytext=(4, -12),
                  textcoords="offset points", fontsize=8.5, color=MUTED)
    _scatter(rate, rows, "lambda2", "rate")
    _label_graphs(rate, rows, "rate", RATE_LABELS)
    _legend(rate, "lower right")
    _title(rate, r"Convergence speeds up with $\lambda_2$, then saturates")

    _style(time, r"algebraic connectivity $\lambda_2(L)$", "time until spread stays below 0.1 m (s)", LAMBDA_TICKS, TIME_TICKS)
    _scatter(time, rows, "lambda2", "t_converge")
    _label_graphs(time, rows, "t_converge", TIME_LABELS)
    _legend(time, "upper right")
    _title(time, "Time to converge falls the same way")
    fig.tight_layout(w_pad=3)
    return _save(fig, Path(folder) / "lambda2_convergence.png")


def plot_agreement(rows, folder):
    """Measured against predicted, from lambda_2 alone and from the whole spectrum, on the same scale."""
    rows = [r for r in rows if r["drop_prob"] == 0.0]
    fig, (alone, whole) = plt.subplots(1, 2, figsize=(10, 4.8))
    limits = _rate_limits(rows, "lambda2_rate")
    for ax, key, title in ((alone, "lambda2_rate", r"Predicted from $\lambda_2$ alone"),
                           (whole, "fleet_rate", "Predicted from every eigenvalue of $L$")):
        _style(ax, "predicted decay rate (1/s)", "measured decay rate (1/s)", RATE_TICKS, RATE_TICKS)
        _identity(ax, limits)
        _scatter(ax, rows, key, "rate")
        _legend(ax, "upper left")
        _title(ax, title)
    fig.tight_layout(w_pad=3)
    return _save(fig, Path(folder) / "lambda2_agreement.png")


def plot_packet_loss(rows, folder, k_p, k_v, dt):
    """Against the mean lambda_2(t) on top, and against (1 - p) * L below, one column per loss rate."""
    drops = sorted({r["drop_prob"] for r in rows if r["drop_prob"] > 0.0})
    if not drops:
        return None
    lossy = [r for r in rows if r["drop_prob"] > 0.0]
    limits = _rate_limits(lossy, "fleet_rate")
    span = (min(r["lambda2"] for r in lossy) * 0.5, max(r["lambda2"] for r in lossy) * 2.0)
    rates = (min(r["rate"] for r in lossy if np.isfinite(r["rate"])) * 0.6, 2.0)
    fig, axes = plt.subplots(2, len(drops), figsize=(5.4 * len(drops), 9.0), squeeze=False)
    for column, drop_prob in enumerate(drops):
        chosen = [r for r in rows if r["drop_prob"] == drop_prob]
        top, bottom = axes[0, column], axes[1, column]
        _style(top, r"mean algebraic connectivity $\lambda_2(t)$", "decay rate of position spread (1/s)", LAMBDA_TICKS, RATE_TICKS)
        _prediction(top, chosen, k_p, k_v, dt)
        _scatter(top, chosen, "lambda2", "rate")
        _label_graphs(top, chosen, "rate", {"star": (0, 9), "line": (-2, 9)})
        top.set_xlim(span)
        top.set_ylim(rates)
        _legend(top, "lower right")
        _title(top, f"{drop_prob:.0%} of packets lost: against the mean of $\\lambda_2(t)$")

        _style(bottom, r"predicted decay rate from $(1-p)\,L$ (1/s)", "measured decay rate (1/s)", RATE_TICKS, RATE_TICKS)
        _identity(bottom, limits)
        _scatter(bottom, chosen, "fleet_rate", "rate")
        _legend(bottom, "upper left")
        _title(bottom, f"{drop_prob:.0%} of packets lost: against the average graph")
    fig.tight_layout(w_pad=3, h_pad=2.5)
    return _save(fig, Path(folder) / "lambda2_packet_loss.png")


def plot_all(rows, folder, k_p, k_v, dt):
    figures = [plot_convergence(rows, folder, k_p, k_v, dt), plot_agreement(rows, folder),
               plot_packet_loss(rows, folder, k_p, k_v, dt)]
    return [figure for figure in figures if figure is not None]
