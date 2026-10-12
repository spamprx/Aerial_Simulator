"""The figures for the robustness and degradation studies (T2.6, T2.7).

    python scripts/plot_robustness.py    # redraw both from the latest rows, without re-running the studies
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITION = {True: "#2a78d6", False: "#eb6834"}  # validated with --pairs all
BOUNDS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}  # the first three slots, validated with --pairs all
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_success(rows, folder):
    """Success rate against max_r, one panel per F, coloured by whether (2F + 1)-robustness holds."""
    rows = [r for r in rows if r["filter"] == "wmsr"]
    bounds = sorted({r["F"] for r in rows})
    fig, axes = plt.subplots(1, len(bounds), figsize=(5.8 * len(bounds), 4.4), squeeze=False, sharey=True)
    for ax, F in zip(axes[0], bounds):
        of_f = [r for r in rows if r["F"] == F]
        _style(ax, r"largest $r$ for which the graph is $r$-robust", "runs that succeeded (%)",
               f"F = {F}: the condition is {2 * F + 1}-robust")
        for holds, colour in CONDITION.items():
            values = sorted({r["max_r"] for r in of_f if r["condition_holds"] == holds})
            if not values:
                continue
            counts = [sum(1 for r in of_f if r["max_r"] == v) for v in values]
            rates = [100.0 * np.mean([r["success"] for r in of_f if r["max_r"] == v]) for v in values]
            label = f"{2 * F + 1}-robust" if holds else f"not {2 * F + 1}-robust"
            ax.bar(values, rates, width=0.55, color=colour, label=label, zorder=2)
            for value, rate, count in zip(values, rates, counts):
                ax.annotate(f"n={count}", (value, rate), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=7.5, color=MUTED)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 25))
        ax.set_xticks(sorted({r["max_r"] for r in of_f}))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=MUTED)
    fig.tight_layout(w_pad=3)
    return _save(fig, Path(folder) / "er_robustness.png")


def _medians(rows, key, F, actual):
    return [np.median([r[key] for r in rows if r["assumed_F"] == F and r["actual"] == a]) for a in actual]


def plot_degradation(rows, folder):
    """Four panels against how many are really compromised, with one line per assumed F; the last
    shows what the filter costs, against F itself."""
    actual = sorted({r["actual"] for r in rows})
    assumed = sorted({r["assumed_F"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    panels = ((axes[0, 0], "final_spread", "final honest spread (m)", "Agreement fails once liars outnumber the bound (1e-13 = agreed)", 1e-13),
              (axes[0, 1], "hull_overshoot", "furthest outside the start box (m)", "Honest agents get dragged away (0.001 = never)", 1e-3))
    for ax, key, ylabel, title, floor in panels:
        _style(ax, "agents really compromised", ylabel, title)
        ax.set_yscale("log")
        for F in assumed:
            ax.plot(actual, np.maximum(_medians(rows, key, F, actual), floor), color=BOUNDS[F], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=f"assumed F = {F}", zorder=3)
        ax.set_xticks(actual)

    success = axes[1, 0]
    _style(success, "agents really compromised", "runs that succeeded (%)", "Success falls as soon as liars exceed F")
    for F in assumed:
        rates = [100.0 * np.mean([r["success"] for r in rows if r["assumed_F"] == F and r["actual"] == a]) for a in actual]
        success.plot(actual, rates, color=BOUNDS[F], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                     markeredgewidth=1.0, label=f"assumed F = {F}", zorder=3)
    success.set_xticks(actual)
    success.set_ylim(-4, 104)

    cost = axes[1, 1]
    _style(cost, "assumed F", "honest messages thrown away (%)", "The price, with nobody lying: honest messages discarded")
    clean = [100.0 * np.mean([r["honest_discarded"] for r in rows if r["assumed_F"] == F and r["actual"] == 0]) for F in assumed]
    cost.bar(assumed, clean, width=0.5, color=MUTED, zorder=2)
    for F, value in zip(assumed, clean):
        cost.annotate(f"{value:.0f}%", (F, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=MUTED)
    cost.set_xticks(assumed)
    cost.set_ylim(0, max(max(clean) * 1.2, 1.0))
    handles, labels = success.get_legend_handles_labels()
    fig.tight_layout(w_pad=3, h_pad=2.5, rect=(0, 0, 1, 0.96))
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(assumed), labelcolor=MUTED)
    return _save(fig, Path(folder) / "degradation.png")


def load_rows(path):
    """Rows from a study's rows.csv, with numbers and booleans restored."""
    def parse(text):
        if text in ("True", "False"):
            return text == "True"
        try:
            value = float(text)
        except ValueError:
            return text
        return int(value) if value.is_integer() else value

    with open(path) as f:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(f)]


def newest_recording(study, root="results"):
    """The latest folder of a study.  rows.csv is the last thing written, so one without it never finished."""
    return sorted(path for path in (Path(root) / study).iterdir() if (path / "rows.csv").exists())[-1]


def latest_rows(study, root="results"):
    return load_rows(newest_recording(study, root) / "rows.csv")


if __name__ == "__main__":
    print(plot_success(latest_rows("er_robustness"), Path("results") / "figures"))
    print(plot_degradation(latest_rows("degradation"), Path("results") / "figures"))
