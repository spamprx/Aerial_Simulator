"""Matplotlib helpers (2-D and 3-D).  The Streamlit dashboard will reuse these."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.metrics import min_separation_series, position_spread_series, velocity_spread_series
from ..core.simulator import SimResult


def _axes(dim: int, ax=None):
    if ax is not None:
        return ax
    fig = plt.figure(figsize=(6, 5))
    return fig.add_subplot(111, projection="3d") if dim == 3 else fig.add_subplot(111)


def plot_snapshot(pos, adj=None, compromised=None, ax=None, title: Optional[str] = None):
    """Agents and communication links at one instant.  Compromised agents are red."""
    dim = pos.shape[1]
    ax = _axes(dim, ax)
    colors = np.where(compromised, "tab:red", "tab:blue") if compromised is not None else "tab:blue"
    if adj is not None:
        for i, j in zip(*np.nonzero(adj)):
            seg = np.stack([pos[j], pos[i]])
            ax.plot(*seg.T, color="0.7", lw=0.6, zorder=1)
    ax.scatter(*pos.T, c=colors, s=35, zorder=2)
    if title:
        ax.set_title(title)
    return ax


def plot_trajectories(res: SimResult, ax=None, snapshot_step: Optional[int] = None):
    """Paths of all agents; start = hollow circle, end = filled.  Optionally overlays the
    communication graph at ``snapshot_step``."""
    dim = res.pos.shape[2]
    ax = _axes(dim, ax)
    for i in range(res.pos.shape[1]):
        c = "tab:red" if res.compromised[i] else None
        line, = ax.plot(*res.pos[:, i, :].T, lw=1.0, color=c)
        ax.scatter(*res.pos[0, i : i + 1, :].T, facecolors="none", edgecolors=line.get_color(), s=30)
        ax.scatter(*res.pos[-1, i : i + 1, :].T, color=line.get_color(), s=30)
    if snapshot_step is not None:
        k = min(snapshot_step, res.adj.shape[0] - 1)
        for i, j in zip(*np.nonzero(res.adj[k])):
            seg = np.stack([res.pos[k, j], res.pos[k, i]])
            ax.plot(*seg.T, color="0.8", lw=0.5, zorder=0)
    ax.set_title("Trajectories")
    return ax


def plot_timeseries(res: SimResult, fig=None):
    """Spread, separation (with d_safe line) and lambda_2(t) in one figure."""
    fig = fig or plt.figure(figsize=(9, 6))
    axs = fig.subplots(3, 1, sharex=True)
    axs[0].plot(res.t, position_spread_series(res), label="position spread")
    axs[0].plot(res.t, velocity_spread_series(res), label="velocity spread")
    axs[0].set_ylabel("spread"); axs[0].legend()
    axs[1].plot(res.t, min_separation_series(res))
    axs[1].axhline(res.cfg.d_safe, color="r", ls="--", label="d_safe"); axs[1].legend()
    axs[1].set_ylabel("min separation")
    axs[2].step(res.t[:-1], res.lambda2, where="post")
    axs[2].set_ylabel(r"$\lambda_2(L_t)$"); axs[2].set_xlabel("time [s]")
    fig.tight_layout()
    return fig
