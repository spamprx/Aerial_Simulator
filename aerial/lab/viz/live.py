"""Live (animated) view: the simulator is stepped *while* it is drawn.

    from aerial import SimConfig
    from aerial.live import animate
    animate(SimConfig(n_agents=12), speed=1.0)            # opens a window
    animate(SimConfig(n_agents=12), save="run.gif")       # headless: write a GIF instead

``speed`` is simulated seconds per wall-clock second (1.0 = real time).  The target is
approximate: if drawing is slower than the requested rate the animation simply lags.
Keys in the window:  space = pause / resume.

What is drawn: agents (blue; compromised = red), their recent trails, the communication
links of the current step (grey), and a title with time, link count, lambda_2 and minimum
separation.  The title turns red while any pair is closer than ``cfg.d_safe``.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..config import SimConfig
from ..network.graph import algebraic_connectivity, pairwise_distances
from ..core.simulator import Simulator


def _plot(ax, dim, *args, **kw):
    """Create an empty Line2D / Line3D artist."""
    return ax.plot(*([[]] * dim), *args, **kw)[0]


def _set(line, P, dim):
    line.set_data(P[:, 0], P[:, 1])
    if dim == 3:
        line.set_3d_properties(P[:, 2])


def animate(
    sim: Union[Simulator, SimConfig],
    speed: float = 1.0,
    trail: int = 40,
    save: Optional[str] = None,
    show: Optional[bool] = None,
    max_frames: Optional[int] = None,
    dpi: int = 90,
):
    """Step the simulator and draw it.  Returns the ``FuncAnimation`` (keep a reference).

    save        path ending in .gif (needs only Pillow) or .mp4 (needs ffmpeg).  When given,
                the window is not shown unless ``show=True`` is forced.
    trail       number of past positions kept per agent (0 = no trails).
    max_frames  stop early (handy for quick previews / tests).
    """
    if isinstance(sim, SimConfig):
        sim = Simulator(sim)
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if trail < 0:
        raise ValueError("trail must be >= 0")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    cfg, dim, n = sim.cfg, sim.dim, sim.n
    show = (save is None) if show is None else show

    # pacing: pick steps-per-frame so a frame lasts >= ~30 ms of wall time
    spf = max(1, math.ceil(0.03 * speed / cfg.dt))
    interval_ms = spf * cfg.dt * 1000.0 / speed
    remaining_steps = max(0, cfg.n_steps - sim.k)
    n_frames = max(1, math.ceil(remaining_steps / spf))
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    fig = plt.figure(figsize=(7, 6.2))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    for name, label in zip(("set_xlabel", "set_ylabel", "set_zlabel"), "xyz"):
        if hasattr(ax, name):
            getattr(ax, name)(label)
    if dim == 2:
        ax.set_aspect("equal")

    # one cubic / square view window that only ever grows, so nothing flies out of frame
    box = {"lo": -0.1 * cfg.arena_size, "hi": 1.1 * cfg.arena_size}

    def fit(P):
        lo, hi = P.min(), P.max()
        pad = 0.05 * max(hi - lo, 1.0)
        changed = False
        if lo - pad < box["lo"]:
            box["lo"], changed = lo - pad, True
        if hi + pad > box["hi"]:
            box["hi"], changed = hi + pad, True
        if changed or not ax.get_xlim() == (box["lo"], box["hi"]):
            ax.set_xlim(box["lo"], box["hi"]); ax.set_ylim(box["lo"], box["hi"])
            if dim == 3:
                ax.set_zlim(box["lo"], box["hi"])

    bad = sim.compromised
    honest = _plot(ax, dim, "o", color="tab:blue", ms=7, zorder=3, label="honest")
    comp = _plot(ax, dim, "o", color="tab:red", ms=7, zorder=3, label="compromised")
    trails = [_plot(ax, dim, "-", lw=0.9, alpha=0.45,
                    color="tab:red" if bad[i] else "tab:blue") for i in range(n)]
    if dim == 3:
        # add_collection3d autoscales from the segments, which fails when empty -> seed one
        links = Line3DCollection([np.zeros((2, 3))], colors="0.65", linewidths=0.7)
        ax.add_collection3d(links)
    else:
        links = LineCollection([], colors="0.65", linewidths=0.7)
        ax.add_collection(links)
    if bad.any():
        ax.legend(loc="upper right")

    history = deque(maxlen=max(trail, 1))
    state = {"adj": None}

    def draw():
        P = sim.pos
        fit(P)
        _set(honest, P[~bad], dim)
        _set(comp, P[bad], dim)
        if trail > 0:
            history.append(P.copy())
            H = np.stack(history)  # (h, N, d)
            for i, ln in enumerate(trails):
                _set(ln, H[:, i, :], dim)
        adj = state["adj"]
        if adj is None:
            links.set_segments([])
            title, red = f"t = {sim.t:5.1f} s", False
        else:
            und = np.triu(adj | adj.T, 1)
            links.set_segments([np.stack([P[i], P[j]]) for i, j in np.argwhere(und)])
            dmin = (pairwise_distances(P) + np.eye(n) * 1e9).min() if n > 1 else np.inf
            title = (f"t = {sim.t:5.1f} s    links {int(und.sum())}    "
                     rf"$\lambda_2$ = {algebraic_connectivity(adj):.2f}    min sep = {dmin:.2f}")
            red = dmin < cfg.d_safe
        ax.set_title(title, color="tab:red" if red else "black", fontsize=10)

    def init():
        draw()
        return []

    def update(_frame):
        for _ in range(spf):
            if sim.k >= cfg.n_steps:
                break
            obs, _ = sim.step()
            state["adj"] = obs.adj
        draw()
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=interval_ms, blit=False, repeat=False)

    paused = {"v": False}

    def on_key(event):
        if event.key == " ":
            paused["v"] = not paused["v"]
            try:
                anim.pause() if paused["v"] else anim.resume()
            except AttributeError:  # very old matplotlib
                pass

    fig.canvas.mpl_connect("key_press_event", on_key)

    if save:
        if str(save).lower().endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=max(1, round(1000.0 / interval_ms))), dpi=dpi)
        else:
            anim.save(save, dpi=dpi)  # mp4 etc. -> needs ffmpeg
        print(f"saved {save}")
    if show:
        plt.show()
    return anim
