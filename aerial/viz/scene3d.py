"""Plotly 3D scene (T4.6): buildings as merged meshes, drones with FOV frustums and trails, survivor states, the
liar's halo once caught, and a time slider. ``side_by_side`` puts two paired runs next to each other."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:          # the dashboard extra is optional
    go = None

COLORS = {"honest": "#2b6cb0", "liar": "#c53030", "caught": "#ed8936", "alive": "#2f855a", "dead": "#4a5568",
          "unfound": "#e53e3e", "unsure": "#d69e2e"}


def _box_mesh(x0, x1, y0, y1, h):
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [0, 0, 0, 0, h, h, h, h]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return xs, ys, zs, i, j, k


def terrain_mesh(terrain, min_h: float = 0.5, step: int = 2):
    """One mesh for every raised block of terrain, downsampled by ``step`` cells (keeps the figure light)."""
    H = terrain.H
    n = H.shape[0]
    X, Y, Z, I, J, K = [], [], [], [], [], []
    c = terrain.cell
    for a in range(0, n, step):
        for b in range(0, n, step):
            h = float(H[a:a + step, b:b + step].max())
            if h < min_h:
                continue
            xs, ys, zs, i, j, k = _box_mesh(a * c, (a + step) * c, b * c, (b + step) * c, h)
            base = len(X)
            X += xs; Y += ys; Z += zs
            I += [v + base for v in i]; J += [v + base for v in j]; K += [v + base for v in k]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=Z, colorscale="Greys", showscale=False,
                     opacity=0.85, flatshading=True, name="buildings / rubble", hoverinfo="skip")


def frustum_lines(pos, yaw, sensor, length: float = 6.0):
    az = np.radians([-sensor.fov_h / 2, sensor.fov_h / 2])
    el = np.radians([sensor.tilt - sensor.fov_v / 2, sensor.tilt + sensor.fov_v / 2])
    xs, ys, zs = [], [], []
    for p, y in zip(pos, yaw):
        corners = []
        for a in az:
            for e in el:
                d = np.array([np.cos(e) * np.cos(y + a), np.cos(e) * np.sin(y + a), -np.sin(e)])
                corners.append(p + d * length)
        for cpt in corners:
            xs += [p[0], cpt[0], None]; ys += [p[1], cpt[1], None]; zs += [p[2], cpt[2], None]
        for a_, b_ in [(0, 1), (1, 3), (3, 2), (2, 0)]:
            xs += [corners[a_][0], corners[b_][0], None]
            ys += [corners[a_][1], corners[b_][1], None]
            zs += [corners[a_][2], corners[b_][2], None]
    return xs, ys, zs


def _frame_traces(res, scn, k, sensor, caught_at: Optional[float], found: Dict[int, float]):
    t = res.t[k]
    pos, yaw = res.pos[k], res.yaw[k]
    liar = res.liars
    colors = [COLORS["caught"] if (liar[i] and caught_at is not None and t >= caught_at)
              else COLORS["liar"] if liar[i] else COLORS["honest"] for i in range(len(pos))]
    tr = []
    tr.append(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
                           text=[str(i) for i in range(len(pos))], textposition="top center",
                           marker=dict(size=5, color=colors), name="drones"))
    fx, fy, fz = frustum_lines(pos, yaw, sensor)
    tr.append(go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="#63b3ed", width=1), name="camera",
                           hoverinfo="skip"))
    k0 = max(0, k - 150)
    tx, ty, tz = [], [], []
    for i in range(len(pos)):
        tx += list(res.pos[k0:k + 1, i, 0]) + [None]
        ty += list(res.pos[k0:k + 1, i, 1]) + [None]
        tz += list(res.pos[k0:k + 1, i, 2]) + [None]
    tr.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines", line=dict(color="#a0aec0", width=2), name="trails",
                           hoverinfo="skip"))
    sp = np.array([s.pos for s in scn.survivors])
    sc = [(COLORS["alive"] if s.alive else COLORS["dead"]) if found.get(s.cell, np.inf) <= t else COLORS["unfound"]
          for s in scn.survivors]
    tr.append(go.Scatter3d(x=sp[:, 0], y=sp[:, 1], z=sp[:, 2] + 0.5, mode="markers",
                           marker=dict(size=4, symbol="diamond", color=sc), name="survivors"))
    return tr


def scene_figure(res, scn, title: str = "", n_frames: int = 60, caught_at: Optional[float] = None):
    if go is None:
        raise ImportError("install the dashboard extra: pip install -e .[dashboard]")
    sensor = res.config.sensor
    found = res.metrics.get("found_times", {})
    found = {int(k): v for k, v in found.items()}
    ks = np.linspace(0, len(res.t) - 1, n_frames).astype(int)
    static = [terrain_mesh(scn.terrain)]
    fig = go.Figure(data=static + _frame_traces(res, scn, ks[0], sensor, caught_at, found))
    fig.frames = [go.Frame(data=_frame_traces(res, scn, k, sensor, caught_at, found), traces=[1, 2, 3, 4], name=f"{res.t[k]:.1f}")
                  for k in ks]
    fig.update_layout(
        title=title, height=650, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35),
                   xaxis=dict(range=[0, scn.terrain.arena]), yaxis=dict(range=[0, scn.terrain.arena]),
                   zaxis=dict(range=[0, 40])),
        updatemenus=[dict(type="buttons", showactive=False, y=0, x=0.05, buttons=[
            dict(label="play", method="animate", args=[None, dict(frame=dict(duration=150), fromcurrent=True)]),
            dict(label="pause", method="animate", args=[[None], dict(mode="immediate")])])],
        sliders=[dict(steps=[dict(method="animate", label=f.name, args=[[f.name], dict(mode="immediate")])
                             for f in fig.frames], currentvalue=dict(prefix="t = "))])
    return fig


def coverage_figure(scn, cleared_masks: Dict[str, np.ndarray]):
    """End-of-run maps: one heatmap per entry (e.g. naive believed, gated known)."""
    n = scn.grid.n
    fig = make_subplots(rows=1, cols=len(cleared_masks), subplot_titles=list(cleared_masks))
    sp = np.array([s.pos for s in scn.survivors])
    for c, (name, m) in enumerate(cleared_masks.items(), start=1):
        z = m.reshape(n, n).T.astype(float)
        z[(~scn.coverable).reshape(n, n).T] = np.nan
        fig.add_trace(go.Heatmap(z=z, x0=scn.grid.origin[0], dx=scn.grid.size, y0=scn.grid.origin[1],
                                 dy=scn.grid.size, colorscale=[[0, "#fed7d7"], [1, "#c6f6d5"]], showscale=False),
                      row=1, col=c)
        fig.add_trace(go.Scatter(x=sp[:, 0], y=sp[:, 1], mode="markers", marker=dict(color="black", symbol="x"),
                                 showlegend=False), row=1, col=c)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig
