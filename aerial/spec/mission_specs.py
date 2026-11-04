"""Mission STL specifications (T4.1), graded offline with the ported robustness engine.

The safety predicate is the single Safe predicate of D2, written as signed margins on the recorded trace:

- phi_safe  = G[0,T]( min_pair_sep - d_safe >= 0  and  clearance - d_obs_floor >= 0  and  inside arena and band )
- phi_find  = F[0,T_find]( detected(s) ),  one per survivor; margin = +1 once detected, -1 before
- phi_know  = F[0,T_k]( every honest drone knows every liar ),  needs the ``known_faulty`` extra
- phi_cover = F[0,T]( known_cleared_frac - 0.95 * coverable >= 0 )

``safe_margins`` is shared with the analysis code so that the grade and the metrics agree (D2).
"""
from __future__ import annotations

import numpy as np

from .stl import Always, And, Eventually, Pred, prefix_robustness, robustness


def sep_margin(res) -> np.ndarray:
    pos = res.pos
    d = np.linalg.norm(pos[:, :, None] - pos[:, None, :], axis=-1)
    n = pos.shape[1]
    d[:, np.arange(n), np.arange(n)] = np.inf
    return d.min(axis=(1, 2)) - res.config.gate.d_safe


def clearance_margin(res, terrain) -> np.ndarray:
    """Vertical clearance over the terrain column under each drone (collision if negative)."""
    h = terrain.height_at(res.pos[..., :2])
    return (res.pos[..., 2] - h).min(axis=1)


def geofence_margin(res) -> np.ndarray:
    c = res.config
    p = res.pos
    m = np.minimum.reduce([p[..., 0].min(1), p[..., 1].min(1), c.world.arena - p[..., 0].max(1),
                           c.world.arena - p[..., 1].max(1)])
    return m


def phi_safe(terrain, T: float):
    return Always(0, T, And(Pred(sep_margin), Pred(lambda r: clearance_margin(r, terrain)), Pred(geofence_margin)))


def phi_find(found_time: float, T_find: float):
    """F[0,T_find](detected): margin +1 from the detection time on."""
    return Eventually(0, T_find, Pred(lambda r: np.where(r.t >= found_time, 1.0, -1.0)))


def phi_know(T_k: float):
    def margin(r):
        kf = r.extras.get("known_faulty")
        liars = np.flatnonzero(r.liars)
        honest = np.flatnonzero(~r.liars)
        if kf is None or len(liars) == 0:
            return np.ones(len(r.t))
        knows = kf[:, honest][:, :, liars].all(axis=(1, 2)).astype(float) * 2 - 1
        return np.concatenate([[-1.0], knows])[: len(r.t)]
    return Eventually(0, T_k, Pred(margin))


def phi_cover(T: float, target: float):
    def margin(r):
        kc = r.extras.get("known_cleared_frac")
        honest = np.flatnonzero(~r.liars)
        if kc is None:
            return -np.ones(len(r.t))
        m = kc[:, honest].max(axis=1) - 0.95 * target
        return np.concatenate([[m[0]], m])[: len(r.t)]
    return Eventually(0, T, Pred(margin))


def grade(res, scn) -> dict:
    """Robustness rho of every spec at t = 0, plus rho_safe(t) for the dashboard."""
    T = float(res.t[-1])
    safe = phi_safe(scn.terrain, T)
    out = {"rho_safe": float(robustness(safe, res)[0]),
           "rho_safe_t": prefix_robustness(safe, res)}
    ft = res.metrics.get("found_times", {})
    finds = [float(robustness(phi_find(ft.get(s.cell, np.inf), T), res)[0]) for s in scn.survivors]
    out["rho_find_min"] = min(finds) if finds else np.nan
    out["rho_know"] = float(robustness(phi_know(T), res)[0])
    out["rho_cover"] = float(robustness(phi_cover(T, float(scn.coverable.mean())), res)[0])
    return out
