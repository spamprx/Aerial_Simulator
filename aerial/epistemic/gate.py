"""The motion gate (T3.12, part a): advance only if K_i(Safe(plan)).

The plan is the next tick of the commanded acceleration followed by braking until tau. For every neighbour and every
status it can have in some world of H_i:

- status honest: a ball of radius 1/2 a_max t^2 (+eps) around p_hat + v_hat t (honest reports are true);
- status faulty: the UWB shell |x - p_i| in [r - w, r + w], w = eps' + v_max_3d (age dt + dt + tau), minus what the
  camera has just seen empty in the direction of motion;
- drones silent for more than L ticks are out of reach (the silence rule, the single probabilistic assumption).

Faulty shells are checked analytically: the plan stays within ``reach`` of p_i, so a shell whose inner radius exceeds
reach + d_safe is harmless. Otherwise the shell point closest to the plan end must be seen empty.
"""
from __future__ import annotations

import numpy as np


def plan_points(pos, vel, u, dt, tau, a_max, n=6):
    v1 = vel + u * dt
    p1 = pos + 0.5 * (vel + v1) * dt
    sp = np.linalg.norm(v1)
    tb = min(tau, sp / a_max) if a_max > 0 else tau
    ts = np.linspace(0, tb, n)
    dirn = v1 / sp if sp > 1e-9 else np.zeros(3)
    pts = p1 + dirn * (sp * ts - 0.5 * a_max * ts ** 2)[:, None]
    return np.vstack([pos, pts])


def knows_safe(obs, u, status_honest, status_faulty, cfg, silent_ticks: int):
    """Return (safe, blocker_id or None)."""
    g, dr = cfg.gate, cfg.drone
    dt = obs.static.dt
    me = obs.own.pos
    pts = plan_points(me, obs.own.vel, u, dt, g.tau, dr.a_max)
    reach = float(np.max(np.linalg.norm(pts - me, axis=1)))
    eps = max(g.eps, 4 * cfg.sensor.range_sigma)
    vis = obs.frame.visible if obs.frame is not None else None
    for j, b in obs.beacons.items():
        if b.age > silent_ticks:
            continue
        if status_honest[j]:
            t = b.age * dt + dt + g.tau
            centre = b.claim.pos + b.claim.vel * t
            rad = 0.5 * dr.a_max * t * t + eps
            if np.min(np.linalg.norm(pts - centre, axis=1)) - rad < g.d_safe:
                return False, j
        if status_faulty[j]:
            r = b.uwb_range if b.uwb_range is not None else float(np.linalg.norm(b.claim.pos - me))
            if b.uwb_range is None:
                continue          # no physical evidence beyond silence: covered by the silence rule
            w = eps + dr.v_max_3d * (b.age * dt + dt + g.tau)
            if r - w - reach >= g.d_safe:
                continue
            end = pts[-1] - me
            nrm = np.linalg.norm(end)
            if nrm < 1e-6 or vis is None:
                return False, j
            probe = me + end / nrm * max(r - w, 0.1)
            seen = vis.contains(np.array([probe, me + end / nrm * (reach + g.d_safe)]), margin=0.5)
            if not seen.all():
                return False, j
    return True, None
