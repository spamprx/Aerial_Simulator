"""3D double integrator with separate horizontal/vertical limits, plus first-order yaw (T0.6)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def clip_norm(x: np.ndarray, limit: Optional[float]) -> np.ndarray:
    """Scale each row of ``x`` so its norm is at most ``limit``."""
    if limit is None:
        return x
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    scale = np.where(n > limit, limit / np.maximum(n, 1e-12), 1.0)
    return x * scale


def clip_hv(x: np.ndarray, h_limit: float, v_limit: float) -> np.ndarray:
    """Clip the horizontal (x, y) norm and the vertical (z) magnitude separately."""
    out = x.copy()
    out[..., :2] = clip_norm(x[..., :2], h_limit)
    out[..., 2] = np.clip(x[..., 2], -v_limit, v_limit)
    return out


def step_double_integrator(pos: np.ndarray, vel: np.ndarray, u: np.ndarray, dt: float, a_max: float,
                           v_h: float, v_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One tick: u <- sat(u, a_max); v' = sat(v + u dt); p' = p + (v + v')/2 dt.

    Exact under zero-order hold when nothing saturates, trapezoidal otherwise. Returns (pos', vel', u_applied).
    """
    u = clip_norm(u, a_max)
    v_new = clip_hv(vel + u * dt, v_h, v_z)
    p_new = pos + 0.5 * (vel + v_new) * dt
    return p_new, v_new, u


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_yaw(yaw: np.ndarray, yaw_rate_cmd: np.ndarray, dt: float, max_rate: float) -> np.ndarray:
    return wrap_angle(yaw + np.clip(yaw_rate_cmd, -max_rate, max_rate) * dt)
