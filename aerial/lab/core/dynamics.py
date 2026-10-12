"""Double-integrator UAV kinematics:   p' = v,   v' = u,   p, v, u in R^d.

No aerodynamics and no rotor model (explicitly out of scope in the proposal).  Optional
speed / acceleration saturation keeps the commands physically plausible.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def clip_norm(x: np.ndarray, limit: Optional[float]) -> np.ndarray:
    """Scale each row vector of ``x`` so its Euclidean norm is at most ``limit``."""
    if limit is None:
        return x
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return x * scale


def step_double_integrator(
    pos: np.ndarray,
    vel: np.ndarray,
    u: np.ndarray,
    dt: float,
    max_speed: Optional[float] = None,
    max_accel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance all agents by one step with the control held constant (zero-order hold).

    Returns ``(new_pos, new_vel, applied_u)``.  Without saturation this is the *exact*
    solution of the continuous double integrator for piecewise-constant u.
    """
    u = clip_norm(u, max_accel)
    new_vel = clip_norm(vel + u * dt, max_speed)
    new_pos = pos + 0.5 * (vel + new_vel) * dt
    return new_pos, new_vel, u
