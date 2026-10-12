"""How compromised UAVs physically fly (D6).  This is physics, not messages, so it uses true states."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..safety import brake_command

INTERCEPT_LOOKAHEAD = 1.0  # seconds along the target's current velocity
INTERCEPT_GAIN = 2.0
UNLIMITED_INTERCEPT_SPEED = 5.0  # what "flat out" means when the config sets no speed limit


def adversary_commands(motion: str, u: np.ndarray, pos: np.ndarray, vel: np.ndarray, compromised: np.ndarray, dt: float,
                       max_speed: Optional[float], max_accel: Optional[float]) -> np.ndarray:
    """``u`` with the rows of compromised agents replaced according to ``cfg.adversary_motion``.

    ``same``       they fly the fleet's controller on their true state, as honest agents do
    ``hover``      they brake to zero velocity and stay there
    ``intercept``  they fly flat out at the point the nearest honest UAV will reach in one second
    """
    if motion == "same" or not compromised.any():
        return u
    u = u.copy()
    honest = np.flatnonzero(~compromised)
    for i in np.flatnonzero(compromised):
        if motion == "hover" or len(honest) == 0:
            u[i] = brake_command(vel[i], dt, max_accel)
            continue
        nearest = honest[np.argmin(np.linalg.norm(pos[honest] - pos[i], axis=1))]
        to_target = pos[nearest] + INTERCEPT_LOOKAHEAD * vel[nearest] - pos[i]
        speed = UNLIMITED_INTERCEPT_SPEED if max_speed is None else max_speed
        distance = np.linalg.norm(to_target)
        desired = np.zeros_like(to_target) if distance < 1e-12 else speed * to_target / distance
        u[i] = INTERCEPT_GAIN * (desired - vel[i])
    return u
