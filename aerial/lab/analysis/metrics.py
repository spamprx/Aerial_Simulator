"""Scalar and time-series metrics computed from a SimResult.

The mission-level metrics (completion time, hesitation time, target coverage) and the STL
robustness live with the scenario / specification modules; these are the base-simulator
ones every experiment needs.  Each ``*_series`` returns one value per recorded time step.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..core.simulator import SimResult
from ..safety import separation_margins
from .mission import MISSION_METRICS, safety_violation_steps


def _select(res: SimResult, agents: str) -> np.ndarray:
    return res.honest if agents == "honest" else np.ones_like(res.honest)


def position_spread_series(res: SimResult, agents: str = "honest") -> np.ndarray:
    """RMS distance of the selected agents from their centroid (0 = rendezvous reached)."""
    P = res.pos[:, _select(res, agents), :]
    centred = P - P.mean(axis=1, keepdims=True)
    return np.sqrt((centred ** 2).sum(axis=-1).mean(axis=1))


def velocity_spread_series(res: SimResult, agents: str = "honest") -> np.ndarray:
    """RMS deviation of velocities from the mean velocity (0 = velocity alignment)."""
    V = res.vel[:, _select(res, agents), :]
    centred = V - V.mean(axis=1, keepdims=True)
    return np.sqrt((centred ** 2).sum(axis=-1).mean(axis=1))


def formation_error_series(res: SimResult, offsets: np.ndarray, agents: str = "honest") -> np.ndarray:
    """RMS of ``(p_i - c_i) - mean(p - c)`` over the selected agents (0 = in formation)."""
    mask = _select(res, agents)
    corrected = res.pos[:, mask, :] - np.asarray(offsets)[mask]
    centred = corrected - corrected.mean(axis=1, keepdims=True)
    return np.sqrt((centred ** 2).sum(axis=-1).mean(axis=1))


def min_separation_series(res: SimResult, agents: str = "all") -> np.ndarray:
    """Smallest pairwise distance at each step (collision safety uses *all* agents)."""
    return separation_margins(res.pos[:, _select(res, agents), :], 0.0).min(axis=1, initial=np.inf)


def time_to_converge(series: np.ndarray, t: np.ndarray, tol: float) -> float:
    """First time after which the series stays below ``tol``; inf if it never settles."""
    below = series <= tol
    if not below[-1]:
        return float("inf")
    bad = np.flatnonzero(~below)
    return float(t[0] if len(bad) == 0 else t[bad[-1] + 1])


def decay_rate(series: np.ndarray, t: np.ndarray, high: float = 1e-1, low: float = 1e-9) -> float:
    """Slope per second of log(series) while it falls from ``high`` to ``low`` of its first value,
    which leaves out the initial transient and numerical noise.  NaN if it never gets that far."""
    window = (series < high * series[0]) & (series > low * series[0])
    if window.sum() < 20:
        return float("nan")
    return -float(np.polyfit(t[window], np.log(series[window]), 1)[0])


def f_local_fraction(res: SimResult) -> float:
    """Fraction of steps on which no honest agent heard more than F compromised agents."""
    held = res.extras.get("f_local_ok")
    return float("nan") if held is None or len(held) == 0 else float(np.mean(held))


# ----------------------------------------------------------------------------------
Metric = Callable[[SimResult], float]

DEFAULT_METRICS: Dict[str, Metric] = {
    "final_pos_spread": lambda r: float(position_spread_series(r)[-1]),
    "final_vel_spread": lambda r: float(velocity_spread_series(r)[-1]),
    "t_converge_pos": lambda r: time_to_converge(position_spread_series(r), r.t, 0.1),
    "min_separation": lambda r: float(min_separation_series(r).min()),
    "mean_lambda2": lambda r: float(np.nanmean(r.lambda2)) if np.isfinite(r.lambda2).any() else float("nan"),
    "f_local_fraction": f_local_fraction,
    "connected_fraction": lambda r: float(np.mean(r.lambda2 > 1e-9)) if np.isfinite(r.lambda2).any() else float("nan"),
    **MISSION_METRICS,
}
