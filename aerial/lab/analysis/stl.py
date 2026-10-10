"""Offline quantitative STL robustness monitors, as thin wrappers over ``aerial.stl`` and ``aerial.specs_library``."""
from __future__ import annotations

import numpy as np

from ..core.simulator import SimResult
from ..safety import separation_margins
from ..specs_library import separation_spec, target_margins, target_spec
from ..stl import robustness


def separation_robustness(res: SimResult) -> np.ndarray:
    """Return ``min_j ||p_i-p_j|| - d_safe`` at every recorded time."""
    return separation_margins(res.pos, res.cfg.d_safe).min(axis=1)


def stl_safety_robustness(res: SimResult) -> float:
    """Quantitative robustness of ``G[0,T](pairwise separation >= d_safe)`` over all agents,
    compromised ones included: a collision is physical."""
    return float(robustness(separation_spec(), res)[0])


def target_robustness_series(res: SimResult) -> np.ndarray:
    """Per-agent margin of ``position in any target`` at every recorded time."""
    return target_margins(res)


def stl_target_robustness(res: SimResult) -> float:
    """Quantitative robustness of eventual target reach for every honest agent."""
    return float(robustness(target_spec(np.inf), res)[0])
