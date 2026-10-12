"""Offline mission metrics, one number per run.  Mission metrics cover honest agents and safety covers everyone (D9)."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..core.simulator import SimResult
from ..safety import obstacle_margins, safety_margins
from ..specs_library import safety_spec
from ..stl import robustness
from .stl import stl_safety_robustness, stl_target_robustness, target_robustness_series

DEFAULT_R_DETECT = 4.0  # metres: the survivor sensor's radius, for a run whose mission does not state one
COVERAGE_CELL = 1.0  # metres: the side of the cells ``area_coverage`` counts


def target_coverage_series(res: SimResult) -> np.ndarray:
    """Fraction of honest agents inside a target at each recorded time."""
    if not res.world.targets or not res.honest.any():
        return np.zeros(len(res.t))
    reached = target_robustness_series(res)[:, res.honest] >= 0.0
    return reached.mean(axis=1)


def arrival_times(res: SimResult) -> np.ndarray:
    """(N,) first time each agent finished; infinity if it never did.

    Finishing means whatever the controller logged as ``mission_done`` (visiting an assigned
    target or sector), or else being inside any target.
    """
    done = res.extras.get("mission_done")
    if done is not None:
        return np.where(done.any(axis=0), res.t[:-1][done.argmax(axis=0)], np.inf)
    reached = target_robustness_series(res) >= 0.0
    return np.where(reached.any(axis=0), res.t[reached.argmax(axis=0)], np.inf)


def completion_time(res: SimResult) -> float:
    """First time by which every honest agent has finished; infinity if unfinished.

    An agent that has arrived stays counted after it moves on.  With no ``mission_done`` log,
    this is finite exactly when ``stl_target_robustness`` is non-negative.
    """
    arrivals = arrival_times(res)[res.honest]
    return float(arrivals.max()) if len(arrivals) else float("inf")


def obstacle_clearance_series(res: SimResult) -> np.ndarray:
    """Minimum signed distance to an obstacle over all agents at each time."""
    return obstacle_margins(res.pos, res.world).min(axis=1)


def safety_robustness(res: SimResult) -> float:
    """rho of the whole of D2 for the run: separation, obstacle clearance and the geofence.  Negative if it ever failed."""
    return float(robustness(safety_spec(), res)[0])


def safety_violation_steps(res: SimResult) -> int:
    """Recorded samples on which some agent breaks D2: too close to another, inside an obstacle or outside the fence."""
    return int((safety_margins(res.pos, res.world, res.cfg.d_safe).min(axis=1) < 0.0).sum())


def safety_violation_time(res: SimResult) -> float:
    """``safety_violation_steps`` in seconds."""
    return safety_violation_steps(res) * res.cfg.dt


def n_collisions(res: SimResult) -> int:
    """Distinct pairs that were ever closer than d_safe.  A pair that parts and meets again still counts once."""
    distance = np.linalg.norm(res.pos[:, :, None, :] - res.pos[:, None, :, :], axis=-1)
    below = (distance < res.cfg.d_safe).any(axis=0)
    return int(np.triu(below, k=1).sum())


def hesitation_time(res: SimResult, acceleration_tolerance: float = 1e-12) -> float:
    """Mean time per honest UAV spent held before that UAV finished.

    Held means ``gate_open`` is False.  With no gate log, it means a near-zero command, a proxy which on a double
    integrator cannot tell a hold from coasting.  Time after a UAV has finished costs its mission nothing, so it does
    not count, in either case.
    """
    honest = res.honest
    if len(res.u) == 0 or not honest.any():
        return 0.0
    if "gate_open" in res.extras:
        held = ~res.extras["gate_open"]
    elif res.world.targets or "mission_done" in res.extras:
        held = np.linalg.norm(res.u, axis=-1) <= acceleration_tolerance
    else:
        return 0.0
    active = res.t[:-1, None] < arrival_times(res)
    return float(np.sum((active & held)[:, honest]) * res.cfg.dt / honest.sum())


def n_requeries(res: SimResult) -> int:
    """How many re-queries honest agents asked for over the run, each one a step's request to every neighbour in range."""
    asked = res.extras.get("requery")
    return 0 if asked is None else int(asked[:, res.honest].sum())


def _detection_radius(res: SimResult) -> float:
    return float(res.cfg.mission.get("r_detect", DEFAULT_R_DETECT))


def survivors_found(res: SimResult) -> float:
    """Fraction of survivors (``world.targets``) that came within r_detect of an honest UAV.  NaN if there are none."""
    targets = res.world.targets
    if not targets:
        return float("nan")
    if not res.honest.any():
        return 0.0
    centres = np.array([target.center for target in targets])
    distance = np.linalg.norm(res.pos[:, res.honest, None, :] - centres, axis=-1)
    return float((distance.min(axis=(0, 1)) <= _detection_radius(res)).mean())


def area_coverage(res: SimResult) -> float:
    """Fraction of the arena's cells, seen from above, whose centre came within r_detect of an honest UAV.

    The cells are ``COVERAGE_CELL`` square over x-y, and height is ignored, since a sensor looks down.  Cells under a
    building count like any other: the sensor sweeps over them, and leaving them out would need a sensor model.
    """
    if not res.honest.any():
        return 0.0
    side = max(int(np.ceil(res.cfg.arena_size / COVERAGE_CELL)), 1)
    axis = (np.arange(side) + 0.5) * res.cfg.arena_size / side
    centres = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    # A UAV lingers, so many samples repeat; rounding finer than the cells only drops those.
    flown = np.unique(np.round(res.pos[:, res.honest, :2].reshape(-1, 2) * 20.0 / COVERAGE_CELL), axis=0) * COVERAGE_CELL / 20.0
    reach = _detection_radius(res) ** 2
    covered = np.zeros(len(centres), dtype=bool)
    for start in range(0, len(flown), 512):
        offset = centres[~covered, None, :] - flown[start:start + 512][None, :, :]
        covered[~covered] = ((offset ** 2).sum(axis=-1) <= reach).any(axis=1)
    return float(covered.mean())


def honest_msgs_discarded_frac(res: SimResult) -> float:
    """Of the honest reports honest agents received, the fraction W-MSR threw away on at least one coordinate.

    This is the "filtering makes it ignorant" number of T2.7.  NaN if nothing was filtered or nothing received.
    """
    rejected = res.extras.get("rejected_pos")
    if rejected is None or len(rejected) != len(res.adj):
        return float("nan")
    honest = np.flatnonzero(res.honest)
    among = np.ix_(np.arange(len(res.adj)), honest, honest)
    received = np.count_nonzero(res.adj[among])
    return float(np.count_nonzero(rejected[among]) / received) if received else float("nan")


MISSION_METRICS: Dict[str, Callable[[SimResult], float]] = {
    "stl_safety_robustness": stl_safety_robustness,
    "stl_target_robustness": stl_target_robustness,
    "safety_robustness": safety_robustness,
    "completion_time": completion_time,
    "target_coverage": lambda res: float(target_coverage_series(res)[-1]),
    "obstacle_clearance": lambda res: float(obstacle_clearance_series(res).min()),
    "hesitation_time": hesitation_time,
    "safety_violation_steps": lambda res: float(safety_violation_steps(res)),
    "safety_violation_time": safety_violation_time,
    "n_collisions": lambda res: float(n_collisions(res)),
    "n_requeries": lambda res: float(n_requeries(res)),
    "survivors_found": survivors_found,
    "area_coverage": area_coverage,
    "honest_msgs_discarded_frac": honest_msgs_discarded_frac,
}


def mission_metrics(res: SimResult) -> Dict[str, float]:
    """Return the report metrics for one recorded run."""
    return {name: metric(res) for name, metric in MISSION_METRICS.items()}
