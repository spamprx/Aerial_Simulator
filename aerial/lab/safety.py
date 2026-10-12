"""What "safe" means (D2), and the never-gated terms that keep a UAV there.

``safety_margins`` is the one definition of Safe: the STL specs in ``specs_library`` grade runs with
it, so that a gate built on it and the grade it receives cannot disagree about what counts as safe.
``safe_for_agent`` is the same requirement asked of a plan, with what could be occupied in place of where
others are.  It is at least as strict everywhere, so a plan it passes meets ``safety_margins``.

The rest is the never-gated part of control: obstacle and geofence repulsion, and the braking hold.
A knowledge gate may withhold a controller's "advance" command, never these.  Each repulsion is
the gradient of the standard potential U = k/2 * (1/rho - 1/d)^2, which is active inside the
influence distance d and rises steeply towards the surface, capped at ``cap``.

A plain potential field reacts to where a UAV is and ignores how fast it arrives, but a double
integrator closing at speed w needs w^2 / (2 * a) of room to stop.  So rho here is the distance
that would be left after braking at ``cap``.  Pass the real ``cfg.max_accel`` as ``cap`` when
there is one, and give it priority over everything else with ``prioritise``: both assumptions
are what make the stopping distance true.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import gate_settings
from .core.dynamics import clip_norm, step_double_integrator
from .core.world import World

DEFAULT_CAP = 10.0


def separation_margins(pos: np.ndarray, d_safe: float) -> np.ndarray:
    """(..., N) distance from each agent to its nearest other agent, minus ``d_safe``; +inf if alone.

    ``pos`` is (N, d) or a history (K+1, N, d).  Every agent counts, compromised ones included: a
    collision is physical.
    """
    distance = np.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], axis=-1)
    n = pos.shape[-2]
    distance[..., np.arange(n), np.arange(n)] = np.inf
    return distance.min(axis=-1, initial=np.inf) - d_safe


def obstacle_margins(pos: np.ndarray, world: World) -> np.ndarray:
    """(..., N) signed distance from each agent to the nearest obstacle's surface; +inf if there are none."""
    centres, radii = world.obstacle_arrays
    return (np.linalg.norm(pos[..., None, :] - centres, axis=-1) - radii).min(axis=-1, initial=np.inf)


def geofence_margins(pos: np.ndarray, arena_size: float) -> np.ndarray:
    """(..., N) distance from each agent to the nearest wall of [0, arena_size]^d, negative outside."""
    return np.minimum(pos, arena_size - pos).min(axis=-1)


def safety_margins(pos: np.ndarray, world: World, d_safe: float) -> np.ndarray:
    """(..., N) the D2 predicate: separation, obstacle clearance and geofence all at once.

    An agent is safe where this is non-negative.
    """
    return np.minimum.reduce([separation_margins(pos, d_safe), obstacle_margins(pos, world),
                              geofence_margins(pos, world.arena_size)])


def predict_plan(pos: np.ndarray, vel: np.ndarray, u: np.ndarray, dt: float, steps: int,
                 max_speed: Optional[float] = None, max_accel: Optional[float] = None) -> np.ndarray:
    """(steps + 1, N, d) where each agent goes if it applies ``u`` for one step and then brakes as hard as
    it can (D3), on the same integrator as the real thing.  Row 0 is where it is now."""
    plan = np.empty((steps + 1, *pos.shape))
    plan[0] = pos
    position, velocity, _ = step_double_integrator(pos, vel, u, dt, max_speed, max_accel)
    plan[1] = position
    for s in range(2, steps + 1):
        position, velocity, _ = step_double_integrator(position, velocity, brake_command(velocity, dt, max_accel), dt,
                                                        max_speed, max_accel)
        plan[s] = position
    return plan


def occupied_clearance(d_safe: float, h: float, dim: int) -> float:
    """How far a plan must stay from an occupied cell's centre: a point in the cell can be h * sqrt(d) / 2
    from its centre, and must still be d_safe from the plan."""
    return d_safe + h * np.sqrt(dim) / 2.0


def plan_clear_of_world(plan_i: np.ndarray, world: World, d_safe: float) -> bool:
    """Whether a plan stays inside the geofence and at least ``d_safe`` from every obstacle."""
    if geofence_margins(plan_i, world.arena_size).min() < 0.0:
        return False
    if not world.obstacles:
        return True
    reach = float(np.linalg.norm(plan_i - plan_i[0], axis=1).max())
    if obstacle_margins(plan_i[0], world) > reach + d_safe:  # the usual case, and far cheaper
        return True
    ends = plan_i if len(plan_i) > 1 else np.vstack([plan_i, plan_i])  # one point is a path of no length
    return all(world.path_is_clear(ends[s], ends[s + 1], margin=d_safe) for s in range(len(ends) - 1))


def safe_for_agent(i: int, plan_i: np.ndarray, occupied_cells: np.ndarray, world: World, cfg,
                   h: Optional[float] = None) -> bool:
    """Whether agent i's plan is safe with these cells occupied.

    The plan must stay at least d_safe + h * sqrt(d) / 2 from the centre of every occupied cell, at least
    d_safe from every obstacle, and inside the geofence.  ``h`` defaults to the config's cell side.
    """
    plan_i = np.asarray(plan_i, dtype=float)
    if plan_i.ndim != 2:
        raise ValueError(f"agent {i}'s plan must have shape (steps, d), got {plan_i.shape}")
    h = gate_settings(cfg)["h"] if h is None else h
    if not plan_clear_of_world(plan_i, world, cfg.d_safe):
        return False
    occupied_cells = np.asarray(occupied_cells).reshape(-1, plan_i.shape[1])
    if len(occupied_cells) == 0:
        return True
    centres = (occupied_cells + 0.5) * h
    distance = np.linalg.norm(centres[:, None, :] - plan_i[None, :, :], axis=-1)
    return bool(distance.min() >= occupied_clearance(cfg.d_safe, h, plan_i.shape[1]))


def gate_clearance(d_safe: float, h: float, dim: int, max_speed: float, tau: float, eps: float = 0.0) -> float:
    """The distance beyond which a neighbour can never close the gate on a UAV at rest, however it falls
    on the grid and whether it is honest or not.  A moving UAV also needs the distance its plan covers.

    An honest neighbour's cells can reach h + max_speed * tau from its report, plus h * sqrt(d) / 2 to their
    centres, and the plan must clear those by ``occupied_clearance``.  If it might be faulty, its range shell
    reaches eps + max_speed * tau inside a range that may itself read eps short.
    """
    return d_safe + h * np.sqrt(dim) + max(h, 2.0 * eps) + max_speed * tau


def _repulsion(rho: float, d_influence: float, k: float, cap: float) -> float:
    """Magnitude of the potential's gradient at distance ``rho``."""
    if rho >= d_influence:
        return 0.0
    if rho <= 0.0:
        return cap
    return min(cap, k * (1.0 / rho - 1.0 / d_influence) / rho ** 2)


def _after_braking(rho: float, closing_speed: float, cap: float) -> float:
    """Distance left once a UAV closing at this speed has braked to a stop at ``cap``."""
    return rho - max(closing_speed, 0.0) ** 2 / (2.0 * cap)


def obstacle_repulsion(p_i: np.ndarray, v_i: np.ndarray, world: World, d_influence: float, k: float,
                       cap: float = DEFAULT_CAP) -> np.ndarray:
    """(d,) acceleration away from the nearest surface point of each obstacle in range."""
    total = np.zeros_like(p_i, dtype=float)
    for obstacle in world.obstacles:
        offset = p_i - np.asarray(obstacle.center)
        distance = float(np.linalg.norm(offset))
        normal = offset / distance if distance > 1e-12 else np.eye(len(p_i))[0]
        rho = _after_braking(distance - obstacle.radius, -float(v_i @ normal), cap)
        total += _repulsion(rho, d_influence, k, cap) * normal
    return clip_norm(total, cap)


def geofence_repulsion(p_i: np.ndarray, arena_size: float, margin: float, k: float,
                       v_i: Optional[np.ndarray] = None, cap: float = DEFAULT_CAP) -> np.ndarray:
    """(d,) acceleration back into the box [0, arena_size]^d, from every wall within ``margin``."""
    velocity = np.zeros_like(p_i, dtype=float) if v_i is None else v_i
    total = np.zeros_like(p_i, dtype=float)
    for d in range(len(p_i)):
        low = _after_braking(float(p_i[d]), -float(velocity[d]), cap)
        high = _after_braking(arena_size - float(p_i[d]), float(velocity[d]), cap)
        total[d] = _repulsion(low, margin, k, cap) - _repulsion(high, margin, k, cap)
    return clip_norm(total, cap)


def brake_command(vel: np.ndarray, dt: float, max_accel: Optional[float] = None) -> np.ndarray:
    """The hold of D3: the acceleration that stops the UAV, within its limit.  u = 0 only coasts."""
    return clip_norm(-np.asarray(vel, dtype=float) / dt, max_accel)


def prioritise(advance: np.ndarray, safety: np.ndarray, max_accel: Optional[float]) -> np.ndarray:
    """Scale ``advance`` to fit whatever authority ``safety`` leaves.  Works per row.

    Saturating the plain sum would shrink both together, and a goal pulling into an obstacle
    could then cancel part of the repulsion that is meant to stop it.
    """
    if max_accel is None:
        return advance
    room = np.maximum(max_accel - np.linalg.norm(safety, axis=-1, keepdims=True), 0.0)
    return clip_norm(advance, room)


def separation_repulsion(p_i: np.ndarray, others: np.ndarray, r_sep: float, k: float, cap: float = DEFAULT_CAP,
                         v_i: Optional[np.ndarray] = None) -> np.ndarray:
    """(d,) sum of k * (p_i - p_j) / r^2 * (1 - r / r_sep) over the reports with r < ``r_sep``,
    where r = ||p_i - p_j||.

    Close up this is the lecture's (p_i - p_j) / ||p_i - p_j||^2.  The last factor brings it to
    zero at ``r_sep``: cut off abruptly, the force jumps by k / r_sep there, agents settle on that
    edge and chatter, and a flock's velocities never align.

    With ``v_i`` given, r is what would be left after this agent braked, as for obstacles.  That uses
    only its own true velocity.  A neighbour's velocity is just a report, and this term is not
    filtered, so trusting it would let a liar claiming a huge closing speed seize every agent's
    control authority from far away.  Honest neighbours each allow for their own stop.

    It takes every position report as given: a collision is physical, so a neighbour that might be
    lying about being close is still treated as close.
    """
    away = p_i - others
    distance = np.linalg.norm(away, axis=1)
    normal = away / np.maximum(distance[:, None], 1e-12)
    rho = distance
    if v_i is not None:
        closing = np.maximum(-(normal @ v_i), 0.0)
        rho = distance - closing ** 2 / (2.0 * cap)
    magnitude = np.full(len(rho), cap)
    positive = rho > 0.0
    magnitude[positive] = np.minimum(k * (1.0 - rho[positive] / r_sep) / rho[positive], cap)
    magnitude[rho >= r_sep] = 0.0
    return clip_norm((magnitude[:, None] * normal).sum(axis=0), cap)


def govern_speed(advance: np.ndarray, v_i: np.ndarray, limit: float, gain: float = 2.0) -> np.ndarray:
    """Stop ``advance`` adding speed past ``limit``, and brake back down to it if already over.

    Stopping distance grows with the square of speed, so this is what keeps the repulsion terms
    able to do their job.  ``advance`` may still turn the UAV.
    """
    speed = float(np.linalg.norm(v_i))
    if speed <= limit:
        return advance
    heading = v_i / speed
    return advance - max(float(advance @ heading), 0.0) * heading - gain * (speed - limit) * heading


def steer_around(advance: np.ndarray, p_i: np.ndarray, centres: np.ndarray, clearances: np.ndarray,
                 influence: float) -> np.ndarray:
    """Turn the part of ``advance`` that heads into a nearby body so that it slides past instead.

    Repulsion alone stops a UAV dead in front of an obstacle it is aimed straight at, and two UAVs
    head-on stop each other: the usual potential-field local minimum.  Within ``influence`` of a
    body's surface, this blends ``advance`` towards the direction along the surface with the same
    magnitude, fully so at the surface.  Dead-on there is no such direction, so it takes the
    normal turned by 90 degrees; two UAVs head-on then both turn to their own side and pass.

    ``clearances`` is the distance from each centre at which the surface lies.  This changes only
    the advance command, so a gate that withholds advance also withholds the steering.
    """
    magnitude = float(np.linalg.norm(advance))
    if magnitude < 1e-12:
        return advance
    steered = advance
    for centre, clearance in zip(centres, clearances):
        offset = p_i - centre
        distance = float(np.linalg.norm(offset))
        weight = 1.0 - (distance - clearance) / influence
        if distance < 1e-12 or weight <= 0.0:
            continue
        normal = offset / distance
        into = -float(steered @ normal)
        if into <= 0.0:
            continue
        along = steered + into * normal
        length = float(np.linalg.norm(along))
        if length < 1e-3 * magnitude:
            along = np.zeros_like(normal)
            along[:2] = -normal[1], normal[0]
            length = float(np.linalg.norm(along))
            if length < 1e-12:  # the normal is straight up or down
                along = np.eye(len(normal))[0]
                length = 1.0
        steered = (1.0 - min(weight, 1.0)) * steered + min(weight, 1.0) * along * (magnitude / length)
    return steered
