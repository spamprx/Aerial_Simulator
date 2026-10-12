"""The mission's requirements as STL formulas, built on the predicate that ``aerial.safety`` defines (D2).

Each ``*_spec`` returns a formula for the engine in ``aerial.stl``, reusable across runs:

    separation_spec(d_safe)        G[0,T] min over pairs ||p_i - p_j|| - d_safe
    obstacle_spec()                G[0,T] min clearance over agents and obstacles
    geofence_spec()                G[0,T] min distance to the arena's walls, over agents
    safety_spec(d_safe)            G[0,T] the whole D2 predicate: the conjunction of the three above
    target_spec(T1, assignment)    and over honest i of F[0,T1](p_i in S_target(i))
    mission_spec(T1, assignment)   safety and targets together

Safety covers every agent and targets only the honest ones (D9).  T is the end of the run.  Each
``*_by_agent`` gives the same requirement agent by agent as (K+1, N), for the dashboard, and its minimum
over agents is the fleet's value.  ``prefix=True`` asks for the rho(t) curve instead of rho(phi, k).

For a curve whose zero line marks violations, plot ``safety_spec``.  The curve of ``target_spec``, and
so of ``mission_spec``, stays negative until the last honest agent arrives, however well the run is going.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np

from .safety import geofence_margins, obstacle_margins, safety_margins, separation_margins
from .stl import Always, And, Deferred, Eventually, Formula, Pred, prefix_robustness, robustness


def _separation(d_safe: Optional[float], res) -> np.ndarray:
    return separation_margins(res.pos, res.cfg.d_safe if d_safe is None else d_safe)


def _obstacle(res) -> np.ndarray:
    return obstacle_margins(res.pos, res.world)


def _geofence(res) -> np.ndarray:
    return geofence_margins(res.pos, res.world.arena_size)


def _safety(d_safe: Optional[float], res) -> np.ndarray:
    return safety_margins(res.pos, res.world, res.cfg.d_safe if d_safe is None else d_safe)


def target_margins(res, assignment: Optional[Sequence] = None) -> np.ndarray:
    """(K+1, N) how far inside S_target(i) each agent is, negative outside.

    ``assignment`` has one entry per agent: a target's index into ``world.targets``, a list of them
    (being in any one will do), or None or an empty list if that agent has nothing to reach, which
    gives +inf.  With no assignment at all, S_target(i) is every target for every agent, and -inf
    if the world has none.
    """
    targets = res.world.targets
    shape = res.pos.shape[:2]
    each = np.stack([target.radius - np.linalg.norm(res.pos - np.asarray(target.center), axis=-1)
                     for target in targets]) if targets else np.empty((0, *shape))
    if assignment is None:
        return each.max(axis=0) if targets else np.full(shape, -np.inf)
    if len(assignment) != shape[1]:
        raise ValueError(f"assignment must have one entry per agent ({shape[1]}), got {len(assignment)}")
    margins = np.full(shape, np.inf)
    for i, chosen in enumerate(assignment):
        chosen = np.atleast_1d([] if chosen is None else chosen).astype(int)
        if len(chosen) and (chosen.min() < 0 or chosen.max() >= len(targets)):
            raise ValueError(f"agent {i} is assigned targets {chosen.tolist()}, but the world has {len(targets)}")
        if len(chosen):
            margins[:, i] = each[chosen, :, i].max(axis=0)
    return margins


def _column(margins: np.ndarray, i: int) -> Pred:
    return Pred(lambda _res: margins[:, i])


def _worst(margins: Callable, res) -> np.ndarray:
    return margins(res).min(axis=1)


def _all_honest_reach(T1: float, assignment: Optional[Sequence], res) -> Formula:
    # With nobody honest the mission cannot be finished, as completion_time also reports, whereas a
    # bare conjunction over nobody would call it satisfied.
    if not res.honest.any():
        return Pred(lambda _res: np.full(len(_res.t), -np.inf))
    margins = target_margins(res, assignment)
    return And(*[Eventually(0.0, T1, _column(margins, i)) for i in np.flatnonzero(res.honest)])


def separation_spec(d_safe: Optional[float] = None) -> Formula:
    """Nobody ever closer than ``d_safe``, which defaults to the run's ``cfg.d_safe``."""
    return Always(0.0, np.inf, Pred(partial(_worst, partial(_separation, d_safe))))


def obstacle_spec() -> Formula:
    """Nobody ever inside an obstacle."""
    return Always(0.0, np.inf, Pred(partial(_worst, _obstacle)))


def geofence_spec() -> Formula:
    """Nobody ever outside the arena."""
    return Always(0.0, np.inf, Pred(partial(_worst, _geofence)))


def safety_spec(d_safe: Optional[float] = None) -> Formula:
    """Safe throughout, in the one sense of D2 that the gate also uses."""
    return Always(0.0, np.inf, Pred(partial(_worst, partial(_safety, d_safe))))


def target_spec(T1: float, assignment: Optional[Sequence] = None) -> Formula:
    """Every honest agent reaches its target within ``T1`` seconds.  See ``target_margins``."""
    return Deferred(partial(_all_honest_reach, T1, assignment))


def mission_spec(T1: float, assignment: Optional[Sequence] = None, d_safe: Optional[float] = None) -> Formula:
    """Safe throughout, and every honest agent at its target by ``T1``."""
    return And(separation_spec(d_safe), obstacle_spec(), geofence_spec(), target_spec(T1, assignment))


# ---- agent by agent, as (K+1, N) ---------------------------------------------------------------
def _by_agent(res, margins: np.ndarray, operator: Callable, prefix: bool) -> np.ndarray:
    evaluate = prefix_robustness if prefix else robustness
    return np.stack([evaluate(operator(_column(margins, i)), res) for i in range(margins.shape[1])], axis=1)


_always = partial(Always, 0.0, np.inf)


def separation_by_agent(res, d_safe: Optional[float] = None, prefix: bool = False) -> np.ndarray:
    return _by_agent(res, _separation(d_safe, res), _always, prefix)


def obstacle_by_agent(res, prefix: bool = False) -> np.ndarray:
    return _by_agent(res, _obstacle(res), _always, prefix)


def geofence_by_agent(res, prefix: bool = False) -> np.ndarray:
    return _by_agent(res, _geofence(res), _always, prefix)


def safety_by_agent(res, d_safe: Optional[float] = None, prefix: bool = False) -> np.ndarray:
    return _by_agent(res, _safety(d_safe, res), _always, prefix)


def target_by_agent(res, T1: float, assignment: Optional[Sequence] = None, prefix: bool = False) -> np.ndarray:
    """A compromised agent has no target to reach, so its column is +inf.  With nobody honest, then, this
    array is all +inf while ``target_spec`` is -inf: the one case where the minimum is not the fleet's value."""
    result = _by_agent(res, target_margins(res, assignment), partial(Eventually, 0.0, T1), prefix)
    result[:, res.compromised] = np.inf
    return result


def mission_by_agent(res, T1: float, assignment: Optional[Sequence] = None, d_safe: Optional[float] = None,
                     prefix: bool = False) -> np.ndarray:
    return np.minimum(safety_by_agent(res, d_safe, prefix), target_by_agent(res, T1, assignment, prefix))
