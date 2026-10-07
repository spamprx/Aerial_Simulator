"""Formation flying: every agent holds an offset from a common, agreed reference point."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np

from ..registry import controller_registry
from ..resilience.aggregators import FilterLog, build_aggregator
from .base import SplitController
from .consensus import default_damping

FORMATION_SHAPES = ("line", "wedge", "grid", "circle")


def formation_offsets(shape: str, n: int, dim: int, spacing: float = 2.0) -> np.ndarray:
    """(n, dim) offsets for a named shape, centred on the origin, flat in z.

    The closest pair of agents is exactly ``spacing`` apart in every shape.
    """
    if shape not in FORMATION_SHAPES:
        raise ValueError(f"unknown formation shape '{shape}', choose from {FORMATION_SHAPES}")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    k = np.arange(n)
    if shape == "line":
        xy = np.column_stack([k * spacing, np.zeros(n)])
    elif shape == "wedge":  # agent 0 at the apex, the others alternating down the two arms
        arm = (k + 1) // 2
        side = np.where(k % 2 == 1, 1.0, -1.0)
        step = spacing / np.sqrt(2.0)
        xy = np.column_stack([-arm * step, side * arm * step])
    elif shape == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        xy = np.column_stack([(k % cols) * spacing, (k // cols) * spacing])
    else:  # circle: neighbours on the circle are one chord, spacing, apart
        radius = spacing / (2.0 * np.sin(np.pi / n)) if n > 1 else 0.0
        angle = 2.0 * np.pi * k / max(n, 1)
        xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    offsets = np.zeros((n, dim))
    offsets[:, :2] = xy - xy.mean(axis=0)
    return offsets


def resolve_offsets(offsets: Union[str, Sequence], n: int, dim: int, spacing: float) -> np.ndarray:
    """Turn a shape name or an explicit (n, dim) list into an offset array."""
    if isinstance(offsets, str):
        return formation_offsets(offsets, n, dim, spacing)
    resolved = np.array(offsets, dtype=float)
    if resolved.shape != (n, dim):
        raise ValueError(f"offsets must have shape ({n}, {dim}), got {resolved.shape}")
    return resolved


@controller_registry.register("formation")
class Formation(SplitController):
    """Consensus on the offset-corrected positions and on the velocities:

        u_i = k_p * (agg({p~_j - c_j}) - (p_i - c_i))  +  k_v * (agg({v~_j}) - v_i)

    The fleet agrees where the formation is, not where it should go, so without a leader it
    settles wherever consensus takes it.  Neighbour weights are normalised (D11), so the gains
    stay stable whatever the fleet size.  ``filter`` picks the aggregator, and W-MSR's ``f``
    defaults to ``cfg.F``.  After ``reset``, ``self.offsets`` holds the resolved (N, d) array.

    ``damping="own"`` replaces the velocity term with -k_v * v_i, and is the default under W-MSR, for
    the reason given on ``SecondOrderConsensus``.
    """

    DAMPING = ("relative", "own")

    def __init__(self, offsets: Union[str, Sequence] = "line", spacing: float = 2.0, k_p: float = 1.0,
                 k_v: float = 1.5, filter: str = "mean", f: Optional[int] = None, damping: Optional[str] = None):
        damping = default_damping(filter) if damping is None else damping
        if damping not in self.DAMPING:
            raise ValueError(f"damping must be one of {self.DAMPING}")
        self.offsets_spec, self.spacing = offsets, spacing
        self.k_p, self.k_v = k_p, k_v
        self.filter, self.f, self.damping = filter, f, damping
        self.offsets = None
        self.aggregator = None
        self.log = None

    def reset(self, sim) -> None:
        self.offsets = resolve_offsets(self.offsets_spec, sim.n, sim.dim, self.spacing)
        self.aggregator = build_aggregator(self.filter, sim.cfg.F if self.f is None else self.f)
        self.log = FilterLog(sim.n)

    def normalised_gains(self):
        return self.k_p, self.k_v

    def filter_offsets(self):
        return self.offsets

    def formation_term(self, obs, i: int) -> np.ndarray:
        ids, pos, vel = obs.received(i)
        own = obs.pos[i] - self.offsets[i]
        position = self.aggregator.aggregate(own, pos - self.offsets[ids], ids)
        self.log.record("pos", i, position)
        if self.damping == "own":
            return self.k_p * (position.value - own) - self.k_v * obs.vel[i]
        velocity = self.aggregator.aggregate(obs.vel[i], vel, ids)
        self.log.record("vel", i, velocity)
        return self.k_p * (position.value - own) + self.k_v * (velocity.value - obs.vel[i])

    def control_parts(self, obs):
        self.log.clear()
        advance = np.array([self.formation_term(obs, i) for i in range(obs.n)])
        return {"advance": advance, "safety": np.zeros_like(advance)}

    def diagnostics(self):
        return self.log.diagnostics()


class WaypointReference:
    """r(t) that traverses a polyline at constant speed, then rests at the last point."""

    def __init__(self, waypoints: Sequence, speed: float):
        self.points = np.array(waypoints, dtype=float)
        if self.points.ndim != 2 or len(self.points) == 0:
            raise ValueError("waypoints must be a non-empty (M, d) list")
        if speed <= 0:
            raise ValueError("the reference speed must be positive")
        self.speed = speed
        self.segments = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(self.lengths)])

    def __call__(self, t: float):
        """Return (r(t), r'(t))."""
        s = self.speed * t
        if s >= self.arc[-1]:
            return self.points[-1].copy(), np.zeros(self.points.shape[1])
        k = np.searchsorted(self.arc, s, side="right") - 1  # never a zero-length segment
        direction = self.segments[k] / self.lengths[k]
        return self.points[k] + (s - self.arc[k]) * direction, self.speed * direction


class ConstantVelocityReference:
    """r(t) = start + t * velocity."""

    def __init__(self, start: Sequence, velocity: Sequence):
        self.start, self.velocity = np.array(start, dtype=float), np.array(velocity, dtype=float)
        if self.start.shape != self.velocity.shape:
            raise ValueError("start and velocity must have the same dimension")

    def __call__(self, t: float):
        """Return (r(t), r'(t))."""
        return self.start + t * self.velocity, self.velocity.copy()


@controller_registry.register("leader_follower")
class LeaderFollower(Formation):
    """Formation law plus pinning of a few agents to a virtual leader's trajectory r(t):

        u_i += k_pin * (r(t) + c_i - p_i)  +  k_pin_v * (r'(t) - v_i)     for i in ``pinned``

    Only the pinned agents know r(t), and it is never broadcast, so it cannot be spoofed; the
    others follow through the formation law.  A compromised agent that is pinned still flies
    true to the reference, but lies about it to everyone it talks to.

    The reference is ``waypoints`` at speed ``v_ref``, or ``velocity`` from ``start``, or a
    fixed point at ``start``, which defaults to the arena centre.

    W-MSR would filter a lone leader out.  Whoever leads the fleet holds the most extreme value in the
    direction of travel, and that is exactly what W-MSR drops, so its followers stop hearing it and it
    flies off alone.  So ``pinned`` defaults to the first 2F + 1 agents under W-MSR: one honest leader
    then survives both F lying leaders and F values trimmed.  Otherwise it defaults to agent 0.  The
    velocity term stays relative whatever the filter, since followers must match a moving reference,
    and the pins hold the fleet's common velocity to it.

    All the pinning force reaches the fleet through a few agents, so the fleet as a whole sees
    roughly (|pinned| / N) * (k_pin * error + k_pin_v * error'): it settles slowly at large N,
    and raising k_pin alone just makes it ring.  Raise k_pin_v with it, or pin more agents.

    With ``wait_tol`` set, each pinned agent keeps its own clock on r and stops it while the
    formation, as far as that agent can tell from what it holds, is further than ``wait_tol`` from
    agreeing with it.  So lagging agents can catch up.  One wait lasts at most ``wait_limit``
    seconds, and that budget refills once the formation is back within ``wait_tol``.
    """

    def __init__(self, offsets: Union[str, Sequence] = "line", spacing: float = 2.0, k_p: float = 1.0,
                 k_v: float = 1.5, filter: str = "mean", f: Optional[int] = None,
                 pinned: Optional[Sequence[int]] = None, k_pin: float = 1.0, k_pin_v: float = 3.0,
                 waypoints: Optional[Sequence] = None, v_ref: float = 1.0,
                 start: Optional[Sequence] = None, velocity: Optional[Sequence] = None,
                 wait_tol: Optional[float] = None, wait_limit: float = float("inf")):
        super().__init__(offsets, spacing, k_p, k_v, filter, f, damping="relative")
        if waypoints is not None and (start is not None or velocity is not None):
            raise ValueError("give either waypoints or start/velocity, not both")
        self.pinned_param = pinned
        self.pinned = [] if pinned is None else list(pinned)
        self.k_pin, self.k_pin_v = k_pin, k_pin_v
        self.waypoints, self.v_ref, self.start, self.velocity = waypoints, v_ref, start, velocity
        self.wait_tol, self.wait_limit = wait_tol, wait_limit
        self.reference = None
        self.clock = {}
        self.waited = {}
        self.waiting = {}
        self.paused = set()  # pinned agents told to hold at their current place on the reference
        self._clock_step = None

    def reset(self, sim) -> None:
        super().reset(sim)
        needed = 2 * self.aggregator.f + 1 if self.filter == "wmsr" else 1
        if self.pinned_param is None:
            self.pinned = list(range(min(needed, sim.n)))
        elif len(self.pinned) < needed:
            warnings.warn(f"W-MSR with f={self.aggregator.f} needs {needed} pinned agents: with {len(self.pinned)} "
                          "it can filter the leaders out, and the fleet stops following")
        if any(not 0 <= i < sim.n for i in self.pinned):
            raise ValueError("pinned ids out of range")
        if sim.compromised[self.pinned].any():
            warnings.warn("a compromised agent is pinned: it follows the reference but lies about it")
        if self.waypoints is not None:
            self.reference = WaypointReference(self.waypoints, self.v_ref)
        else:
            start = np.full(sim.dim, sim.cfg.arena_size / 2.0) if self.start is None else self.start
            velocity = np.zeros(sim.dim) if self.velocity is None else self.velocity
            self.reference = ConstantVelocityReference(start, velocity)
        if self.reference(0.0)[0].shape != (sim.dim,):
            raise ValueError(f"the reference must have dimension {sim.dim}")
        self.clock = {i: 0.0 for i in self.pinned}
        self.waited = {i: 0.0 for i in self.pinned}
        self.waiting = {i: False for i in self.pinned}
        self.paused = set()
        self._clock_step = None

    def local_formation_error(self, obs, i: int) -> float:
        """RMS disagreement between agent i and the neighbours it trusts, in the formation frame."""
        ids, pos, _ = obs.received(i)
        own = obs.pos[i] - self.offsets[i]
        corrected = pos - self.offsets[ids]
        trusted = np.isin(ids, self.aggregator.aggregate(own, corrected, ids).trusted)
        if not trusted.any():
            return 0.0
        return float(np.sqrt(((corrected[trusted] - own) ** 2).sum(axis=1).mean()))

    def _advance_clocks(self, obs) -> None:
        # Once per step, however many times control is asked for: a gate may evaluate it twice.
        if obs.step == self._clock_step:
            return
        elapsed = 0.0 if self._clock_step is None else (obs.step - self._clock_step) * obs.dt
        self._clock_step = obs.step
        for i in self.pinned:
            behind = self.wait_tol is not None and self.local_formation_error(obs, i) > self.wait_tol
            self.waiting[i] = i in self.paused or (behind and self.waited[i] < self.wait_limit)
            self.waited[i] = self.waited[i] + elapsed if behind and i not in self.paused else 0.0
            if not self.waiting[i]:
                self.clock[i] += elapsed

    def pin_term(self, obs, i: int) -> np.ndarray:
        r, r_dot = self.reference(self.clock[i])
        if self.waiting[i]:
            r_dot = np.zeros_like(r_dot)
        return self.k_pin * (r + self.offsets[i] - obs.pos[i]) + self.k_pin_v * (r_dot - obs.vel[i])

    def control_parts(self, obs):
        self._advance_clocks(obs)
        parts = super().control_parts(obs)
        for i in self.pinned:
            parts["advance"][i] += self.pin_term(obs, i)
        return parts
