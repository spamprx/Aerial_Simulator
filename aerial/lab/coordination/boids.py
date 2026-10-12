"""BOIDS flocking: separation, alignment and cohesion, with optional goal and obstacle terms."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.dynamics import clip_norm
from ..registry import controller_registry
from ..resilience.aggregators import FilterLog, build_aggregator
from ..safety import DEFAULT_CAP, obstacle_repulsion, prioritise, separation_repulsion
from .base import SplitController


def arrive_velocity(p_i: np.ndarray, goal: np.ndarray, speed: float, slow_radius: float = 2.0) -> np.ndarray:
    """(d,) velocity to fly at to reach ``goal``: ``speed``, slowing down inside ``slow_radius``."""
    to_goal = goal - p_i
    distance = float(np.linalg.norm(to_goal))
    return np.zeros_like(p_i) if distance < 1e-12 else speed * min(distance / slow_radius, 1.0) * to_goal / distance


def seek(p_i: np.ndarray, v_i: np.ndarray, goal: np.ndarray, speed: float, gain: float,
         slow_radius: float = 2.0) -> np.ndarray:
    """(d,) acceleration that tracks ``arrive_velocity``."""
    return gain * (arrive_velocity(p_i, goal, speed, slow_radius) - v_i)


@controller_registry.register("boids")
class Boids(SplitController):
    """Reynolds flocking on the received reports:

        advance:  k_coh * (agg({p~_j}) - p_i)  +  k_ali * (agg({v~_j}) - v_i)  +  seek(goal)
        safety:   separation from every report closer than r_sep  +  obstacle repulsion

    Cohesion and alignment go through the aggregator, with normalised weights (D11).  Separation
    is a safety term and deliberately does not: it uses every received report, so a liar who
    claims to be close still gets avoided.

    ``goal`` is one point for the whole flock or an (N, d) list, one per agent.  ``max_accel`` from
    the config caps the safety term, and ``advance`` gets whatever authority that leaves.
    """

    def __init__(self, k_sep: float = 10.0, r_sep: float = 3.0, k_ali: float = 1.5, k_coh: float = 0.5,
                 filter: str = "mean", f: Optional[int] = None,
                 goal: Optional[Sequence] = None, k_goal: float = 1.0, v_goal: float = 2.0,
                 avoid_obstacles: bool = False, d_influence: float = 5.0, k_obs: float = 5.0):
        self.k_sep, self.r_sep, self.k_ali, self.k_coh = k_sep, r_sep, k_ali, k_coh
        self.filter, self.f = filter, f
        self.goal, self.k_goal, self.v_goal = goal, k_goal, v_goal
        self.avoid_obstacles, self.d_influence, self.k_obs = avoid_obstacles, d_influence, k_obs
        self.goals = None
        self.aggregator = None
        self.log = None
        self.max_accel = None

    def reset(self, sim) -> None:
        self.aggregator = build_aggregator(self.filter, sim.cfg.F if self.f is None else self.f)
        self.log = FilterLog(sim.n)
        self.max_accel = sim.cfg.max_accel
        if self.goal is None:
            self.goals = None
        else:
            goal = np.array(self.goal, dtype=float)
            if goal.shape not in ((sim.dim,), (sim.n, sim.dim)):
                raise ValueError(f"goal must have shape ({sim.dim},) or ({sim.n}, {sim.dim})")
            self.goals = np.broadcast_to(goal, (sim.n, sim.dim))

    def normalised_gains(self):
        return self.k_coh, self.k_ali

    def control_parts(self, obs):
        cap = DEFAULT_CAP if self.max_accel is None else self.max_accel
        advance, safety = np.zeros_like(obs.pos), np.zeros_like(obs.pos)
        self.log.clear()
        for i in range(obs.n):
            ids, pos, vel = obs.received(i)
            p_i, v_i = obs.pos[i], obs.vel[i]
            cohesion = self.aggregator.aggregate(p_i, pos, ids)
            alignment = self.aggregator.aggregate(v_i, vel, ids)
            self.log.record("pos", i, cohesion)
            self.log.record("vel", i, alignment)
            advance[i] = self.k_coh * (cohesion.value - p_i) + self.k_ali * (alignment.value - v_i)
            if self.goals is not None:
                advance[i] += seek(p_i, v_i, self.goals[i], self.v_goal, self.k_goal)
            safety[i] = separation_repulsion(p_i, pos, self.r_sep, self.k_sep, cap, v_i)
            if self.avoid_obstacles:
                safety[i] = clip_norm(safety[i] + obstacle_repulsion(p_i, v_i, obs.world, self.d_influence, self.k_obs, cap), cap)
        return {"advance": prioritise(advance, safety, self.max_accel), "safety": safety}

    def diagnostics(self):
        return self.log.diagnostics()
