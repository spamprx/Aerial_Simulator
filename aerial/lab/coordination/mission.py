"""The mission controller: the behaviour both fleets actually fly (naive and gated, D1)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.dynamics import clip_norm
from ..registry import controller_registry
from ..safety import (DEFAULT_CAP, geofence_repulsion, govern_speed, obstacle_repulsion, prioritise,
                      separation_repulsion, steer_around)
from .base import SplitController
from .boids import Boids, arrive_velocity
from .formation import LeaderFollower, resolve_offsets

MISSION_KEYS = {"waypoints", "agent_waypoints", "home", "sectors", "assignment", "r_detect"}
BEHAVIOURS = ("formation", "boids")
FORMATION_K_SEP, FLOCK_K_SEP = 40.0, 15.0  # separation gain per metre of d_safe


def read_mission(mission: Dict, n: int, dim: int) -> Dict:
    """Validate a ``cfg.mission`` section.

    ``waypoints``        (M, d): one route for the whole fleet.
    ``agent_waypoints``  N lists of (M_i, d): a route per agent, for assigned targets or sector
                         sweeps.  An empty list means that agent has nothing to do.
    ``home``             (d,): where the fleet returns to, if the controller has a return leg.
    ``sectors``          (S, 2, 2): the low and high corner of each sector, in x-y.
    ``assignment``       N lists of sector indices: what each agent's route sweeps.  Controllers fly the route, and
                         this says what it is for.
    ``r_detect``         the radius of the sensor that finds survivors and sweeps area, for the mission metrics.
    """
    unknown = set(mission) - MISSION_KEYS
    if unknown:
        raise KeyError(f"unknown mission keys: {sorted(unknown)}")
    result = {}
    if "assignment" in mission:
        if len(mission["assignment"]) != n:
            raise ValueError(f"assignment must list {n} agents, got {len(mission['assignment'])}")
        count = len(mission.get("sectors", ()))
        if any(not 0 <= index < count for chosen in mission["assignment"] for index in chosen):
            raise ValueError(f"assignment names a sector outside the {count} listed")
    if mission.get("r_detect", 1.0) <= 0:
        raise ValueError("r_detect must be positive")
    if "waypoints" in mission:
        result["waypoints"] = _route(mission["waypoints"], dim)
    if "agent_waypoints" in mission:
        routes = mission["agent_waypoints"]
        if len(routes) != n:
            raise ValueError(f"agent_waypoints must list {n} routes, got {len(routes)}")
        result["agent_waypoints"] = [_route(route, dim) for route in routes]
    if "home" in mission:
        result["home"] = np.array(mission["home"], dtype=float)
        if result["home"].shape != (dim,):
            raise ValueError(f"home must have shape ({dim},)")
    return result


def _route(points, dim: int) -> np.ndarray:
    return np.array(points, dtype=float).reshape(-1, dim)


class GuidedController(SplitController):
    """What every mission-level controller shares: slots, the safety part, steering and seeking.

    ``spacing``, ``r_sep`` and ``k_sep`` default to multiples of ``cfg.d_safe``.  Keep ``spacing`` at
    or above ``r_sep``, or the slots of a formation push each other apart for ever.
    """

    def __init__(self, offsets="circle", spacing: Optional[float] = None, filter: str = "mean",
                 f: Optional[int] = None, arrive_tol: float = 1.0, v_ref: float = 1.5, speed_limit: float = 2.0,
                 k_goal: float = 2.0, v_goal: Optional[float] = None, wait_tol: Optional[float] = 2.0,
                 wait_limit: float = 20.0, r_sep: Optional[float] = None, k_sep: Optional[float] = None,
                 d_influence: float = 5.0, k_obs: float = 5.0, fence_margin: float = 2.0, k_fence: float = 2.0):
        self.offsets_spec, self.spacing_param = offsets, spacing
        self.filter, self.f = filter, f
        self.arrive_tol, self.v_ref, self.speed_limit = arrive_tol, v_ref, speed_limit
        self.k_goal, self.v_goal = k_goal, v_ref if v_goal is None else v_goal
        self.wait_tol, self.wait_limit = wait_tol, wait_limit
        self.r_sep_param, self.k_sep_param = r_sep, k_sep
        self.d_influence, self.k_obs = d_influence, k_obs
        self.fence_margin, self.k_fence = fence_margin, k_fence
        self.d_safe, self.spacing, self.r_sep, self.k_sep = None, None, None, None
        self.offsets = None
        self.max_accel, self.cap, self.arena_size = None, DEFAULT_CAP, None
        self.centres, self.radii = None, None

    def _setup(self, sim, k_sep_factor: float) -> None:
        self.d_safe = sim.cfg.d_safe
        self.spacing = 3.0 * self.d_safe if self.spacing_param is None else self.spacing_param
        self.r_sep = 2.5 * self.d_safe if self.r_sep_param is None else self.r_sep_param
        self.k_sep = k_sep_factor * self.d_safe if self.k_sep_param is None else self.k_sep_param
        if self.r_sep <= self.d_safe:
            raise ValueError("r_sep must exceed d_safe, or separation acts too late to keep it")
        self.offsets = resolve_offsets(self.offsets_spec, sim.n, sim.dim, self.spacing)
        self.max_accel = sim.cfg.max_accel
        self.cap = DEFAULT_CAP if self.max_accel is None else self.max_accel
        self.arena_size = sim.cfg.arena_size
        obstacles = sim.world.obstacles
        self.centres = np.array([o.center for o in obstacles]).reshape(len(obstacles), sim.dim)
        self.radii = np.array([o.radius for o in obstacles])

    def _make_leader(self, waypoints: Optional[np.ndarray], **params) -> LeaderFollower:
        route = {} if waypoints is None else {"waypoints": waypoints.tolist(), "v_ref": self.v_ref}
        return LeaderFollower(**{"offsets": self.offsets_spec, "spacing": self.spacing, "filter": self.filter,
                                 "f": self.f, "wait_tol": self.wait_tol, "wait_limit": self.wait_limit,
                                 **route, **params})

    @staticmethod
    def _check_reachable(sim, i: int, route: np.ndarray) -> None:
        for point in route:
            if not sim.world.is_clear(point):
                raise ValueError(f"agent {i}'s route has {point.tolist()} inside an obstacle: it can never arrive")
            if point.min() < 0.0 or point.max() > sim.cfg.arena_size:
                raise ValueError(f"agent {i}'s route has {point.tolist()} outside the arena: it can never arrive")

    def _bodies(self, pos: np.ndarray):
        """What agent i steers around: the obstacles, and each neighbour out to d_safe."""
        return ((self.centres, self.radii, self.d_influence),
                (pos, np.full(len(pos), self.d_safe), self.r_sep - self.d_safe))

    def _seek_term(self, obs, i: int, goal: Optional[np.ndarray]) -> np.ndarray:
        """Fly to ``goal``, or stop where it is if there is none.

        This steers the velocity to fly at, not the acceleration: seeking damps velocity, and that
        damping would otherwise point backwards along a surface and fight the slide past it.
        """
        _, pos, _ = obs.received(i)
        p_i, v_i = obs.pos[i], obs.vel[i]
        desired = np.zeros_like(p_i) if goal is None else arrive_velocity(p_i, goal, self.v_goal)
        for bodies in self._bodies(pos):
            desired = steer_around(desired, p_i, *bodies)
        return self.k_goal * (desired - v_i)

    def _with_safety(self, obs, advance: np.ndarray, separation: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Finish ``advance`` and build the safety part.  ``advance`` is modified in place."""
        safety = np.zeros_like(advance)
        for i in range(obs.n):
            _, pos, _ = obs.received(i)
            p_i, v_i = obs.pos[i], obs.vel[i]
            for bodies in self._bodies(pos):
                advance[i] = steer_around(advance[i], p_i, *bodies)
            advance[i] = govern_speed(advance[i], v_i, self.speed_limit)
            shove = separation_repulsion(p_i, pos, self.r_sep, self.k_sep, self.cap, v_i) \
                if separation is None else separation[i]
            # An obstacle cannot get out of the way, so neighbours must not push an agent into one.
            shove = steer_around(shove, p_i, self.centres, self.radii, self.d_influence)
            obstacle = obstacle_repulsion(p_i, v_i, obs.world, self.d_influence, self.k_obs, self.cap)
            fence = geofence_repulsion(p_i, self.arena_size, self.fence_margin, self.k_fence, v_i, self.cap)
            safety[i] = clip_norm(shove + obstacle + fence, self.cap)
        return {"advance": prioritise(advance, safety, self.max_accel), "safety": safety}


@controller_registry.register("mission")
class Mission(GuidedController):
    """Go somewhere, in two parts that a knowledge gate can tell apart:

        advance:  the wrapped behaviour plus the goal, steered around whatever is close -> gated
        safety:   separation, obstacle and geofence repulsion                           -> never gated

    ``behaviour="formation"`` flies leader-follower along ``waypoints``, and an agent is done once
    it has visited its slot at the end of the route.  ``behaviour="boids"`` flocks while each agent
    seeks its own next waypoint in turn, and an agent is done after the last one.  The route comes
    from ``cfg.mission``, or from the ``mission`` argument if given (see ``read_mission``).  With no
    route the fleet holds formation at the arena centre and everyone is done from the start.

    A shared route is offset per agent by ``offsets``: a flock cannot all stand on one point, since
    separation forbids it.  ``diagnostics`` logs ``mission_done``, (N,) bool, which is what
    ``completion_time`` reads.  Each agent decides from its own position and what it holds.
    """

    def __init__(self, behaviour: str = "formation", behaviour_params: Optional[Dict] = None,
                 mission: Optional[Dict] = None, **guided):
        super().__init__(**guided)
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"behaviour must be one of {BEHAVIOURS}")
        self.behaviour_name, self.behaviour_params = behaviour, dict(behaviour_params or {})
        self.mission_override = mission
        self.behaviour = None
        self.routes = None
        self.next_waypoint = None
        self.done = None
        self._progress_step = None

    def reset(self, sim) -> None:
        mission = read_mission(sim.cfg.mission if self.mission_override is None else self.mission_override,
                               sim.n, sim.dim)
        if "waypoints" in mission and "agent_waypoints" in mission:
            raise ValueError("give either waypoints or agent_waypoints, not both")
        # Formation fights a stiff spring, so a firm shove helps it; in a flock a harder shove just
        # throws an agent into its other neighbours.  Measured over dense launches up to N = 20.
        self._setup(sim, FORMATION_K_SEP if self.behaviour_name == "formation" else FLOCK_K_SEP)
        shared = mission.get("waypoints")
        if self.behaviour_name == "formation":
            if "agent_waypoints" in mission:
                raise ValueError("behaviour 'formation' flies one route: use waypoints, or behaviour 'boids'")
            self.behaviour = self._make_leader(shared, **self.behaviour_params)
            ends = np.empty((0, sim.dim)) if shared is None else shared[-1:]
            self.routes = [ends + self.offsets[i] for i in range(sim.n)]
        else:
            # Flocking only makes sense on a shared route.  Cohesion grows with distance, so it would
            # drag agents with routes of their own back towards each other.
            flocking = {"k_coh": 0.1, "k_ali": 0.5} if shared is not None else {"k_coh": 0.0, "k_ali": 0.0}
            self.behaviour = Boids(**{**flocking, "filter": self.filter, "f": self.f, "r_sep": self.r_sep,
                                      "k_sep": self.k_sep, **self.behaviour_params})
            if shared is not None:
                self.routes = [shared + self.offsets[i] for i in range(sim.n)]
            else:
                self.routes = mission.get("agent_waypoints", [np.empty((0, sim.dim))] * sim.n)
        self.behaviour.reset(sim)
        for i, route in enumerate(self.routes):
            self._check_reachable(sim, i, route)
        self.next_waypoint = np.zeros(sim.n, dtype=int)
        self.done = np.array([len(route) == 0 for route in self.routes])
        self._progress_step = None

    def _update_progress(self, obs) -> None:
        # Once per step, however many times control is asked for: a gate may evaluate it twice.
        if obs.step == self._progress_step:
            return
        self._progress_step = obs.step
        for i, route in enumerate(self.routes):
            while not self.done[i] and np.linalg.norm(route[self.next_waypoint[i]] - obs.pos[i]) <= self.arrive_tol:
                self.next_waypoint[i] += 1
                self.done[i] = self.next_waypoint[i] == len(route)

    def goal_of(self, i: int) -> Optional[np.ndarray]:
        return None if self.done[i] else self.routes[i][self.next_waypoint[i]]

    def skip_waypoint(self, i: int) -> None:
        """Count agent i's next waypoint as visited, for when something is keeping it from getting any closer."""
        if not self.done[i]:
            self.next_waypoint[i] += 1
            self.done[i] = self.next_waypoint[i] == len(self.routes[i])

    def normalised_gains(self):
        return self.behaviour.normalised_gains()

    def filter_offsets(self):
        return self.behaviour.filter_offsets()

    def control_parts(self, obs):
        self._update_progress(obs)
        parts = self.behaviour.control_parts(obs)
        advance = parts["advance"].copy()
        if self.behaviour_name == "formation":
            return self._with_safety(obs, advance)
        for i in range(obs.n):
            advance[i] += self._seek_term(obs, i, self.goal_of(i))
        return self._with_safety(obs, advance, parts["safety"])

    def diagnostics(self):
        return {**self.behaviour.diagnostics(), "mission_done": self.done}
