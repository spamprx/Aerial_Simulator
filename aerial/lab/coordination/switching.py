"""The behaviour-switching controller: one finite-state machine per agent, for a whole sortie."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..registry import controller_registry
from .mission import FORMATION_K_SEP, GuidedController, read_mission

ASSEMBLE, TRANSIT, SWEEP, RETURN = range(4)
MODE_NAMES = ("assemble", "transit", "sweep", "return")


@controller_registry.register("switching")
class Switching(GuidedController):
    """Each agent moves through the same four modes, in order and never back:

        ASSEMBLE  rendezvous into formation; a pinned agent holds at the start of the route
        TRANSIT   leader-follower along ``waypoints``, in formation
        SWEEP     leave the formation and visit this agent's own ``agent_waypoints`` in turn
        RETURN    fly to this agent's slot around ``home``, and stay on it

    The switches are each agent's own decision, from its own position and the messages it holds:

        ASSEMBLE -> TRANSIT  it has agreed with its neighbours to within ``assemble_tol`` for
                             ``dwell`` seconds
        TRANSIT  -> SWEEP    it is within ``sector_tol`` of its slot at the end of the route
        SWEEP    -> RETURN   it has visited its last waypoint (at once, if it has none)

    Without limits a lost radio or a stuck neighbour would hold an agent in a mode for ever, so
    ``assemble_limit`` and ``transit_limit`` (seconds in that mode) move it on regardless.
    ``transit_limit`` defaults to twice the route's flying time plus one full wait.

    The route comes from ``cfg.mission`` or the ``mission`` argument: ``waypoints``,
    ``agent_waypoints`` and optionally ``home``, which defaults to the start of the route.
    ``diagnostics`` logs ``mode`` and ``mission_done``, each (N,); an agent is done when it is home.
    """

    def __init__(self, mission: Optional[Dict] = None, assemble_tol: float = 1.0, dwell: float = 1.0,
                 assemble_limit: float = 60.0, sector_tol: float = 3.0, transit_limit: Optional[float] = None,
                 leader_params: Optional[Dict] = None, **guided):
        super().__init__(**guided)
        self.mission_override = mission
        self.assemble_tol, self.dwell, self.assemble_limit = assemble_tol, dwell, assemble_limit
        self.sector_tol, self.transit_limit_param = sector_tol, transit_limit
        self.leader_params = dict(leader_params or {})
        self.leader = None
        self.slots, self.homes, self.routes = None, None, None
        self.transit_limit = None
        self.mode, self.done, self.next_waypoint = None, None, None
        self.settled_for, self.time_in_mode = None, None
        self._mode_step = None

    def reset(self, sim) -> None:
        mission = read_mission(sim.cfg.mission if self.mission_override is None else self.mission_override,
                               sim.n, sim.dim)
        if len(mission.get("waypoints", ())) == 0 or "agent_waypoints" not in mission:
            raise ValueError("switching needs both waypoints (the transit route) and agent_waypoints (the sweeps)")
        self._setup(sim, FORMATION_K_SEP)
        route = mission["waypoints"]
        self.leader = self._make_leader(route, **self.leader_params)
        self.leader.reset(sim)
        home = mission.get("home", route[0])
        self.slots = route[-1] + self.offsets
        self.homes = home + self.offsets
        self.routes = mission["agent_waypoints"]
        for i in range(sim.n):
            self._check_reachable(sim, i, np.vstack([self.slots[i:i + 1], self.homes[i:i + 1], self.routes[i]]))
        length = float(self.leader.reference.arc[-1])
        self.transit_limit = 2.0 * length / self.v_ref + self.wait_limit \
            if self.transit_limit_param is None else self.transit_limit_param
        self.mode = np.full(sim.n, ASSEMBLE)
        self.done = np.zeros(sim.n, dtype=bool)
        self.next_waypoint = np.zeros(sim.n, dtype=int)
        self.settled_for = np.zeros(sim.n)
        self.time_in_mode = np.zeros(sim.n)
        self._mode_step = None

    # ---- the state machine --------------------------------------------------------
    def _enter(self, i: int, mode: int) -> None:
        self.mode[i] = mode
        self.time_in_mode[i] = 0.0

    def _update_modes(self, obs) -> None:
        # Once per step, however many times control is asked for: a gate may evaluate it twice.
        if obs.step == self._mode_step:
            return
        elapsed = 0.0 if self._mode_step is None else (obs.step - self._mode_step) * obs.dt
        self._mode_step = obs.step
        self.time_in_mode += elapsed
        for i in range(obs.n):
            p_i = obs.pos[i]
            if self.mode[i] == ASSEMBLE:
                heard = obs.adj[i].any() or obs.n == 1
                agreed = heard and self.leader.local_formation_error(obs, i) <= self.assemble_tol
                self.settled_for[i] = self.settled_for[i] + elapsed if agreed else 0.0
                if self.settled_for[i] >= self.dwell or self.time_in_mode[i] >= self.assemble_limit:
                    self._enter(i, TRANSIT)
            elif self.mode[i] == TRANSIT:
                arrived = np.linalg.norm(self.slots[i] - p_i) <= self.sector_tol
                if arrived or self.time_in_mode[i] >= self.transit_limit:
                    self._enter(i, SWEEP)
            if self.mode[i] == SWEEP:
                route = self.routes[i]
                while self.next_waypoint[i] < len(route) and \
                        np.linalg.norm(route[self.next_waypoint[i]] - p_i) <= self.arrive_tol:
                    self.next_waypoint[i] += 1
                if self.next_waypoint[i] == len(route):
                    self._enter(i, RETURN)
            if self.mode[i] == RETURN and np.linalg.norm(self.homes[i] - p_i) <= self.arrive_tol:
                self.done[i] = True

    def goal_of(self, i: int) -> np.ndarray:
        # An agent that is home keeps flying to its slot, so a later arrival cannot nudge it off.
        return self.routes[i][self.next_waypoint[i]] if self.mode[i] == SWEEP else self.homes[i]

    # ---- control ------------------------------------------------------------------
    def normalised_gains(self):
        return self.leader.normalised_gains()

    def filter_offsets(self):
        return self.leader.filter_offsets()

    def control_parts(self, obs):
        self._update_modes(obs)
        self.leader.paused = {i for i in self.leader.pinned if self.mode[i] == ASSEMBLE}
        advance = self.leader.control_parts(obs)["advance"].copy()
        for i in np.flatnonzero(self.mode >= SWEEP):
            advance[i] = self._seek_term(obs, i, self.goal_of(i))
        return self._with_safety(obs, advance)

    def diagnostics(self):
        return {**self.leader.diagnostics(), "mode": self.mode, "mission_done": self.done}
