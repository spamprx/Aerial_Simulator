"""What a held UAV does once holding has not helped: look for a move it knows to be safe, and go round.

A gate that only ever holds is stuck for good behind anything that stays put.  Going round adds nothing to what the gate
trusts, because each move still has to pass K_i(Safe), the same test as any other.  If it is wrong or out of date, the cost
is time.  The one exception is backing away, below, which has its own test.

Three things stand in the way, and each has its own answer here:

- **Tiny steps.**  A one-step plan from rest moves about a centimetre, so nearly anything passes it and then fails at speed,
  and a UAV flickers at the edge of what it knows.  So a heading is judged by whether the UAV could follow it for two
  lookaheads at cruising speed.
- **Pockets.**  Between a liar and a wall, every heading known to be safe can lead nowhere, however close the far side looks.
  So the heading comes from a breadth-first search over the grid, round every cell any neighbour could be in.
- **Already too close.**  A UAV can coast inside what it needs while braking, and then even holding is not known to be safe, so
  no move can pass K_i(Safe).  It backs away with the gate shut, but only on a heading that gets nearer nowhere a neighbour could
  be, so that a neighbour that stays put cannot get any closer than if it had held.  Gaining on the nearest cell alone is not
  enough: that flew a UAV into a liar it had measured, closing on the liar's real cell while backing off from another.  Against
  a neighbour known only by its range this finds nothing, since every heading goes towards some point of the ring around it,
  which is the right answer: with no idea of the direction, no move is better than any other.

A UAV also gives up on a waypoint it has been kept from once it is within separation distance.  Otherwise two UAVs with goals
close together would wait for ever inside each other's clearance, each for the other to leave.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.world import breadth_first_steps
from ..safety import occupied_clearance, plan_clear_of_world, predict_plan, prioritise
from .abstraction import Grid
from .worlds import PossibleWorlds, knows_safe, only_moves_away, possibly_occupied, worst_margin

TURNS = np.deg2rad([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])  # how far from the search's heading to try, on either side
REPLAN_EVERY = 0.5  # seconds between searches: what blocks a UAV changes slowly, and a search costs time
SEARCH_REACH = 20.0  # metres: a neighbour further away than this is not in the way yet
LOOK_AHEAD = 2.0  # lookaheads at cruising speed that a heading must be known safe for
CELLS_AHEAD = 3  # how far along the search's path to aim, so as to follow the path and not the staircase of its cells


@dataclass
class Move:
    """``command`` is an advance command, or None to keep holding.  ``known_safe`` says it passed K_i(Safe), and ``clear_ahead``
    that the way straight at the goal did too.  A command that is not known safe is a retreat."""

    command: Optional[np.ndarray] = None
    known_safe: bool = False
    clear_ahead: bool = False


class Detours:
    """The per-agent state of going round, for a gate wrapping a controller that exposes ``goal_of``, ``k_goal`` and ``v_goal``."""

    def __init__(self, patience: float, cfg, inner, tau: float, grid: Grid, plan_steps: int, n: int):
        self.patience_steps = int(round(patience / cfg.dt))
        self.cfg, self.inner, self.grid, self.plan_steps = cfg, inner, grid, plan_steps
        self.tau = tau
        self.closed_since = np.full(n, -1)  # the step this agent was first held up, or -1 if it is not
        self.clear_for = np.zeros(n, dtype=int)  # consecutive steps with the way to its goal clear
        self.side = np.ones(n)
        self.steps_to_goal: List[Optional[np.ndarray]] = [None] * n
        self.replan_at = np.zeros(n, dtype=int)
        self.planned_goal: List[Optional[np.ndarray]] = [None] * n
        self.open_ground, self.blocked_around = None, None

    def out_of_patience(self, i: int, step: int, closed: bool) -> bool:
        """Whether agent i has been held up for ``patience`` seconds, with no good run of clear headings since.  An agent
        with nowhere left to go has nothing to be held up from."""
        if self.inner.goal_of(i) is None:
            self.closed_since[i], self.clear_for[i], self.side[i] = -1, 0, 1.0
            return False
        if closed and self.closed_since[i] < 0:
            self.closed_since[i] = step
        return bool(self.closed_since[i] >= 0 and step - self.closed_since[i] >= self.patience_steps)

    def choose(self, i: int, obs, worlds: PossibleWorlds, safety_i: np.ndarray) -> Move:
        move = self._best_move(i, obs, worlds, safety_i)
        self.clear_for[i] = self.clear_for[i] + 1 if move.clear_ahead else 0
        if self.clear_for[i] > self.patience_steps:
            # Back to the side everyone starts on.  Two UAVs head-on then turn opposite ways in space, and pass.
            self.closed_since[i], self.side[i], self.clear_for[i] = -1, 1.0, 0
        if not move.clear_ahead:
            self._give_up_on_a_waypoint_in_reach(i, obs)
        return move

    def _give_up_on_a_waypoint_in_reach(self, i: int, obs) -> None:
        goal = self.inner.goal_of(i)
        if goal is not None and np.linalg.norm(goal - obs.pos[i]) <= self.inner.r_sep:
            self.inner.skip_waypoint(i)

    def _best_move(self, i: int, obs, worlds: PossibleWorlds, safety_i: np.ndarray) -> Move:
        goal = self.inner.goal_of(i)
        if goal is None or len(worlds.unknown):
            return Move()
        cfg = self.cfg
        p_i, v_i = obs.pos[i], obs.vel[i]
        distance = float(np.linalg.norm(goal - p_i))
        if distance < 1e-9:
            return Move()
        along = np.linspace(0.0, LOOK_AHEAD * self.inner.v_goal * self.tau, 6)[:, None]

        def flown(direction: np.ndarray):
            command = prioritise(self.inner.k_goal * (self.inner.v_goal * direction - v_i), safety_i, cfg.max_accel)
            plan = predict_plan(p_i[None, :], v_i[None, :], (command + safety_i)[None, :], cfg.dt, self.plan_steps,
                                cfg.max_speed, cfg.max_accel)
            return command, knows_safe(i, worlds, plan[:, 0])

        straight = (goal - p_i) / distance
        if knows_safe(i, worlds, p_i + along * straight):
            command, known = flown(straight)
            if known:
                return Move(command, True, True)

        heading = self._heading_around(i, obs, worlds, goal, straight)
        margin_here = worst_margin(worlds, p_i[None, :])
        retreat, best_gain = None, 0.0
        for turn in TURNS:
            for side in (self.side[i], -self.side[i]) if turn > 0.0 else (self.side[i],):
                turned = heading.copy()
                cos, sin = np.cos(side * turn), np.sin(side * turn)
                turned[:2] = cos * heading[0] - sin * heading[1], sin * heading[0] + cos * heading[1]
                probe = p_i + along * turned
                if knows_safe(i, worlds, probe):
                    command, known = flown(turned)
                    if known:
                        if turn > 0.0:
                            self.side[i] = side
                        return Move(command, True)
                elif margin_here < 0.0 and plan_clear_of_world(probe, worlds.world, worlds.d_safe) and only_moves_away(worlds, probe, turned):
                    gain = worst_margin(worlds, probe[-1:]) - margin_here
                    if gain > best_gain:
                        retreat, best_gain = flown(turned)[0], gain
        return Move(retreat)

    # ---- the search ---------------------------------------------------------------------------
    def _heading_around(self, i: int, obs, worlds: PossibleWorlds, goal: np.ndarray, straight: np.ndarray) -> np.ndarray:
        """(d,) unit heading to start out for ``goal`` on, in the x-y plane, or ``straight`` if the search finds no route."""
        grid = self.grid
        if self.open_ground is None:
            self._prepare_search(obs.world)
        if obs.step >= self.replan_at[i] or not np.array_equal(self.planned_goal[i], goal):
            self.replan_at[i] = obs.step + max(int(round(REPLAN_EVERY / self.cfg.dt)), 1)
            self.planned_goal[i] = goal.copy()
            self.steps_to_goal[i] = self._search(obs.pos[i], worlds, goal)
        steps = self.steps_to_goal[i]
        routes = np.argwhere(steps >= 0)
        if len(routes) == 0:
            return straight
        # Standing inside what is blocked is the usual case here.  It backs out to the nearest cell that has a route, which is
        # what gets it out of a pocket: there is no route from the pocket, however close the far side looks.
        here = grid.cell_of(obs.pos[i])[:2] - grid.low
        cell = routes[np.argmin(np.linalg.norm(routes - here, axis=1))]
        for _ in range(CELLS_AHEAD):
            around = cell + np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            around = around[((around >= 0) & (around < grid.side)).all(axis=1)]
            steps_there = steps[around[:, 0], around[:, 1]]
            nearer = around[(steps_there >= 0) & (steps_there < steps[tuple(cell)])]
            if len(nearer) == 0:
                break
            cell = nearer[0]
        to_cell = (cell + grid.low + 0.5) * grid.h - obs.pos[i][:2]
        length = float(np.linalg.norm(to_cell))
        if length < 1e-9:
            return straight
        heading = np.zeros_like(straight)
        heading[:2] = to_cell / length
        return heading

    def _search(self, position: np.ndarray, worlds: PossibleWorlds, goal: np.ndarray) -> np.ndarray:
        """(side, side) steps to ``goal`` from each cell, -1 where no route gets there."""
        grid = self.grid
        free = self.open_ground.copy()
        occupied = possibly_occupied(worlds, position, SEARCH_REACH)[:, :2] - grid.low
        if len(occupied):
            reached = (occupied[:, None, :] + self.blocked_around[None, :, :]).reshape(-1, 2)
            inside = ((reached >= 0) & (reached < grid.side)).all(axis=1)
            free[reached[inside, 0], reached[inside, 1]] = False
        target = tuple((np.clip(grid.cell_of(goal)[:2], grid.low, grid.high) - grid.low).tolist())
        free[target] = True  # a goal inside what is blocked can still be approached
        return breadth_first_steps(free, target)

    def _prepare_search(self, world) -> None:
        """What never changes in the search: the cells clear of buildings and the fence, and how far an occupied cell blocks."""
        grid = self.grid
        axis = (np.arange(grid.low, grid.high + 1) + 0.5) * grid.h
        centres = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
        half_diagonal = grid.h * np.sqrt(grid.dim) / 2.0
        free = ((centres >= 0.0) & (centres <= grid.arena_size)).all(axis=-1)
        if world.obstacles:
            obstacle_centres, radii = world.obstacle_arrays
            clear = np.linalg.norm(centres[..., None, :] - obstacle_centres[:, :2], axis=-1) - radii
            free &= clear.min(axis=-1) >= self.cfg.d_safe + half_diagonal
        self.open_ground = free
        blocks_within = occupied_clearance(self.cfg.d_safe, grid.h, grid.dim) + half_diagonal + grid.h
        reach = int(np.ceil(blocks_within / grid.h))
        offsets = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * 2, indexing="ij"), axis=-1).reshape(-1, 2)
        self.blocked_around = offsets[np.linalg.norm(offsets, axis=1) * grid.h <= blocks_within]
