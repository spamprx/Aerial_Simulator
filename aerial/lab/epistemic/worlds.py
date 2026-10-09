"""The possible-worlds set I_i(w): everything agent i cannot rule out from what it genuinely observed.

A world says which of the other agents are faulty, and which cell each of them is in over the next tau
seconds.  Agent i may use only its own true state, the messages it received, their ranges, and its memory of
them (``AgentHistory``).  What it can conclude rests on three things:

- A report is what an agent says.  An honest one is true when sent, so an honest agent lies near its report.
- A range is physics.  A faulty agent, whatever it says, lies on the shell its range describes (D5).
- At most F agents are faulty, so once i has caught F of them, everyone else is known to be honest.

If the fleet shares what it measures, the second gets much stronger.  One range says how far away a faulty agent is, and
leaves a ring.  Every agent not faulty along with it says the same from somewhere else, and those rings cross about where it is.

W-MSR's accepted and rejected masks are not used as evidence.  W-MSR trims honest values too (it discarded 29%
of honest messages at F = 1 with nobody lying, in T2.7), so a rejection proves nothing, and a covert liar is
accepted, so acceptance proves nothing either.

Agents i does not hear fall into three groups, and they are what the gate's "the actual world is in I_i(w)"
guarantee rests on:

- Heard recently, within ``silence_limit`` steps: tracked from the last message, with its cells growing by
  max_speed * dt per silent step.  This is the option T4.4 offers for a dropped packet.  Holding also triggers a
  re-query, which usually refreshes it.
- Never heard, with lost packets possible and too early to tell: it could be anywhere, so nothing is known to be
  safe.  This is what closes the gate on the first steps of a run with packet loss.
- Silent for longer than ``silence_limit``: a run of drops that long has probability under 1e-9, so it is taken to
  be out of radio range, and config validation has checked that an agent out of range cannot reach i within tau.
  This one assumption is probabilistic rather than certain.  The world count still counts such an agent as anywhere,
  so that hearing it can only ever shrink the count.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, Iterator, Optional

import numpy as np

from ..config import RANGE_NOISE_BOUND, gate_settings
from ..core.world import World
from ..safety import occupied_clearance, plan_clear_of_world
from .abstraction import Grid

RANGES, RANGE_ORIGIN = "ranges", "range_origin"  # the payload's keys for what an agent measured, and where from
NEVER = -1
CERTAINTY = 1e-9  # how unlikely a run of dropped packets must be before silence means out of range
_SPEED_TOLERANCE = 1e-9  # a speed saturated at the limit can read a hair above it after rounding
_TOLERANCE = 1e-9
HONEST, FAULTY = "honest", "faulty"


def silence_limit(drop_prob: float) -> float:
    """Silent steps after which an agent is taken to be out of radio range: drop_prob ** limit <= CERTAINTY."""
    if drop_prob <= 0.0:
        return 0
    if drop_prob >= 1.0:
        return math.inf
    return math.ceil(math.log(CERTAINTY) / math.log(drop_prob))


def range_tolerance(eps: float, sigma: float) -> float:
    """The half-width of a range shell: ``eps``, widened if the sensor's noise can exceed it."""
    return max(eps, RANGE_NOISE_BOUND * sigma)


# ---- proof of lying: each is impossible for an honest agent, so none can accuse one ------------
def disagrees_with_range(reported_distance: np.ndarray, ranges: np.ndarray, eps: float, h: float, dim: int) -> np.ndarray:
    """Rule 1: the report is further from i than the agent physically is."""
    return np.abs(reported_distance - ranges) > eps + h * np.sqrt(dim)


def impossible_speed(reported_vel: np.ndarray, max_speed: float) -> np.ndarray:
    """Rule 2: it claims to fly faster than any UAV can."""
    return np.linalg.norm(reported_vel, axis=1) > max_speed * (1.0 + _SPEED_TOLERANCE)


def impossible_jump(report: np.ndarray, last_report: np.ndarray, steps_apart: np.ndarray, max_speed: float, dt: float,
                    h: float) -> np.ndarray:
    """Rule 3: consecutive reports are further apart than it could have flown."""
    return np.linalg.norm(report - last_report, axis=1) > max_speed * dt * steps_apart + h


def beyond_radio_range(reported_distance: np.ndarray, comm_radius: float) -> np.ndarray:
    """Rule 4: it claims to be further away than a message can carry, and yet i heard it."""
    return reported_distance > comm_radius * (1.0 + _TOLERANCE)


def contradicted_by_others(report: np.ndarray, origins: np.ndarray, ranges: np.ndarray, allowance: np.ndarray, faulty_budget: int) -> np.ndarray:
    """Rule 5: more agents measure it somewhere other than its report than could all be lying.

    ``report`` is (N, d), ``origins`` (R, d) where R measured from, ``ranges`` (R, N) what they measured, infinite where
    they say nothing, and ``allowance`` (R, N) how far each measurement can honestly differ from that report.  If it were
    honest, every one that disagrees would be faulty, and there are at most ``faulty_budget`` of those left.  So this
    one proof rests on F, where the other four are physics.
    """
    distance = np.linalg.norm(report[None, :, :] - origins[:, None, :], axis=-1)
    disagree = np.isfinite(ranges) & (np.abs(distance - np.where(np.isfinite(ranges), ranges, 0.0)) > allowance)
    return disagree.sum(axis=0) > faulty_budget


class AgentHistory:
    """history_i: what agent i remembers.  For each other agent, its last two messages and what was
    measured with them, and whether it has ever been caught lying.  Being caught lasts for the whole run."""

    def __init__(self, i: int, n: int, dim: int):
        self.i = i
        self.known_faulty = np.zeros(n, dtype=bool)
        self.last_step, self.prev_step = np.full(n, NEVER), np.full(n, NEVER)
        self.report, self.prev_report = np.zeros((n, dim)), np.zeros((n, dim))
        self.range, self.prev_range = np.full(n, np.inf), np.full(n, np.inf)
        self.origin, self.prev_origin = np.zeros((n, dim)), np.zeros((n, dim))
        self.recorded_step = None

    def record(self, obs, cfg, eps: float, h: float) -> None:
        """Take in this step's fresh messages, once per step however often it is asked."""
        if obs.step == self.recorded_step:
            return
        self.recorded_step = obs.step
        fresh = np.flatnonzero(obs.message_fresh[self.i])
        if len(fresh) == 0:
            return
        report, velocity = obs.msg_pos_for(self.i)[fresh], obs.msg_vel_for(self.i)[fresh]
        ranges = obs.ranges[self.i, fresh]
        position = obs.pos[self.i]
        distance = np.linalg.norm(report - position, axis=1)
        caught = disagrees_with_range(distance, ranges, eps, h, obs.pos.shape[1]) | impossible_speed(velocity, cfg.max_speed)
        if cfg.topology == "disk":  # on a fixed graph, a message says nothing about distance
            caught |= beyond_radio_range(distance, cfg.comm_radius)
        heard_before = self.last_step[fresh] != NEVER
        caught[heard_before] |= impossible_jump(report[heard_before], self.report[fresh[heard_before]],
                                                obs.step - self.last_step[fresh[heard_before]], cfg.max_speed, cfg.dt, h)
        self.known_faulty[fresh[caught]] = True
        self.prev_step[fresh], self.prev_report[fresh] = self.last_step[fresh], self.report[fresh]
        self.prev_range[fresh], self.prev_origin[fresh] = self.range[fresh], self.origin[fresh]
        self.last_step[fresh], self.report[fresh], self.range[fresh], self.origin[fresh] = obs.step, report, ranges, position
        self._hear_what_others_measured(obs, cfg, eps, h, fresh)

    def _hear_what_others_measured(self, obs, cfg, eps: float, h: float, fresh: np.ndarray) -> None:
        """Rule 5, from the ranges in this step's messages, each measured on the step before."""
        shared, origins = obs.payload_for(self.i, RANGES), obs.payload_for(self.i, RANGE_ORIGIN)
        budget = cfg.F - int(self.known_faulty.sum())
        reporters = fresh[~self.known_faulty[fresh]]
        heard = self.last_step != NEVER
        if shared is None or origins is None or budget < 0 or len(reporters) == 0 or not heard.any():
            return
        measured = np.where(heard, shared[reporters], np.inf)
        measured[np.arange(len(reporters)), reporters] = np.inf  # nobody speaks for itself: its own report is the question
        allowance = eps + h * np.sqrt(obs.pos.shape[1]) + cfg.max_speed * cfg.dt * np.abs(self.last_step - (obs.step - 1))
        self.known_faulty[contradicted_by_others(self.report, origins[reporters], measured, allowance[None, :], budget)] = True


@dataclass
class PossibleWorlds:
    """I_i(w), held as what each neighbour could be instead of as a list of worlds.

    The neighbours are the agents i is tracking.  Arrays are indexed by position in ``neighbours``.  Each neighbour
    has the cells of two regions: where it is if honest, and where it is if faulty.  A neighbour heard this step
    must also be consistent with its message before, which is why each region has a second constraint.  That one is
    infinitely wide where it does not apply.  ``faulty_budget`` is how many more faulty agents there can be, and it
    is None if i has caught more than F, which proves the assumption wrong.

    ``shared_*`` is what the fleet measured, if it shares that: R agents ``shared_by``, measuring from ``shared_origin``
    (R, d) the distance ``shared_range`` (R, m) to each neighbour, infinite where one says nothing, with shells
    ``shared_width`` (R,) wide.  A neighbour's cells if it is faulty depend on who else is, so these are the union over
    every consistent H holding it: ``shared_slack`` (m,) says how many of the R can be faulty along with it.
    """

    i: int
    position: np.ndarray
    neighbours: np.ndarray
    age: np.ndarray
    can_be_honest: np.ndarray
    can_be_faulty: np.ndarray
    report: np.ndarray
    honest_radius: np.ndarray
    prev_report: np.ndarray
    prev_honest_radius: np.ndarray
    origin: np.ndarray
    range: np.ndarray
    shell_width: np.ndarray
    prev_origin: np.ndarray
    prev_range: np.ndarray
    prev_shell_width: np.ndarray
    known_faulty: np.ndarray
    known_honest: np.ndarray
    unknown: np.ndarray
    out_of_range: np.ndarray
    faulty_budget: Optional[int]
    grid: Grid
    world: World
    d_safe: float
    shared_by: Optional[np.ndarray] = None
    shared_origin: Optional[np.ndarray] = None
    shared_range: Optional[np.ndarray] = None
    shared_width: Optional[np.ndarray] = None
    shared_slack: Optional[np.ndarray] = None

    # ---- which cells ------------------------------------------------------------------------------
    def honest_mask(self, cells: np.ndarray, which) -> np.ndarray:
        """(W, M) whether each of these neighbours could be in each cell, if it is honest."""
        grid = self.grid
        nearest, _ = grid.distance_range(cells, self.report[which])
        inside = nearest <= self.honest_radius[which, None] + _TOLERANCE
        limited = np.isfinite(self.prev_honest_radius[which])
        if limited.any():
            nearest, _ = grid.distance_range(cells, self.prev_report[which][limited])
            inside[limited] &= nearest <= self.prev_honest_radius[which][limited, None] + _TOLERANCE
        return inside

    def faulty_mask(self, cells: np.ndarray, which) -> np.ndarray:
        """(W, M) whether each of these neighbours could be in each cell, if it is faulty."""
        grid = self.grid
        nearest, farthest = grid.distance_range(cells, self.origin[which])
        low, high = self.range[which] - self.shell_width[which], self.range[which] + self.shell_width[which]
        inside = (nearest <= high[:, None] + _TOLERANCE) & (farthest >= low[:, None] - _TOLERANCE)
        limited = np.isfinite(self.prev_shell_width[which])
        if limited.any():
            nearest, farthest = grid.distance_range(cells, self.prev_origin[which][limited])
            low = self.prev_range[which][limited] - self.prev_shell_width[which][limited]
            high = self.prev_range[which][limited] + self.prev_shell_width[which][limited]
            inside[limited] &= (nearest <= high[:, None] + _TOLERANCE) & (farthest >= low[:, None] - _TOLERANCE)
        return self._agreed_by_others(cells, which, inside)

    def _agreed_by_others(self, cells: np.ndarray, which, inside: np.ndarray) -> np.ndarray:
        """Narrow ``inside`` to the cells that what the other agents measured allows.

        If a neighbour is faulty, then the rest of the faulty agents number at most its slack, and all the others told the
        truth about how far away it was.  So a cell it could be in lies on the shells of all but ``shared_slack`` of them.
        Their own reports count for nothing here: what a liar says can only remove cells from worlds in which it is honest,
        and the actual world, where it is in H, never reads what it said.
        """
        if self.shared_range is None or len(self.shared_range) == 0:
            return inside
        for row, neighbour in enumerate(np.atleast_1d(which)):
            candidates = np.flatnonzero(inside[row])
            speaking = np.flatnonzero(np.isfinite(self.shared_range[:, neighbour]))
            if len(candidates) == 0 or len(speaking) == 0:
                continue
            nearest, farthest = self.grid.distance_range(cells[candidates], self.shared_origin[speaking])
            measured, width = self.shared_range[speaking, neighbour], self.shared_width[speaking]
            agree = (nearest <= (measured + width)[:, None] + _TOLERANCE) & (farthest >= (measured - width)[:, None] - _TOLERANCE)
            inside[row, candidates] = agree.sum(axis=0) >= len(speaking) - self.shared_slack[neighbour]
        return inside

    def cells(self, j: int, status: str) -> np.ndarray:
        """cells_j(status): the (M, d) cells agent ``j`` could be in over the lookahead."""
        where = np.flatnonzero(self.neighbours == j)
        if len(where) == 0:
            raise KeyError(f"agent {j} is not a neighbour of agent {self.i}")
        which = where[:1]
        if status == HONEST:
            region = self.grid.cells_in_ball(self.report[which[0]], self.honest_radius[which[0]])
            return region[self.honest_mask(region, which)[0]]
        if status == FAULTY:
            region = self.grid.cells_in_shell(self.origin[which[0]], self.range[which[0]], self.shell_width[which[0]])
            return region[self.faulty_mask(region, which)[0]]
        raise ValueError(f"status must be '{HONEST}' or '{FAULTY}'")

    # ---- which hypotheses ---------------------------------------------------------------------
    def hypotheses(self) -> Iterator[FrozenSet[int]]:
        """Each set H of neighbours that could be the faulty ones: it holds those known to be, and at most F in all."""
        certain = [int(j) for j, honest in zip(self.neighbours, self.can_be_honest) if not honest]
        open_ids = [int(j) for j, honest, faulty in zip(self.neighbours, self.can_be_honest, self.can_be_faulty)
                    if honest and faulty]
        budget = len(open_ids) if self.faulty_budget is None else self.faulty_budget
        for size in range(min(budget, len(open_ids)) + 1):
            for chosen in itertools.combinations(open_ids, size):
                yield frozenset(certain) | frozenset(chosen)

    @property
    def n_hypotheses(self) -> int:
        open_count = int(np.count_nonzero(self.can_be_honest & self.can_be_faulty))
        if self.faulty_budget is None:
            return 2 ** open_count
        return sum(math.comb(open_count, size) for size in range(min(self.faulty_budget, open_count) + 1))

    # ---- how many worlds ----------------------------------------------------------------------
    def _cell_counts(self):
        """(m,) cells for each neighbour if honest and if faulty.  A status whose cells all lie beyond the grid counts
        as one: it is somewhere out there, where nothing can touch a plan inside the arena."""
        honest_count, faulty_count = np.ones(len(self.neighbours)), np.ones(len(self.neighbours))
        honest = np.flatnonzero(self.can_be_honest)
        honest_count[honest] = self.grid.count_in_balls(self.report[honest], self.honest_radius[honest])
        faulty = np.flatnonzero(self.can_be_faulty)
        for origin in np.unique(self.origin[faulty], axis=0):
            same = faulty[(self.origin[faulty] == origin).all(axis=1)]
            faulty_count[same] = self.grid.count_in_shells(origin, self.range[same], self.shell_width[same])
        # A second constraint matters only if the first region is not already inside it.
        for which in honest[np.linalg.norm(self.report[honest] - self.prev_report[honest], axis=1)
                            > self.prev_honest_radius[honest] - self.honest_radius[honest]]:
            honest_count[which] = len(self.cells(self.neighbours[which], HONEST))
        moved = np.abs(self.range[faulty] - self.prev_range[faulty]) + np.linalg.norm(self.origin[faulty] - self.prev_origin[faulty], axis=1)
        narrowed = moved > self.prev_shell_width[faulty] - self.shell_width[faulty]
        if self.shared_range is not None:
            narrowed |= np.isfinite(self.shared_range[:, faulty]).any(axis=0)
        for which in faulty[narrowed]:
            faulty_count[which] = len(self.cells(self.neighbours[which], FAULTY))
        return np.maximum(honest_count, 1.0), np.maximum(faulty_count, 1.0)

    @cached_property
    def log10_n_worlds(self) -> float:
        """log10 of the sum over H of the product over j of |cells_j(H)|, over all N - 1 other agents.

        An agent i is not tracking could be anywhere, so it counts the whole grid.  This is what makes the count fall
        whenever i hears more.  The sum over H is written as a product times a short series, in the ratios of
        faulty to honest cells: 1, plus the sum of the ratios, plus the sum over pairs, and F <= 2 stops it there.
        """
        honest_count, faulty_count = self._cell_counts()
        untracked = np.concatenate([self.unknown, self.out_of_range]).astype(int)
        whole = float(self.grid.n_cells)
        certain = (~self.can_be_honest).astype(bool)
        total = float(np.log10(faulty_count[certain]).sum() + np.log10(honest_count[~certain]).sum())
        total += len(untracked) * math.log10(whole)
        open_tracked = self.can_be_honest & self.can_be_faulty
        ratios = faulty_count[open_tracked] / honest_count[open_tracked]
        open_untracked = np.ones(np.count_nonzero(~self.known_honest[untracked] & ~self.known_faulty[untracked]))
        ratios = np.concatenate([ratios, open_untracked])
        if self.faulty_budget is None:
            return total + float(np.log10(1.0 + ratios).sum())
        series = 1.0
        if self.faulty_budget >= 1:
            series += float(ratios.sum())
        if self.faulty_budget >= 2:
            series += (float(ratios.sum()) ** 2 - float((ratios ** 2).sum())) / 2.0
        return total + math.log10(series)

    # ---- which neighbours can matter to a plan --------------------------------------------------
    def within_reach_of(self, plan_i: np.ndarray, clearance: float):
        """(honest, faulty), masks (m,) of neighbours whose region could touch a cell within ``clearance`` of the plan.
        This only rules out what is too far away, and it is what makes ``knows_safe`` cheap in the usual case."""
        start = plan_i[0]
        half_diagonal = self.grid.h * np.sqrt(self.grid.dim) / 2.0
        spread = float(np.linalg.norm(plan_i - start, axis=1).max()) + clearance + half_diagonal + _TOLERANCE
        honest = np.linalg.norm(self.report - start, axis=1) - spread <= self.honest_radius
        centre = np.linalg.norm(self.origin - start, axis=1)
        faulty = (self.range - self.shell_width <= centre + spread) & (self.range + self.shell_width >= centre - spread)
        return honest & self.can_be_honest, faulty & self.can_be_faulty


def build_possible_worlds(i: int, obs, history_i: AgentHistory, F: int, grid: Grid, cfg, tau: Optional[float] = None,
                          eps: Optional[float] = None, true_faulty: Optional[np.ndarray] = None) -> PossibleWorlds:
    """I_i(w) from what agent i holds.  ``tau`` and ``eps`` default to the config's (D10).

    With ``true_faulty``, the oracle's case (D1), agent i is simply told who is faulty: there is one hypothesis.
    Where each of them is still has to come from messages and ranges, like anyone else.
    """
    if obs.ranges is None:
        raise ValueError('possible worlds need ranging (D5): set "sensing": {"ranging": true}')
    settings = gate_settings(cfg, tau=tau, eps=eps)
    tau = settings["tau"]
    eps = range_tolerance(settings["eps"], cfg.range_sigma)
    history_i.record(obs, cfg, eps, grid.h)
    n = obs.n
    step = obs.step
    others = np.arange(n) != i
    ever = history_i.last_step != NEVER
    limit = silence_limit(cfg.drop_prob)
    tracked = ever & (step - history_i.last_step <= limit) & others
    unknown = ~ever & others & (step + 1 <= limit)
    out_of_range = others & ~tracked & ~unknown

    known_faulty = (history_i.known_faulty if true_faulty is None else true_faulty) & others
    caught = int(known_faulty.sum())
    if true_faulty is not None:
        known_honest, budget = others & ~known_faulty, 0
    elif caught == F:
        known_honest, budget = others & ~known_faulty, 0
    elif caught < F:
        known_honest, budget = np.zeros(n, dtype=bool), F - caught
    else:  # more liars than F: the assumption is false, and counting on it would prove nothing
        known_honest, budget = np.zeros(n, dtype=bool), None

    neighbours = np.flatnonzero(tracked)
    age = step - history_i.last_step[neighbours]
    speed_dt = cfg.max_speed * cfg.dt
    ahead = cfg.max_speed * tau
    prev_ok = (age == 0) & (history_i.prev_step[neighbours] != NEVER)
    prev_gap = np.where(prev_ok, step - history_i.prev_step[neighbours], 0)
    shared = _shared_measurements(i, obs, neighbours, known_faulty, budget, speed_dt, eps, ahead)
    return PossibleWorlds(
        i=i, position=obs.pos[i].copy(), neighbours=neighbours, age=age,
        can_be_honest=~known_faulty[neighbours], can_be_faulty=~known_honest[neighbours],
        report=history_i.report[neighbours], honest_radius=grid.h + age * speed_dt + ahead,
        prev_report=history_i.prev_report[neighbours],
        prev_honest_radius=np.where(prev_ok, grid.h + prev_gap * speed_dt + ahead, np.inf),
        origin=history_i.origin[neighbours], range=history_i.range[neighbours], shell_width=eps + age * speed_dt + ahead,
        prev_origin=history_i.prev_origin[neighbours], prev_range=np.where(prev_ok, history_i.prev_range[neighbours], 0.0),
        prev_shell_width=np.where(prev_ok, eps + prev_gap * speed_dt + ahead, np.inf),
        known_faulty=known_faulty, known_honest=known_honest, unknown=np.flatnonzero(unknown),
        out_of_range=np.flatnonzero(out_of_range), faulty_budget=budget, grid=grid, world=obs.world, d_safe=cfg.d_safe, **shared,
    )


def _shared_measurements(i: int, obs, neighbours: np.ndarray, known_faulty: np.ndarray, budget: Optional[int], speed_dt: float,
                         eps: float, ahead: float):
    """What the messages i holds say about how far each neighbour is from their senders, for ``PossibleWorlds``.

    Each was measured on the step before its message went out, so it has aged by one step more than the message.  With
    more liars caught than F, the assumption it all rests on is false, and none of it can be used.
    """
    ranges, origins = obs.payload_for(i, RANGES), obs.payload_for(i, RANGE_ORIGIN)
    if ranges is None or origins is None or budget is None:
        return {}
    reporters = np.flatnonzero(obs.adj[i] & ~known_faulty)
    reporters = reporters[reporters != i]
    measured = ranges[reporters][:, neighbours]
    measured[reporters[:, None] == neighbours[None, :]] = np.inf  # nobody vouches for where it is itself
    # A faulty neighbour already caught leaves the whole budget for the reporters; one still open uses up one of it.
    slack = np.where(known_faulty[neighbours], budget, budget - 1)
    width = eps + (obs.message_age[i, reporters] + 1.0) * speed_dt + ahead
    return dict(shared_by=reporters, shared_origin=origins[reporters], shared_range=measured, shared_width=width, shared_slack=slack)


def actual_world_violation(worlds: PossibleWorlds, compromised: np.ndarray, trajectory: np.ndarray) -> Optional[str]:
    """Why the actual world is not in I_i(w), or None if it is.  This is the invariant the gate's soundness rests on.

    ``trajectory`` is (s, N, d): where everyone truly is over the lookahead, starting now.  The true faulty neighbours
    must be one of the hypotheses, and each neighbour must stay in its cells for its actual status.  A position beyond
    the grid is not checked, since no cell represents it.
    """
    actual = frozenset(int(j) for j in worlds.neighbours if compromised[j])
    if actual not in set(worlds.hypotheses()):
        return f"agent {worlds.i}: the true faulty neighbours {sorted(actual)} are not one of its hypotheses"
    for which, j in enumerate(worlds.neighbours):
        cells = worlds.grid.cell_of(trajectory[:, j])
        cells = cells[worlds.grid.contains(cells)]
        if len(cells) == 0:
            continue
        mask = worlds.faulty_mask if compromised[j] else worlds.honest_mask
        if not mask(cells, np.array([which])).all():
            return f"agent {worlds.i}: agent {j} leaves its cells within the lookahead"
    return None


def possibly_occupied(worlds: PossibleWorlds, start: np.ndarray, reach: float) -> np.ndarray:
    """(M, d) the cells that some neighbour could be in, in some world, among the neighbours whose regions come within
    ``reach`` of ``start``.  The rest are too far away to matter yet, and listing their cells would only cost time."""
    centre = np.linalg.norm(worlds.origin - start, axis=1)
    near_shell = (worlds.range - worlds.shell_width <= centre + reach) & (worlds.range + worlds.shell_width >= centre - reach)
    near_report = np.linalg.norm(worlds.report - start, axis=1) - worlds.honest_radius <= reach
    regions = [worlds.cells(worlds.neighbours[which], HONEST) for which in np.flatnonzero(near_report & worlds.can_be_honest)]
    regions += [worlds.cells(worlds.neighbours[which], FAULTY) for which in np.flatnonzero(near_shell & worlds.can_be_faulty)]
    return np.vstack(regions) if regions else np.empty((0, worlds.grid.dim), dtype=int)


def occupied_near(worlds: PossibleWorlds, plan_i: np.ndarray) -> np.ndarray:
    """(M, d) the centres of the cells that some neighbour could be in, in some world, close enough to this plan to matter."""
    nothing = np.empty((0, worlds.grid.dim))
    clearance = occupied_clearance(worlds.d_safe, worlds.grid.h, worlds.grid.dim)
    honest, faulty = worlds.within_reach_of(plan_i, clearance)
    if not (honest.any() or faulty.any()):
        return nothing
    danger = worlds.grid.cells_near(plan_i, clearance)
    if len(danger) == 0:
        return nothing
    occupied = np.zeros(len(danger), dtype=bool)
    if honest.any():
        occupied |= worlds.honest_mask(danger, np.flatnonzero(honest)).any(axis=0)
    if faulty.any():
        occupied |= worlds.faulty_mask(danger, np.flatnonzero(faulty)).any(axis=0)
    return worlds.grid.center(danger[occupied])


def worst_margin(worlds: PossibleWorlds, plan_i: np.ndarray) -> float:
    """How far inside the distance safety needs from occupied cells this plan gets, in the worst world: 0.0 if it never
    does, and otherwise negative.  It says how bad a plan is that ``knows_safe`` has already failed."""
    centres = occupied_near(worlds, plan_i)
    if len(centres) == 0:
        return 0.0
    distance = np.linalg.norm(centres[:, None, :] - plan_i[None, :, :], axis=-1)
    return float(distance.min() - occupied_clearance(worlds.d_safe, worlds.grid.h, worlds.grid.dim))


def only_moves_away(worlds: PossibleWorlds, plan_i: np.ndarray, direction: np.ndarray) -> bool:
    """Whether a straight plan along this unit ``direction`` never gets nearer any point of any cell close enough to matter that
    a neighbour could be in.  A point can be half a diagonal from its cell's centre, so the plan has to gain on the centre at
    least that fast.  What this rules out is a plan that gains on the nearest cell while closing on another, where the neighbour
    may really be."""
    centres = occupied_near(worlds, plan_i)
    return bool(((plan_i[0] - centres) @ direction >= worlds.grid.h * np.sqrt(worlds.grid.dim) / 2.0).all())


def knows_safe(i: int, worlds: PossibleWorlds, plan_i: np.ndarray) -> bool:
    """K_i(Safe): whether the plan is safe in every world agent i considers possible.

    This does not enumerate worlds, of which there can be 10^44.  It tests each neighbour's cells on their own:

        K_i(Safe)  <=>  for every neighbour j, for every status j can have in some consistent H,
                        every cell in cells_j(status) keeps the plan safe.

    Proof.  Safe is a conjunction with one clause per neighbour's cell, and a world assigns each neighbour a status
    and a cell.  (<=)  Take any world.  Each neighbour's status there is consistent with some H, namely that world's,
    and its cell is in the set for that status, so every clause holds, and the world is safe.  (=>)  Take any j,
    any status s consistent with some H, and any cell c in cells_j(s).  Since H is consistent, and every other
    neighbour has a cell under H (a status whose cells lie beyond the grid has the abstract cell "out there", which
    cannot touch a plan), there is a world with j at c.  It must be safe, so c keeps the plan safe.  Nothing here
    uses how the statuses of different neighbours are coupled through |H| <= F, which is why that coupling affects
    only the number of worlds.  The check is exact, and ``tests/test_knows_safe.py`` compares it with enumerating
    the worlds through the Kripke engine.

    The world itself comes first: no knowledge of neighbours helps a plan that leaves the fence.  An agent that could
    be anywhere makes nothing safe, because it could be in any cell near the plan.
    """
    if i != worlds.i:
        raise ValueError(f"these are agent {worlds.i}'s worlds, not agent {i}'s")
    if len(worlds.unknown):
        return False
    if not plan_clear_of_world(plan_i, worlds.world, worlds.d_safe):
        return False
    clearance = occupied_clearance(worlds.d_safe, worlds.grid.h, worlds.grid.dim)
    honest, faulty = worlds.within_reach_of(plan_i, clearance)
    if not (honest.any() or faulty.any()):
        return True
    danger = worlds.grid.cells_near(plan_i, clearance)
    if len(danger) == 0:
        return True
    if honest.any() and worlds.honest_mask(danger, np.flatnonzero(honest)).any():
        return False
    return not (faulty.any() and worlds.faulty_mask(danger, np.flatnonzero(faulty)).any())
