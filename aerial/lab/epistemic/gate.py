"""The knowledge gate: a UAV advances only when it knows the move is safe, and otherwise holds and re-queries."""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from ..config import check_gate_config, gate_settings
from ..coordination.base import SplitController
from ..registry import controller_registry
from ..safety import brake_command, gate_clearance, prioritise, predict_plan
from .abstraction import Grid
from .detour import Detours
from .nested import nested_safe_row
from .worlds import RANGE_ORIGIN, RANGES, AgentHistory, build_possible_worlds, knows_safe, range_tolerance


@controller_registry.register("gated")
class KnowledgeGate(SplitController):
    """Wrap a controller that splits its output, and withhold its ``advance`` unless that is known to be safe.

    Each step, for each agent i: take the inner controller's parts, build I_i(w), and test the plan "apply advance +
    safety for one step, then brake" against it.  If K_i(Safe) holds, the agent applies advance + safety.  If
    not, it holds (brake + safety, D3) and asks for a re-query.  The safety part is never withheld.

    ``h``, ``tau`` and ``eps`` are the caution knobs of D10, and they default to ``cfg.gate``, then D10's values.
    ``requery=False`` holds without asking.  ``size_every`` is how many steps apart to count worlds, the one costly
    diagnostic, and it changes slowly: the log holds the last count in between.  0 never counts, and logs NaN.

    Both of these are off by default, and what they add still has to pass K_i(Safe), apart from backing away from a neighbour
    when even holding has not passed it, which may only move away from everywhere a neighbour could be (``detour.py``):

    ``share_ranges`` has each agent also broadcast the ranges it just measured.  That locates a liar: a hovering liar on a
    UAV's route otherwise blocks it for ever, since one range says how far away it is and not in which direction.  It also
    gives one more proof of lying (``worlds.contradicted_by_others``), and an agent stops steering by reports it has caught.
    ``patience`` is how many seconds an agent holds before it looks for a way round (``detour.py``), and None never does.

    ``diagnostics`` logs ``gate_open`` and ``k_safe``, (N,), which are the same thing today: the move it applied is known
    to be safe.  ``detour`` says that move was a way round instead of the wrapped controller's, and ``retreat`` that the
    agent is backing away, with the gate still closed.  ``known_faulty`` and ``known_honest``, (N, N) with row i saying what
    agent i knows; ``k_k_safe``, (N, N), from ``nested.py``; ``log10_worlds``, ``n_hypotheses`` and ``requery``, (N,).
    ``worlds`` holds the latest I_i(w) of each agent, for tests and the dashboard.  Each agent broadcasts its ``k_safe`` and
    its row of ``known_faulty``.
    """

    def __init__(self, inner: str = "mission", inner_params: Optional[Dict] = None, h: Optional[float] = None,
                 tau: Optional[float] = None, eps: Optional[float] = None, requery: Optional[bool] = None,
                 size_every: Optional[int] = None, share_ranges: Optional[bool] = None, patience: Optional[float] = None):
        self.inner_name, self.inner_params = inner, dict(inner_params or {})
        self.given = dict(h=h, tau=tau, eps=eps, requery=requery, size_every=size_every, share_ranges=share_ranges, patience=patience)
        self.inner = None
        self.grid, self.cfg, self.histories, self.worlds = None, None, None, None
        self.tau, self.eps, self.plan_steps, self.requery_enabled, self.size_every = None, None, None, None, None
        self.true_faulty = None
        self.share_ranges, self.detours = False, None
        self._detour, self._retreat, self._ranges, self._range_origin = None, None, None, None
        self._open, self._known_faulty, self._known_honest = None, None, None
        self._log10_worlds, self._n_hypotheses, self._requery, self._k_k_safe = None, None, None, None

    def reset(self, sim) -> None:
        cfg = sim.cfg
        settings = gate_settings(cfg, **self.given)
        check_gate_config(cfg, settings)
        self.cfg = cfg
        self.inner = controller_registry.create(self.inner_name, **self.inner_params)
        if not isinstance(self.inner, SplitController):
            raise TypeError(f"the gate needs a controller with an advance part to withhold, and '{self.inner_name}' has none")
        self.inner.reset(sim)
        if settings["patience"] is not None and not hasattr(self.inner, "skip_waypoint"):
            raise TypeError(f"going round needs a route to go round to, and '{self.inner_name}' has none: set patience to None")
        h, self.tau = float(settings["h"]), float(settings["tau"])
        self.eps, self.requery_enabled, self.size_every = float(settings["eps"]), bool(settings["requery"]), int(settings["size_every"])
        self.plan_steps = max(int(round(self.tau / cfg.dt)), 1)
        self.share_ranges = bool(settings["share_ranges"])
        # Wide enough that a cell beyond the grid is too far from any point inside the arena to matter.
        self.grid = Grid(h, cfg.arena_size, cfg.dim, cfg.d_safe + h * np.sqrt(cfg.dim))
        self.histories = [AgentHistory(i, sim.n, sim.dim) for i in range(sim.n)]
        self.worlds = None
        self.true_faulty = self._who_is_told_faulty(sim)
        n = sim.n
        self._open = np.zeros(n, dtype=bool)
        self._known_faulty, self._known_honest = np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool)
        self._log10_worlds, self._n_hypotheses = np.full(n, np.nan), np.zeros(n, dtype=int)
        self._requery = np.zeros(n, dtype=bool)
        self._k_k_safe = np.zeros((n, n), dtype=bool)
        self.detours = None if settings["patience"] is None else Detours(settings["patience"], cfg, self.inner, self.tau, self.grid, self.plan_steps, n)
        self._detour, self._retreat = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        self._ranges, self._range_origin = np.full((n, n), np.inf), np.zeros((n, sim.dim))
        self._warn_if_it_would_deadlock(cfg, h)

    def _who_is_told_faulty(self, sim) -> Optional[np.ndarray]:
        return None

    def _warn_if_it_would_deadlock(self, cfg, h: float) -> None:
        """Neighbours closer than the gate's clearance hold each other, and only the never-gated repulsion can part
        them.  A repulsion that stops short of the clearance leaves them holding for ever, with nobody lying at all.
        With ``patience`` they can also back away or go round, so this is no longer a deadlock."""
        if cfg.n_agents < 2 or self.detours is not None:
            return
        clearance = gate_clearance(cfg.d_safe, h, cfg.dim, cfg.max_speed, self.tau, range_tolerance(self.eps, cfg.range_sigma))
        r_sep = getattr(self.inner, "r_sep", None)
        if r_sep is not None and r_sep <= clearance:
            warnings.warn(f"r_sep {r_sep:g} does not reach the gate's clearance of {clearance:.3g}, so neighbours caught between "
                          "them will hold for ever: raise r_sep, or lower h, tau or max_speed")
        spacing = getattr(self.inner, "spacing", None)
        if spacing is not None and getattr(self.inner, "behaviour_name", "formation") == "formation" and spacing <= clearance:
            warnings.warn(f"formation spacing {spacing:g} is inside the gate's clearance of {clearance:.3g}, so neighbours in "
                          "formation will hold the gate shut: widen the spacing, or lower h, tau or max_speed")

    # ---- passed on to the wrapped controller --------------------------------------------------
    def filter_offsets(self):
        return self.inner.filter_offsets()

    def normalised_gains(self):
        return self.inner.normalised_gains()

    @property
    def damping(self):
        return getattr(self.inner, "damping", None)

    # ---- one step -----------------------------------------------------------------------------
    def _without_known_liars(self, obs):
        """The observation the wrapped controller gets: a report from an agent i has caught is known to be false, so with
        ``share_ranges`` agent i stops steering by it.  The worlds still read everything."""
        if not (self.share_ranges and self._known_faulty.any()):
            return obs
        return replace(obs, adj=obs.adj & ~self._known_faulty)

    def control_parts(self, obs):
        parts = self.inner.control_parts(self._without_known_liars(obs))
        advance, safety = parts["advance"], parts["safety"]
        cfg = self.cfg
        plans = predict_plan(obs.pos, obs.vel, advance + safety, cfg.dt, self.plan_steps, cfg.max_speed, cfg.max_accel)
        self.worlds = [build_possible_worlds(i, obs, self.histories[i], cfg.F, self.grid, cfg, self.tau, self.eps, self.true_faulty)
                       for i in range(obs.n)]
        count_worlds = self.size_every > 0 and obs.step % self.size_every == 0
        applied = advance.copy()
        self._detour.fill(False)
        self._retreat.fill(False)
        holds = prioritise(brake_command(obs.vel, cfg.dt, cfg.max_accel), safety, cfg.max_accel)
        for i, worlds in enumerate(self.worlds):
            self._open[i] = knows_safe(i, worlds, plans[:, i])
            if self.detours is not None and self.detours.out_of_patience(i, obs.step, not self._open[i]):
                move = self.detours.choose(i, obs, worlds, safety[i])
                if not (move.clear_ahead and self._open[i]):  # with the way clear, the wrapped controller's own move stands
                    self._open[i] = move.known_safe
                    if move.known_safe:
                        applied[i], self._detour[i] = move.command, True
                    elif move.command is not None:
                        holds[i], self._retreat[i] = move.command, True
            self._known_faulty[i], self._known_honest[i] = worlds.known_faulty, worlds.known_honest
            self._n_hypotheses[i] = worlds.n_hypotheses
            self._k_k_safe[i] = nested_safe_row(i, worlds.known_honest, obs, self._open[i])
            if count_worlds:
                self._log10_worlds[i] = worlds.log10_n_worlds
        self._requery = ~self._open & self.requery_enabled
        if self.share_ranges:
            self._ranges = np.where(obs.message_fresh, obs.ranges, np.inf)
            self._range_origin = obs.pos.copy()
        return {"advance": np.where(self._open[:, None], applied, holds), "safety": safety}

    def requery_requests(self) -> np.ndarray:
        return self._requery

    def broadcast_payload(self):
        payload = {"k_safe": self._open.copy(), "known_faulty": self._known_faulty.copy()}
        if self.share_ranges:
            payload[RANGES], payload[RANGE_ORIGIN] = self._ranges.copy(), self._range_origin.copy()
        return payload

    def diagnostics(self):
        return {**self.inner.diagnostics(), "gate_open": self._open, "k_safe": self._open, "detour": self._detour, "retreat": self._retreat,
                "known_faulty": self._known_faulty, "known_honest": self._known_honest, "k_k_safe": self._k_k_safe,
                "log10_worlds": self._log10_worlds, "n_hypotheses": self._n_hypotheses, "requery": self._requery}


@controller_registry.register("oracle")
class OracleGate(KnowledgeGate):
    """The same gate, told who is faulty (D1).  It measures the price of ignorance, so it is not decentralised in that
    one respect: where everyone is still has to come from messages and ranges."""

    def _who_is_told_faulty(self, sim) -> np.ndarray:
        return sim.compromised.copy()
