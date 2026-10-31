"""The liar (T3.1): an ordinary fleet member with the same sensors and inbox (A14), that fakes coverage and position.

During transit it behaves honestly and broadcasts its true pose, so it is not caught before it lies. From
``attack.start_time`` it:

- moves physically according to ``attack.motion`` (``hover`` where it is, ``loiter`` around it, ...);
- broadcasts a fake pose (``attack.position_lie``):
    * ``true_pose``         - its real pose (only the coverage is fake; caught by audits R5),
    * ``fake_sweep_naive``  - a lawnmower over its claim region, ignoring ranging (caught by R1 at once),
    * ``range_aware``       - the lawnmower point pulled onto the range shells of whoever hears it, when possible,
    * ``radio_shadow``      - like ``range_aware``, but it also keeps away from other drones;
- claims CLEARED for claim-region cells near its (fake) pose, and suppresses every survivor report.
"""
from __future__ import annotations

import numpy as np

from ..agents.mission import MissionAgent
from ..comms.messages import Draft, EventKind
from ..core.types import Action, ClaimedState
from ..registry import liar_registry


@liar_registry.register("liar")
class LiarAgent(MissionAgent):
    is_liar = True

    def __init__(self, claim_region: np.ndarray, fleet: str = "naive"):
        super().__init__(fleet=fleet)
        self.claim_region = np.asarray(claim_region, int)

    def reset(self, ctx):
        super().reset(ctx)
        a = self.cfg.attack
        self.a = a
        self.lying = False
        self.hover_at = None
        self.claimed = np.zeros(self.cov.n_cells, bool)
        cen = self.cov.centres[self.claim_region]
        # a lawnmower over the claim region's bounding box, 6 m lanes
        x0, y0 = cen.min(axis=0)
        x1, y1 = cen.max(axis=0)
        lanes = np.arange(x0 + 6.0, x1 + 1e-6, 12.0)
        pts = []
        for k, x in enumerate(lanes):
            ys = (y0, y1) if k % 2 == 0 else (y1, y0)
            pts += [(x, ys[0]), (x, ys[1])]
        self.sweep = np.array(pts)
        seg = np.linalg.norm(np.diff(self.sweep, axis=0), axis=1)
        self.sweep_s = np.concatenate([[0], np.cumsum(seg)])
        self.fake = None
        self.track_origin = None
        self.track_t0 = 0.0
        self.lead = 0.0
        self.ranges = {}

    # -------------------------------------------------------------- broadcast
    def beacon(self, own, k):
        t = k * self.ctx.static.dt
        if not self.lying or self.a.position_lie == "true_pose":
            return super().beacon(own, k)
        p = self._fake_pose(t, own.pos)
        self.fake = p
        v = np.zeros(3)
        if self._last_fake is not None:
            v = (p - self._last_fake) / self.ctx.static.dt
            v[:2] *= min(1.0, 0.95 * self.cfg.drone.v_h / (np.linalg.norm(v[:2]) + 1e-9))
            v[2] = np.clip(v[2], -0.95 * self.cfg.drone.v_z, 0.95 * self.cfg.drone.v_z)
        self._last_fake = p
        return ClaimedState(p.copy(), v, own.yaw)

    _last_fake = None

    def _fake_pose(self, t, true_pos):
        # the fake track starts at the liar's real position (no teleport, R2) and flies to the sweep at 0.9 v_h
        if self.track_origin is None:
            self.track_origin = np.array([true_pos[0], true_pos[1]])
            self.track_t0 = t
            self.lead = float(np.linalg.norm(self.sweep[0] - self.track_origin))
        s = (t - self.track_t0) * 0.9 * self.cfg.drone.v_h
        if s < self.lead:
            xy = self.track_origin + (self.sweep[0] - self.track_origin) * (s / max(self.lead, 1e-9))
            return np.array([xy[0], xy[1], self.layers[self.layer_idx]]) if not self.ranges or \
                self.a.position_lie == "fake_sweep_naive" else self._fit_ranges(
                    np.array([xy[0], xy[1], self.layers[self.layer_idx]]), true_pos)
        s -= self.lead
        s = s % self.sweep_s[-1] if self.sweep_s[-1] > 0 else 0.0
        i = int(np.clip(np.searchsorted(self.sweep_s, s) - 1, 0, len(self.sweep) - 2))
        f = (s - self.sweep_s[i]) / max(1e-9, self.sweep_s[i + 1] - self.sweep_s[i])
        xy = self.sweep[i] + f * (self.sweep[i + 1] - self.sweep[i])
        p = np.array([xy[0], xy[1], self.layers[self.layer_idx]])
        if self.a.position_lie in ("range_aware", "radio_shadow") and self.ranges:
            p = self._fit_ranges(p, true_pos)
        return p

    def _fit_ranges(self, target, true_pos):
        """Closest point to ``target`` that matches every receiver's true range (Gauss-Newton on the ranges).

        With 1-2 receivers such points exist away from the truth; with 3+ (non-collinear) they collapse onto the
        true position, so the lie cannot survive and the liar falls back to the truth or gets caught.
        """
        recv = [(np.asarray(q), np.linalg.norm(true_pos - q)) for q in self.ranges.values()]
        p = target.copy()
        for _ in range(200):
            J, r = [], []
            for q, d in recv:
                v = p - q
                n = np.linalg.norm(v) + 1e-9
                J.append(v / n); r.append(n - d)
            J, r = np.array(J), np.array(r)
            if np.abs(r).max() < 0.02:
                break
            step = np.linalg.lstsq(J, r, rcond=None)[0]
            p = p - step
        p[2] = np.clip(p[2], self.ctx.static.z_min, self.ctx.static.z_max)
        if recv and max(abs(np.linalg.norm(p - q) - d) for q, d in recv) > 0.05:
            return true_pos.copy()
        return p

    # -------------------------------------------------------------- behaviour
    def _shadow(self, obs):
        """Tail the nearest honest drone from just above and behind it: its camera looks forward and down, so it
        cannot see the liar there. The liar skips its own separation (a compromised drone does not care), but it
        never aims at a drone: it only flies recklessly close. Honest drones that trust the liar's fake position
        cannot avoid it; a gated drone's UWB range still says something is close."""
        pos, vel = obs.own.pos, obs.own.vel
        cand = {j: b.claim for j, b in obs.beacons.items() if b.age <= 2}
        if self._target not in cand or obs.t - self._target_t > 25.0:
            if cand:
                self._target = min(cand, key=lambda j: np.linalg.norm(cand[j].pos - pos))
                self._target_t = obs.t
        if self._target not in cand:
            return 1.5 * (-vel)
        c = cand[self._target]
        hv = c.vel[:2]
        back = -hv / np.linalg.norm(hv) if np.linalg.norm(hv) > 0.3 else np.zeros(2)
        goal = c.pos + np.array([back[0] * 1.0, back[1] * 1.0, 1.0])
        goal[2] = np.clip(goal[2], self.ctx.static.z_min, self.ctx.static.z_max)
        return 2.5 * (goal - pos) + 2.0 * (c.vel - vel)

    _target = None
    _target_t = -1e9
    _park_at = None
    _park_t = -1e9

    def _park(self, obs):
        """Park in the busiest lane: every 30 s move to where the honest drones are (the centroid of the beacons it
        hears) and hover there, stationary, at the height of the nearest drone. It never moves into anyone; drones
        that trust its fake position fly into it, while a gated drone's range check makes it hold."""
        pos, vel = obs.own.pos, obs.own.vel
        cand = [b.claim.pos for b in obs.beacons.values() if b.age <= 2]
        if (self._park_at is None or obs.t - self._park_t > 30.0) and cand:
            c = np.mean(cand, axis=0)
            near = min(cand, key=lambda q: np.linalg.norm(q - pos))
            self._park_at = np.array([c[0], c[1], near[2]])
            self._park_t = obs.t
        if self._park_at is None:
            return -1.5 * vel
        d = self._park_at - pos
        v = d * min(1.0, 3.0 / (np.linalg.norm(d) + 1e-9))
        return 1.8 * (v - vel) + self._terrain_only(obs)

    def _ambush(self, obs):
        """Sit still in a drone's path: pick the nearest moving honest drone, fly (keeping separation) to a spot
        about 10 m ahead of it at its height, then hover still for a while. It never moves into a drone; an honest
        drone that trusts the liar's fake position flies into it, a gated drone's range check makes it hold."""
        pos, vel = obs.own.pos, obs.own.vel
        cand = {j: b.claim for j, b in obs.beacons.items() if b.age <= 2 and np.linalg.norm(b.claim.vel[:2]) > 1.0}
        if (self._park_at is None or obs.t - self._park_t > 14.0) and cand:
            j = min(cand, key=lambda k: np.linalg.norm(cand[k].pos - pos))
            c = cand[j]
            ahead = c.vel[:2] / np.linalg.norm(c.vel[:2])
            self._park_at = np.array([c.pos[0] + 10 * ahead[0], c.pos[1] + 10 * ahead[1], c.pos[2]])
            self._park_t = obs.t
        if self._park_at is None:
            return -1.5 * vel
        d = self._park_at - pos
        if obs.t - self._park_t < 6.0 and np.linalg.norm(d) > 0.5:      # reposition, politely
            v = d * min(1.0, 4.5 / (np.linalg.norm(d) + 1e-9))
            return 1.8 * (v - vel) + self._safety(obs)
        return 1.8 * (0.5 * d - vel)                                     # then sit still

    def _terrain_only(self, obs):
        """The liar still avoids buildings (it is reckless, not suicidal): the safety terms minus drone separation."""
        saved = obs.frame
        u = self._safety(obs.__class__(**{**obs.__dict__, "beacons": {}, "frame": None}))
        return u

    def step(self, obs):
        fresh = {j: b.claim.pos for j, b in obs.beacons.items() if b.age == 0}
        if obs.t >= self.a.start_time and self.mode != "TRANSIT" and not self.lying and not self.a.seek_quiet:
            self.lying = True
        if obs.t >= self.a.start_time and self.mode != "TRANSIT" and not self.lying:
            # covert: start only somewhere quiet (at most 2 drones hear me), or after waiting too long
            self.seeking = True
            if len(fresh) <= 2 or obs.t >= self.a.start_time + 40.0:
                self.lying, self.seeking = True, False
        if getattr(self, "seeking", False):
            away = sum((obs.own.pos - q) for q in fresh.values()) if fresh else np.zeros(3)
            away[2] = 0.0
            v = 2.5 * away / (np.linalg.norm(away) + 1e-9)
            u = 1.5 * (v - obs.own.vel) + self._safety(obs)
            u[2] = 1.0 * (self.layers[self.layer_idx] - obs.own.pos[2]) - obs.own.vel[2]
            return Action(u=u, yaw_rate=0.3, outbox=[])
        if not self.lying:
            act = super().step(obs)
            act.outbox = [d for d in act.outbox if d.kind not in (EventKind.SURVIVOR, EventKind.LOOKS)]
            return act
        self.ranges = {j: b.claim.pos for j, b in obs.beacons.items() if b.age == 0}
        if self.hover_at is None:
            self.hover_at = obs.own.pos.copy()
        pos, vel = obs.own.pos, obs.own.vel
        goal = self.hover_at.copy()
        if self.a.motion == "loiter":
            goal[:2] += 3.0 * np.array([np.cos(obs.t / 4), np.sin(obs.t / 4)])
        if self.a.position_lie == "radio_shadow" and self.ranges:
            away = sum((pos - q) for q in self.ranges.values())
            goal[:2] = pos[:2] + 3.0 * away[:2] / (np.linalg.norm(away[:2]) + 1e-9)
        u = 1.5 * (0.5 * (goal - pos) - vel) + self._safety(obs)
        if self.a.motion == "shadow":
            u = self._shadow(obs)
        elif self.a.motion == "park":
            u = self._park(obs)
        elif self.a.motion == "ambush":
            u = self._ambush(obs)
        out = []
        ref = self.fake if self.fake is not None else pos
        near = self.claim_region[np.linalg.norm(self.cov.centres[self.claim_region] - ref[:2], axis=1) < 13.0]
        if self.a.claim_all:
            near = self.claim_region
        new = near[~self.claimed[near]]
        if len(new):
            self.claimed[new] = True
            out.append(Draft(EventKind.CLEARED, {"cells": new, "pose": ref.copy(), "t": obs.t}))
        if obs.k % self.cfg.comms.pose_every == 0:
            out.append(Draft(EventKind.POSE, ref.copy()))
        self.mode = "EXPLORE"
        return Action(u=u, yaw_rate=0.3, outbox=out)
