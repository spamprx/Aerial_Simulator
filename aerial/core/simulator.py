"""The simulator loop (T0.7). Tick order (A17):

1. every agent's ``beacon()`` claim;
2. sense at p(k): camera frame every ``sense_every`` ticks, UWB ranges at the transmit tick;
3. communicate: adjacency at p(k), drops, deliver fresh beacons (age 0), relay events one hop, serve re-queries;
4. ``step(obs)`` for every agent, liars included;
5. integrate to p(k+1) and log true separations;
6. record.

The simulator owns ground truth. Agents only ever receive their own ``AgentObservation``.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from ..agents.base import AgentContext, DroneAgent
from ..comms.messages import Event
from ..comms.network import Gossip, Network
from .config import Config
from .dynamics import step_double_integrator, step_yaw
from .result import Recorder, SimResult
from .rng import Streams
from .types import AgentObservation, ClaimedState, DroneState, ReceivedBeacon, StaticInfo

# a sensing hook: (sim, k) -> list of SensorFrame or None per agent. Filled in by aerial.sensing (Phase 1).
SenseHook = Callable[["Simulator", int], List[Optional[object]]]


class Simulator:
    def __init__(self, cfg: Config, agents: Sequence[DroneAgent], init_pos: np.ndarray,
                 init_yaw: Optional[np.ndarray] = None, liars: Optional[np.ndarray] = None,
                 base: Optional[np.ndarray] = None, sense_hook: Optional[SenseHook] = None,
                 world=None, streams: Optional[Streams] = None, static_extra=None):
        cfg.validate()
        self.cfg = cfg
        self.N = cfg.N
        if len(agents) != self.N or init_pos.shape != (self.N, 3):
            raise ValueError("need N agents and init_pos of shape (N, 3)")
        self.agents = list(agents)
        self.world = world
        self.rng = streams or Streams(cfg.sim.seed, self.N)
        self.pos = np.array(init_pos, float)
        self.vel = np.zeros((self.N, 3))
        self.yaw = np.zeros(self.N) if init_yaw is None else np.array(init_yaw, float)
        self.liars = np.zeros(self.N, bool) if liars is None else np.asarray(liars, bool)
        self.base = np.array(base if base is not None else self.pos.mean(axis=0), float)
        self.sense_hook = sense_hook
        self.net = Network(self.N, cfg.comms.R, cfg.comms.p_drop, self.rng.comm, self.rng.requery)
        self.gossip = Gossip(self.N, cfg.comms.max_events)
        self.seq = np.zeros(self.N, dtype=int)
        self.cache: List[dict] = [dict() for _ in range(self.N)]   # receiver -> {sender: (claim, seq)}
        self.requery = np.zeros(self.N, bool)
        self.k = 0
        for i, a in enumerate(self.agents):
            static = StaticInfo(i=i, N=self.N, F=cfg.F, arena=cfg.world.arena, zone=cfg.world.zone,
                                z_min=cfg.world.z_min, z_max=cfg.world.z_max, base=self.base.copy(),
                                dt=cfg.sim.dt, **((static_extra or {}).get(i, {}) if isinstance(static_extra, dict) else
                                                    static_extra[i] if static_extra is not None else {}))
            a.reset(AgentContext(static=static, cfg=cfg, rng=self.rng.ctrl[i]))
        self.statics = [a.ctx.static for a in self.agents]

    # ------------------------------------------------------------------ one tick
    def state(self, i: int) -> DroneState:
        return DroneState(self.pos[i].copy(), self.vel[i].copy(), float(self.yaw[i]))

    def step(self):
        cfg, N, k = self.cfg, self.N, self.k
        t = k * cfg.sim.dt
        # 1. beacons
        claims: List[ClaimedState] = [a.beacon(self.state(i), k) for i, a in enumerate(self.agents)]
        # 2. sense
        frames = [None] * N
        if self.sense_hook is not None and k % cfg.sim.sense_every == 0:
            frames = self.sense_hook(self, k)
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        noise = np.clip(self.rng.sense.standard_normal((N, N)), -4, 4) * cfg.sensor.range_sigma
        ranges = dist + noise
        # 3. communicate
        A = self.net.adjacency(self.pos, self.requery)
        new_events = self.gossip.exchange(A)
        obs_list = []
        for i in range(N):
            beacons = {}
            for j in np.flatnonzero(A[i]):
                self.cache[i][int(j)] = (claims[j], k)
                beacons[int(j)] = ReceivedBeacon(int(j), claims[j], k, 0, float(ranges[i, j]))
            for j, (c, s) in self.cache[i].items():
                if j not in beacons:
                    beacons[j] = ReceivedBeacon(j, c, s, k - s, None)
            obs_list.append(AgentObservation(t=t, k=k, i=i, own=self.state(i), frame=frames[i],
                                             beacons=beacons, events=tuple(new_events[i]),
                                             static=self.statics[i]))
        # 4. decide
        u = np.zeros((N, 3))
        yaw_rate = np.zeros(N)
        requery = np.zeros(N, bool)
        for i, a in enumerate(self.agents):
            act = a.step(obs_list[i])
            u[i] = act.u
            yaw_rate[i] = act.yaw_rate
            requery[i] = act.requery
            for d in act.outbox:
                self.seq[i] += 1
                self.gossip.originate(i, Event(origin=i, seq=int(self.seq[i]), t=t, kind=d.kind,
                                               payload=d.payload))
        self.requery = requery
        # 5. integrate
        self.pos, self.vel, u_app = step_double_integrator(self.pos, self.vel, u, cfg.sim.dt, cfg.drone.a_max,
                                                           cfg.drone.v_h, cfg.drone.v_z)
        self.yaw = step_yaw(self.yaw, yaw_rate, cfg.sim.dt, cfg.drone.yaw_rate)
        self.k += 1
        return A, u_app

    # ------------------------------------------------------------------ full run
    def run(self, steps: Optional[int] = None) -> SimResult:
        K = int(round(self.cfg.sim.T / self.cfg.sim.dt)) if steps is None else steps
        rec = Recorder(self.cfg.sim.record_level)
        pos, vel, yaw, us, lam = [self.pos.copy()], [self.vel.copy()], [self.yaw.copy()], [], []
        min_sep = np.inf
        for _ in range(K):
            A, u = self.step()
            pos.append(self.pos.copy()); vel.append(self.vel.copy()); yaw.append(self.yaw.copy())
            us.append(u)
            lam.append(self.net.lambda2(A) if rec.level != "minimal" else np.nan)
            rec.add("adj", A)
            rec.add_diagnostics([a.diagnostics() for a in self.agents])
            d = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            min_sep = min(min_sep, float(d.min()))
        extras = rec.stacked()
        adj = extras.pop("adj", np.zeros((0, self.N, self.N), bool))
        return SimResult(t=np.arange(K + 1) * self.cfg.sim.dt, pos=np.array(pos), vel=np.array(vel),
                         yaw=np.array(yaw), u=np.array(us), adj=adj, lambda2=np.array(lam), liars=self.liars,
                         config=self.cfg, extras=extras,
                         metrics={"min_separation": min_sep, "msgs_sent": self.gossip.sent.tolist()})
