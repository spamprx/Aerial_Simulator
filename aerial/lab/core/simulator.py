"""Simulator core: state, the step loop, recording.

One step is:

    1. sense      each agent broadcasts its state; compromised agents' broadcasts pass
                  through the Attack; the comm model samples this step's graph G(t)
    2. decide     the controller maps the Observation to accelerations u
    3. integrate  double-integrator update with optional speed / accel saturation

RNG streams (initial conditions, communication, attack, controller, re-query, sensing)
are spawned from one seed, so changing e.g. the controller never changes the packet drops
or the attack noise.  That makes "naive fleet vs gated fleet under the identical attack" a
true paired comparison.
"""
from __future__ import annotations

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..config import RANGE_NOISE_BOUND, SimConfig
from ..coordination.base import Controller
from ..network.comm import CommModel, FixedComm, build_comm_model
from ..network.graph import algebraic_connectivity, pairwise_distances
from ..network.messages import MessageMemory
from ..registry import attack_registry, controller_registry
from ..resilience.adversary import adversary_commands
from ..resilience.attacks import Attack
from ..resilience.flocal import is_f_local, pick_compromised_f_local
from ..stability import check_controller_stability
from .dynamics import step_double_integrator
from .world import World


# ----------------------------------------------------------------------------------
class RecordedDraws:
    """A random generator that keeps what it hands out, so that two runs can be compared draw by draw (T6.1).
    It gives exactly the numbers the generator would give on its own."""

    def __init__(self, generator: np.random.Generator):
        self._generator = generator
        self._drawn: List[np.ndarray] = []

    def __getattr__(self, name: str):
        value = getattr(self._generator, name)
        if not callable(value):
            return value

        def draw(*args, **kwargs):
            drawn = value(*args, **kwargs)
            self._drawn.append(np.ravel(drawn).astype(float))
            return drawn
        return draw

    def take(self) -> np.ndarray:
        """(M,) everything drawn since the last time this was asked."""
        drawn = np.concatenate(self._drawn) if self._drawn else np.empty(0)
        self._drawn = []
        return drawn


# ----------------------------------------------------------------------------------
@dataclass
class Observation:
    """Everything the controller may look at on one step.

    ``pos``/``vel``      true own state of every agent (agent i may use only row i).
    ``adj[i, j]``        agent i can use agent j's message this step (including a
                         bounded-TTL cached message).
    ``msg_pos/msg_vel``  what each agent *broadcast* this step (possibly falsified if
                         compromised).  This is not what a receiver holds once
                         ``message_ttl > 0``: read messages through ``received(i)`` or
                         ``msg_pos_for(i)``, which are correct for every TTL.
    ``message_fresh``    messages received on this step, before cache reuse.
    ``message_age``      age in steps of each available message (infinity if unavailable).
    ``held_pos/held_vel``  ``(N, N, d)`` per-receiver cache, indexed ``[receiver, sender]``;
                         None when ``message_ttl == 0``, where nothing is cached.
    ``ranges``           ``(N, N)`` physical-layer distance for each message held, measured when it
                         arrived; infinity where nothing is held, and None without ranging (D5).
                         Nothing a liar broadcasts can change it.
    ``msg_payload``      what each agent added to its broadcast (``Controller.broadcast_payload``), a
                         dict of ``(N, ...)`` arrays.  It has the same availability as ``adj`` and arrives one
                         step late, so it is empty on the first step.  Read it through ``payload_for(i, key)``.
    ``held_payload``     per-receiver cache of the same, each ``(N, N, ...)``; None with ``message_ttl == 0``.
    """

    t: float
    step: int
    dt: float
    pos: np.ndarray
    vel: np.ndarray
    adj: np.ndarray
    msg_pos: np.ndarray
    msg_vel: np.ndarray
    world: World
    message_fresh: np.ndarray
    message_age: np.ndarray
    held_pos: Optional[np.ndarray] = None
    held_vel: Optional[np.ndarray] = None
    ranges: Optional[np.ndarray] = None
    msg_payload: Dict[str, np.ndarray] = field(default_factory=dict)
    held_payload: Optional[Dict[str, np.ndarray]] = None

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def in_neighbors(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.adj[i])

    def msg_pos_for(self, i: int) -> np.ndarray:
        """(N, d) positions as agent i holds them; only rows with ``adj[i, j]`` are valid."""
        return self.msg_pos if self.held_pos is None else self.held_pos[i]

    def msg_vel_for(self, i: int) -> np.ndarray:
        """(N, d) velocities as agent i holds them; only rows with ``adj[i, j]`` are valid."""
        return self.msg_vel if self.held_vel is None else self.held_vel[i]

    def payload_for(self, i: int, key: str) -> Optional[np.ndarray]:
        """(N, ...) that part of the payload as agent i holds it, or None if nobody has sent one yet.
        Only rows with ``adj[i, j]`` are valid."""
        if self.held_payload is None:
            return self.msg_payload.get(key)
        held = self.held_payload.get(key)
        return None if held is None else held[i]

    def received(self, i: int):
        """(ids, positions, velocities) of the messages agent i actually holds."""
        idx = self.in_neighbors(i)
        return idx, self.msg_pos_for(i)[idx], self.msg_vel_for(i)[idx]

    def received_sums(self):
        """(pos_sum, vel_sum), each (N, d): the sum of the messages every agent holds."""
        A = self.adj.astype(float)
        if self.held_pos is None:
            return A @ self.msg_pos, A @ self.msg_vel
        return np.einsum("ij,ijk->ik", A, self.held_pos), np.einsum("ij,ijk->ik", A, self.held_vel)


@dataclass
class SimResult:
    """Full recorded trajectory of one run (K = n_steps)."""

    cfg: SimConfig
    t: np.ndarray  # (K+1,)
    pos: np.ndarray  # (K+1, N, d)
    vel: np.ndarray  # (K+1, N, d)
    u: np.ndarray  # (K, N, d)   applied (post-saturation) accelerations
    adj: np.ndarray  # (K, N, N)   receive graph used at each step
    lambda2: np.ndarray  # (K,)        algebraic connectivity of G(t) (nan if not recorded)
    compromised: np.ndarray  # (N,) bool
    # (K, ...) per-step logs: controller diagnostics, plus msg_pos and msg_vel (the broadcasts,
    # (K, N, d)), drop_keep (the raw Bernoulli keep mask, (K, N, N)), f_local_ok (whether the F-local
    # assumption held on this step's graph, (K,)) and attack_noise (what rng_attack drew on each step, (K, M),
    # padded with NaN, with what was drawn while setting up on the first step).  Empty when cfg.record_extras is False.
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def world(self) -> World:
        """World geometry for this run, reconstructed from the serializable config."""
        return World.from_dict(self.cfg.world, self.cfg.dim, self.cfg.arena_size)

    @property
    def honest(self) -> np.ndarray:
        return ~self.compromised

    def save(self, path) -> None:
        """Write a compact .npz (arrays + config JSON)."""
        np.savez_compressed(
            path,
            t=self.t, pos=self.pos, vel=self.vel, u=self.u, adj=self.adj,
            lambda2=self.lambda2, compromised=self.compromised,
            cfg=json.dumps(self.cfg.to_dict()),
            **{f"extras__{key}": value for key, value in self.extras.items()},
        )

    @classmethod
    def load(cls, path) -> "SimResult":
        z = np.load(Path(path), allow_pickle=False)
        cfg = SimConfig.from_dict(json.loads(str(z["cfg"])))
        extras = {name[len("extras__"):]: z[name] for name in z.files if name.startswith("extras__")}
        return cls(cfg, z["t"], z["pos"], z["vel"], z["u"], z["adj"], z["lambda2"], z["compromised"], extras)


# ----------------------------------------------------------------------------------
def _padded_with_nan(rows: List[np.ndarray]) -> np.ndarray:
    """(K, M) from K rows of up to M values: nothing makes an attack draw as many on every step."""
    padded = np.full((len(rows), max(len(row) for row in rows)), np.nan)
    for k, row in enumerate(rows):
        padded[k, :len(row)] = row
    return padded


def sample_initial_state(cfg: SimConfig, rng: np.random.Generator):
    """Uniform positions in the arena with a minimum pairwise separation (rejection
    sampling; the requirement is relaxed gracefully if the arena is too crowded)."""
    n, d = cfg.n_agents, cfg.dim
    pos = np.empty((0, d))
    sep = cfg.min_init_separation
    tries = 0
    while len(pos) < n:
        cand = rng.uniform(0.0, cfg.arena_size, size=d)
        if len(pos) == 0 or np.min(np.linalg.norm(pos - cand, axis=1)) >= sep:
            pos = np.vstack([pos, cand])
            tries = 0
        else:
            tries += 1
            if tries > 500:  # too crowded: relax and keep going
                sep *= 0.9
                tries = 0
    vel = np.zeros((n, d))
    if cfg.init_speed > 0:
        direction = rng.normal(size=(n, d))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        vel = direction * rng.uniform(0.0, cfg.init_speed, size=(n, 1))
    return pos, vel


class Simulator:
    """Owns the world state and advances it.  Controllers and attacks are pluggable."""

    def __init__(
        self,
        cfg: SimConfig,
        controller: Optional[Controller] = None,
        attack: Optional[Attack] = None,
        comm: Optional[CommModel] = None,
        init_pos: Optional[np.ndarray] = None,
        init_vel: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg.validate()
        self.n, self.dim = cfg.n_agents, cfg.dim
        self.world = World.from_dict(cfg.world, cfg.dim, cfg.arena_size)
        self.messages = MessageMemory(self.n, self.dim, cfg.message_ttl)

        # New streams go at the end: spawn() numbers its children, so this keeps the earlier
        # ones, and with them every recorded run, unchanged.
        init_ss, comm_ss, attack_ss, ctrl_ss, requery_ss, sense_ss = np.random.SeedSequence(cfg.seed).spawn(6)
        self.rng_init = np.random.default_rng(init_ss)
        self.rng_comm = np.random.default_rng(comm_ss)
        self.rng_attack = RecordedDraws(np.random.default_rng(attack_ss))
        self.rng_ctrl = np.random.default_rng(ctrl_ss)
        self.rng_requery = np.random.default_rng(requery_ss)
        self.rng_sense = np.random.default_rng(sense_ss)

        self.controller = controller or controller_registry.create(cfg.controller, **cfg.controller_params)
        self.attack = attack or attack_registry.create(cfg.attack, **cfg.attack_params)
        self.comm = comm or build_comm_model(cfg, self.rng_comm)
        self._attack_uses_graph = "adj" in inspect.signature(self.attack.corrupt).parameters

        self._init_pos = cfg.init_pos if init_pos is None else init_pos
        self._init_vel = cfg.init_vel if init_vel is None else init_vel
        self.hooks: List[Callable[["Simulator", Observation, np.ndarray], None]] = []
        self._requery = np.zeros(self.n, dtype=bool)
        self._payload_out = {}
        self._attack_noise = np.empty(0)
        self.reset()
        check_controller_stability(self.controller, cfg, self.comm)

    # ---- setup ------------------------------------------------------------------
    def reset(self) -> None:
        if self._init_pos is not None:
            pos = np.array(self._init_pos, dtype=float)
            vel = np.zeros_like(pos)
        else:
            pos, vel = sample_initial_state(self.cfg, self.rng_init)
        if self._init_vel is not None:
            vel = np.array(self._init_vel, dtype=float)
        if pos.shape != (self.n, self.dim) or vel.shape != pos.shape:
            raise ValueError(f"initial state must have shape ({self.n}, {self.dim})")
        self.pos, self.vel = pos, vel
        self.t, self.k = 0.0, 0
        self.compromised = self._pick_compromised()
        self.controller.reset(self)
        self.attack.reset(self)  # after the controller, whose frame a covert attack has to work in
        self.messages.reset()
        self._requery = np.zeros(self.n, dtype=bool)
        self._payload_out = {}

    def _pick_compromised(self) -> np.ndarray:
        mask = np.zeros(self.n, dtype=bool)
        cfg = self.cfg
        if cfg.compromised_ids is not None:
            mask[list(cfg.compromised_ids)] = True
        elif cfg.n_compromised > cfg.F and isinstance(self.comm, FixedComm) and not cfg.allow_beyond_assumption:
            mask = pick_compromised_f_local(self.comm.base, cfg.n_compromised, cfg.F, self.rng_attack)
        elif cfg.n_compromised > 0:
            mask[self.rng_attack.choice(self.n, cfg.n_compromised, replace=False)] = True
        if isinstance(self.comm, FixedComm) and not cfg.allow_beyond_assumption \
                and not is_f_local(self.comm.base, mask, cfg.F):
            # Lost packets only remove links, so a set that is F-local on the full graph stays so.
            raise ValueError(f"compromised agents {np.flatnonzero(mask).tolist()} are not F-local with F = {cfg.F}: "
                             "an honest agent hears more than F of them. Set allow_beyond_assumption to study this.")
        return mask

    def add_hook(self, fn: Callable[["Simulator", Observation, np.ndarray], None]) -> None:
        """``fn(sim, obs, u)`` is called every step after the controller (logging, gating
        diagnostics, live dashboard frames ...)."""
        self.hooks.append(fn)

    # ---- one step ---------------------------------------------------------------
    def sense(self) -> Observation:
        adj = self._with_requeries(self.comm.adjacency(self.pos, self.rng_comm))
        hears = {"adj": adj} if self._attack_uses_graph else {}
        msg_pos, msg_vel = self.attack.corrupt(
            self.t, self.pos, self.vel, self.compromised, self.rng_attack, **hears
        )
        self._attack_noise = self.rng_attack.take()
        msg_pos = np.asarray(msg_pos)
        msg_vel = np.asarray(msg_vel)
        measured = self._measure_ranges()
        payload = self.attack.corrupt_payload(self.t, self._payload_out, self.compromised) if self._payload_out else {}
        self.messages.update(msg_pos, msg_vel, adj, measured, payload)
        held_pos, held_vel, available, age = self.messages.view(msg_pos, msg_vel, adj)
        ranges = None if measured is None else self.messages.view_ranges(measured, adj)
        cached = self.cfg.message_ttl > 0
        return Observation(
            self.t, self.k, self.cfg.dt, self.pos.copy(), self.vel.copy(),
            available, msg_pos.copy(), msg_vel.copy(), self.world, adj.copy(), age,
            held_pos if cached else None, held_vel if cached else None, ranges, payload,
            {key: value.copy() for key, value in self.messages.payload.items()} if cached else None,
        )

    def _with_requeries(self, adj: np.ndarray) -> np.ndarray:
        """Give each agent that asked last step one more attempt from every neighbour in range (D4).

        The attempts are drawn every step whether anyone asked or not, from their own stream, which is
        what keeps the first-attempt drops of a naive and a gated run identical.
        """
        keep = self.rng_requery.random((self.n, self.n)) >= self.cfg.drop_prob
        if not self._requery.any():
            return adj
        adj = adj.copy()
        adj[self._requery] |= (self.comm.reachable(self.pos) & keep)[self._requery]
        return adj

    def _measure_ranges(self) -> Optional[np.ndarray]:
        """(N, N) true distances plus noise, from a full block drawn every step so that runs stay paired.

        This uses true positions, so no attack can touch it.  The noise is cut off at
        ``RANGE_NOISE_BOUND`` sigma, because the gate's soundness needs an error with a hard limit.
        """
        if not self.cfg.ranging:
            return None
        noise = np.clip(self.rng_sense.standard_normal((self.n, self.n)), -RANGE_NOISE_BOUND, RANGE_NOISE_BOUND)
        return np.maximum(pairwise_distances(self.pos) + self.cfg.range_sigma * noise, 0.0)

    def step(self):
        """Advance one step; returns (observation, applied_u)."""
        obs = self.sense()
        u = np.asarray(self.controller.control(obs), dtype=float)
        if u.shape != (self.n, self.dim):
            raise ValueError(f"controller returned shape {u.shape}, expected {(self.n, self.dim)}")
        requests = self.controller.requery_requests()
        self._requery = np.zeros(self.n, dtype=bool) if requests is None else np.array(requests, dtype=bool)
        self._payload_out = self.controller.broadcast_payload()
        u = adversary_commands(self.cfg.adversary_motion, u, self.pos, self.vel, self.compromised, self.cfg.dt,
                               self.cfg.max_speed, self.cfg.max_accel)
        for hook in self.hooks:
            hook(self, obs, u)
        self.pos, self.vel, u_applied = step_double_integrator(
            self.pos, self.vel, u, self.cfg.dt, self.cfg.max_speed, self.cfg.max_accel
        )
        self.k += 1
        self.t = self.k * self.cfg.dt
        return obs, u_applied

    # ---- full run ---------------------------------------------------------------
    def run(self, n_steps: Optional[int] = None) -> SimResult:
        K = self.cfg.n_steps if n_steps is None else n_steps
        N, d = self.n, self.dim
        T = np.empty(K + 1)
        P = np.empty((K + 1, N, d))
        V = np.empty((K + 1, N, d))
        U = np.empty((K, N, d))
        ADJ = np.empty((K, N, N), dtype=bool)
        LAM = np.full(K, np.nan)
        log = defaultdict(list)

        T[0], P[0], V[0] = self.t, self.pos, self.vel
        for k in range(K):
            obs, u = self.step()
            U[k], ADJ[k] = u, obs.adj
            if self.cfg.record_lambda2:
                LAM[k] = algebraic_connectivity(obs.adj)
            if self.cfg.record_extras:
                self._log_extras(log, obs)
            T[k + 1], P[k + 1], V[k + 1] = self.t, self.pos, self.vel
        noise = log.pop("attack_noise", None)
        extras = {key: np.stack(values) for key, values in log.items()}
        if noise is not None:
            extras["attack_noise"] = _padded_with_nan(noise)
        return SimResult(self.cfg, T, P, V, U, ADJ, LAM, self.compromised.copy(), extras)

    def _log_extras(self, log: Dict[str, list], obs: Observation) -> None:
        row = dict(self.controller.diagnostics())
        row.update(msg_pos=obs.msg_pos, msg_vel=obs.msg_vel, f_local_ok=is_f_local(obs.adj, self.compromised, self.cfg.F),
                   attack_noise=self._attack_noise)
        if self.comm.last_keep is not None:
            row["drop_keep"] = self.comm.last_keep
        for key, value in row.items():
            log[key].append(np.array(value))  # a copy: controllers may reuse their buffers

    # ---- convenience ------------------------------------------------------------
    def distances(self) -> np.ndarray:
        return pairwise_distances(self.pos)
