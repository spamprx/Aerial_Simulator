"""Experiment configuration.

One flat, JSON-serialisable dataclass describes an entire run.  Everything that varies
between experiments lives here, so a result can always be reproduced from
(config, seed).  Sweeps are expressed as overrides on a base config, e.g.

    cfg.with_overrides({"drop_prob": 0.2, "controller_params.k_p": 2.0})
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .core.world import World

FIXED_TOPOLOGIES = ("ring", "line", "complete", "star", "grid", "erdos_renyi", "custom")
ADVERSARY_MOTIONS = ("same", "hover", "intercept")
SENSING_KEYS = {"ranging", "sigma"}
DEFAULT_RANGE_SIGMA = 0.05
RANGE_NOISE_BOUND = 4.0  # ranging noise is cut off at this many sigma, so its error has a hard limit
GATE_KEYS = {"h", "tau", "eps", "requery", "size_every", "share_ranges", "patience"}
GATED_CONTROLLERS = ("gated", "oracle")


@dataclass
class SimConfig:
    # ---- fleet ------------------------------------------------------------------
    n_agents: int = 10
    dim: int = 3  # 2 for the 2-D first pass, 3 for the real thing

    # ---- time (fixed-step integration) ------------------------------------------
    dt: float = 0.05
    t_final: float = 20.0

    # ---- initial conditions -----------------------------------------------------
    arena_size: float = 20.0  # positions ~ U[0, arena_size]^dim
    init_speed: float = 0.0  # initial velocities ~ U(ball of this radius)
    min_init_separation: float = 1.0  # rejection sampling so nobody starts on top of anyone
    init_pos: Optional[List[List[float]]] = None  # (n_agents, dim); replaces the sampled positions
    init_vel: Optional[List[List[float]]] = None  # (n_agents, dim); replaces the sampled velocities
    world: Dict[str, Any] = field(default_factory=dict)

    # ---- communication ----------------------------------------------------------
    topology: str = "disk"  # "disk" (position based) or one of FIXED_TOPOLOGIES
    comm_radius: float = 18.0  # used by "disk"; shrink it to study fragmentation
    drop_prob: float = 0.0  # Bernoulli packet loss, per directed link per step
    symmetric_drops: bool = False  # True: a dropped link is dropped in both directions
    message_ttl: int = 0  # reuse a received message for this many dropped steps
    topology_params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"p": 0.3}

    # ---- physical limits (None = unlimited) -------------------------------------
    max_speed: Optional[float] = None
    max_accel: Optional[float] = None

    # ---- adversaries (F-local model; the attack library is Gona's module) -------
    F: int = 1  # the bound W-MSR and the gate assume; n_compromised is the actual number
    allow_beyond_assumption: bool = False  # permit n_compromised > F (degradation study)
    n_compromised: int = 0
    compromised_ids: Optional[List[int]] = None  # explicit ids override n_compromised
    attack: str = "none"
    attack_params: Dict[str, Any] = field(default_factory=dict)
    adversary_motion: str = "same"  # how compromised UAVs physically fly: see ADVERSARY_MOTIONS

    # ---- controller -------------------------------------------------------------
    controller: str = "consensus"
    controller_params: Dict[str, Any] = field(default_factory=dict)

    # ---- the epistemic gate, physical-layer sensing and the mission ---------------
    gate: Dict[str, Any] = field(default_factory=dict)
    sensing: Dict[str, Any] = field(default_factory=dict)  # {"ranging": true, "sigma": 0.05} (D5)
    mission: Dict[str, Any] = field(default_factory=dict)

    # ---- safety / bookkeeping ---------------------------------------------------
    d_safe: float = 1.0  # minimum allowed separation (used by metrics, later the STL spec)
    seed: int = 0
    record_lambda2: bool = True
    record_extras: bool = True  # per-step logs in SimResult.extras; large sweeps can turn this off

    # ------------------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(round(self.t_final / self.dt))

    @property
    def ranging(self) -> bool:
        return bool(self.sensing.get("ranging", False))

    @property
    def range_sigma(self) -> float:
        return float(self.sensing.get("sigma", DEFAULT_RANGE_SIGMA))

    def validate(self) -> "SimConfig":
        if not 1 <= self.n_agents <= 20:
            raise ValueError("n_agents must lie in [1, 20]")
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.dt <= 0 or self.t_final <= 0:
            raise ValueError("dt and t_final must be positive")
        World.from_dict(self.world, self.dim, self.arena_size)
        for name in ("init_pos", "init_vel"):
            value = getattr(self, name)
            if value is not None and np.shape(value) != (self.n_agents, self.dim):
                raise ValueError(f"{name} must have shape ({self.n_agents}, {self.dim})")
        if self.comm_radius <= 0:
            raise ValueError("comm_radius must be positive")
        if self.d_safe <= 0:
            raise ValueError("d_safe must be positive")
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError("drop_prob must lie in [0, 1]")
        if not isinstance(self.message_ttl, int) or isinstance(self.message_ttl, bool):
            raise ValueError("message_ttl must be an integer")
        if self.message_ttl < 0:
            raise ValueError("message_ttl must be non-negative")
        if self.topology != "disk" and self.topology not in FIXED_TOPOLOGIES:
            raise ValueError(f"topology must be 'disk' or one of {FIXED_TOPOLOGIES}")
        if self.adversary_motion not in ADVERSARY_MOTIONS:
            raise ValueError(f"adversary_motion must be one of {ADVERSARY_MOTIONS}")
        if set(self.sensing) - SENSING_KEYS:
            raise KeyError(f"unknown sensing keys: {sorted(set(self.sensing) - SENSING_KEYS)}")
        if self.range_sigma < 0:
            raise ValueError("sensing.sigma must be non-negative")
        if set(self.gate) - GATE_KEYS:
            raise KeyError(f"unknown gate keys: {sorted(set(self.gate) - GATE_KEYS)}")
        if self.controller in GATED_CONTROLLERS:
            check_gate_config(self, gate_settings(self))
        if not isinstance(self.F, int) or isinstance(self.F, bool) or not 0 <= self.F <= 2:
            raise ValueError("F must be an integer in [0, 2]")
        if self.compromised_ids is not None:
            if any(not 0 <= i < self.n_agents for i in self.compromised_ids):
                raise ValueError("compromised_ids out of range")
            actual = len(set(self.compromised_ids))
        elif not 0 <= self.n_compromised <= self.n_agents:
            raise ValueError("n_compromised must lie in [0, n_agents]")
        else:
            actual = self.n_compromised
        # F bounds each neighbourhood, not the fleet.  On a fixed graph Simulator checks exactly that.
        # A disk graph is unknown until the agents move, so here the total has to stay within F.
        if self.topology == "disk" and actual > self.F and not self.allow_beyond_assumption:
            raise ValueError(
                f"{actual} compromised agents exceeds the assumed F = {self.F}; "
                "raise F (at most 2) or set allow_beyond_assumption"
            )
        return self

    # ---- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**copy.deepcopy(d)).validate()

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "SimConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- sweeps -----------------------------------------------------------------
    def with_overrides(self, overrides: Optional[Dict[str, Any]] = None, **kw) -> "SimConfig":
        """Return a modified copy.  Keys may be dotted to reach into dict fields:
        ``{"controller_params.k_p": 2.0}``."""
        d = self.to_dict()
        merged = dict(overrides or {})
        merged.update(kw)
        for key, value in merged.items():
            head, *rest = key.split(".")
            if head not in d:
                raise KeyError(f"unknown config key '{head}'")
            if not rest:
                d[head] = value
            else:
                target = d[head]
                if not isinstance(target, dict):
                    raise KeyError(f"'{head}' is not a dict field, cannot override '{key}'")
                for part in rest[:-1]:
                    target = target.setdefault(part, {})
                target[rest[-1]] = value
        return SimConfig.from_dict(d)


def gate_settings(cfg: SimConfig, **given) -> Dict[str, Any]:
    """The gate's parameters (D10): ``h`` the cell side, ``tau`` the lookahead in seconds, ``eps`` the bound
    on range noise.  The caution knobs come from the defaults, then ``cfg.gate``, then the controller's
    own params, then ``given``."""
    settings = {"h": cfg.d_safe, "tau": 5.0 * cfg.dt, "eps": 0.1, "requery": True, "size_every": 10, "share_ranges": False,
                "patience": None}
    settings.update(cfg.gate)
    if cfg.controller in GATED_CONTROLLERS:
        settings.update({key: value for key, value in cfg.controller_params.items() if key in GATE_KEYS and value is not None})
    settings.update({key: value for key, value in given.items() if value is not None})
    return settings


def check_gate_config(cfg: SimConfig, settings: Dict[str, Any]) -> None:
    """What the gate's soundness rests on.  Without these checks, it could be wrong with no sign of it."""
    if cfg.topology != "disk":
        raise ValueError("the gate needs a disk graph: on a fixed graph, an agent never heard may still be alongside")
    if not cfg.ranging:
        raise ValueError('the gate needs ranging (D5): set "sensing": {"ranging": true}')
    if cfg.max_speed is None:
        raise ValueError("the gate needs a finite max_speed, to bound where a neighbour can get to")
    if settings["h"] <= 0 or settings["eps"] < 0:
        raise ValueError("gate h must be positive and eps non-negative")
    if settings["size_every"] < 0:
        raise ValueError("gate size_every must be non-negative")
    if settings["patience"] is not None and settings["patience"] < 0:
        raise ValueError("gate patience must be non-negative, or None to hold for as long as it takes")
    if settings["tau"] < cfg.dt:
        raise ValueError("gate tau must be at least one time step")
    needed = cfg.d_safe + 2.0 * cfg.max_speed * settings["tau"] + settings["h"]
    if cfg.comm_radius <= needed:
        raise ValueError(f"comm_radius {cfg.comm_radius} must exceed d_safe + 2 * max_speed * tau + h = {needed}: "
                         "an agent out of radio range could otherwise reach a UAV within the lookahead")
    # An honest report above max_speed would count as proof of lying, so no honest agent may start there.
    initial_speed = cfg.init_speed if cfg.init_vel is None else float(np.linalg.norm(cfg.init_vel, axis=1).max())
    if initial_speed > cfg.max_speed:
        raise ValueError(f"initial speed {initial_speed} exceeds max_speed {cfg.max_speed}")
