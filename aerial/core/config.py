"""Configuration for AERIAL v2 (T0.3).

Nested dataclasses, JSON round-trips, dotted overrides (``"comms.R=30"``) and validation, including the
geometry checks that keep the knowledge gate from deadlocking honest drones (review finding 8).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Union

FLEETS = ("unfiltered", "naive", "gated", "oracle")
LIAR_MOTIONS = ("hover", "loiter", "shadow", "park", "ambush", "partial_sweep")
LIAR_POSITION_LIES = ("true_pose", "fake_sweep_naive", "range_aware", "radio_shadow")
RECORD_LEVELS = ("minimal", "standard", "full")
PARTITIONS = ("voronoi", "static_strips")


@dataclass
class SimCfg:
    dt: float = 0.1
    T: float = 240.0
    seed: int = 0
    record_level: str = "standard"
    sense_every: int = 2          # sensor rate = 1 / (sense_every * dt) = 5 Hz


@dataclass
class WorldCfg:
    scenario: str = "district"
    arena: float = 120.0          # square arena side (m)
    zone: float = 100.0           # search-zone side (m)
    terrain_cell: float = 1.0
    coverage_cell: float = 2.0
    z_min: float = 4.0
    z_max: float = 15.0
    layers: tuple = (6.0, 9.0, 12.0)
    block: float = 20.0
    street_min: float = 8.0
    street_max: float = 12.0
    building_h: tuple = (12.0, 35.0)
    rubble_h: tuple = (1.0, 5.0)
    p_collapse: float = 0.4
    n_survivors: int = 16
    p_alive: float = 0.7


@dataclass
class DroneCfg:
    v_h: float = 5.0
    v_z: float = 2.0
    a_max: float = 4.0
    yaw_rate: float = math.radians(90.0)
    v_ref: float = 3.0

    @property
    def v_max_3d(self) -> float:
        return math.hypot(self.v_h, self.v_z)


@dataclass
class SensorCfg:
    fov_h: float = 90.0           # degrees
    fov_v: float = 60.0
    tilt: float = 30.0            # degrees below horizontal (camera axis)
    r_depth: float = 25.0
    r_detect: float = 15.0
    r_see: float = 20.0
    theta_min: float = 20.0       # minimum elevation (deg) for a cell to count as viewed
    rays_h: int = 32
    rays_v: int = 12
    range_sigma: float = 0.05


@dataclass
class CommsCfg:
    R: float = 35.0
    p_drop: float = 0.1
    max_events: int = 64
    pose_every: int = 10          # relayed POSE period in ticks (1 Hz)


@dataclass
class GateCfg:
    tau: float = 0.3
    h: float = 1.0
    eps: float = 0.1
    patience: float = 1.0
    d_safe: float = 1.0
    d_obs: float = 1.5
    r_sep: float = 3.0


@dataclass
class MissionCfg:
    partition: str = "voronoi"
    audit_q: float = 0.1
    t_conf: float = 8.0
    view_attempts: int = 3
    trust_gate: bool = False     # naive/unfiltered also run the motion gate, believing every report (fair collisions)


@dataclass
class AttackCfg:
    n_compromised: int = 0
    motion: str = "hover"
    position_lie: str = "range_aware"
    fake_coverage: bool = True
    start_time: float = 12.0
    estimate_offset: float = 0.0
    claim_all: bool = False      # claim the whole claim region as swept the moment lying starts
    seek_quiet: bool = True      # look for a spot with <= 2 listeners before lying


@dataclass
class Config:
    N: int = 10
    F: int = 1
    fleet: str = "gated"
    allow_beyond_assumption: bool = False
    sim: SimCfg = field(default_factory=SimCfg)
    world: WorldCfg = field(default_factory=WorldCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    comms: CommsCfg = field(default_factory=CommsCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    attack: AttackCfg = field(default_factory=AttackCfg)

    # ------------------------------------------------------------------ derived quantities
    @property
    def eps_prime(self) -> float:
        return max(self.gate.eps, 4.0 * self.sensor.range_sigma)

    @property
    def silent_ticks(self) -> int:
        """L = ceil(ln 1e-9 / ln p_drop): the gate's single probabilistic assumption."""
        p = self.comms.p_drop
        return 1 if p <= 0 else int(math.ceil(math.log(1e-9) / math.log(p)))

    @property
    def silence_radius(self) -> float:
        """A drone unheard for L ticks is outside this radius (v_close = 2 v_max_3d)."""
        return self.comms.R - 2 * self.drone.v_max_3d * self.silent_ticks * self.sim.dt

    @property
    def honest_clearance(self) -> float:
        """Clearance the gate needs from an honest neighbour: d_safe + braking ball over one tick + tau."""
        t = self.sim.dt + self.gate.tau
        return self.gate.d_safe + 0.5 * self.drone.a_max * t * t + self.eps_prime

    # ------------------------------------------------------------------ validation
    def validate(self) -> "Config":
        errs = []
        if not 1 <= self.N <= 20:
            errs.append(f"N must be in [1, 20], got {self.N}")
        if not 0 <= self.F <= 2:
            errs.append(f"F must be in [0, 2], got {self.F}")
        if self.attack.n_compromised > self.F and not self.allow_beyond_assumption:
            errs.append(f"n_compromised={self.attack.n_compromised} > F={self.F}; set allow_beyond_assumption")
        if self.attack.n_compromised >= self.N:
            errs.append("at least one drone must be honest")
        if self.fleet not in FLEETS:
            errs.append(f"fleet must be one of {FLEETS}")
        if self.attack.motion not in LIAR_MOTIONS:
            errs.append(f"attack.motion must be one of {LIAR_MOTIONS}")
        if self.attack.position_lie not in LIAR_POSITION_LIES:
            errs.append(f"attack.position_lie must be one of {LIAR_POSITION_LIES}")
        if self.sim.record_level not in RECORD_LEVELS:
            errs.append(f"sim.record_level must be one of {RECORD_LEVELS}")
        if self.mission.partition not in PARTITIONS:
            errs.append(f"mission.partition must be one of {PARTITIONS}")
        if not 0 < self.sensor.fov_h <= 180 or not 0 < self.sensor.fov_v <= 180:
            errs.append("FOV must be in (0, 180] degrees")
        w = self.world
        if not 0 <= w.z_min < w.z_max:
            errs.append("need 0 <= z_min < z_max")
        if any(not w.z_min <= z <= w.z_max for z in w.layers):
            errs.append("every cruise layer must lie inside [z_min, z_max]")
        if w.zone > w.arena:
            errs.append("search zone must fit inside the arena")
        if self.comms.R <= 0 or self.gate.d_safe <= 0 or self.sim.dt <= 0:
            errs.append("R, d_safe and dt must be positive")
        if not 0 <= self.comms.p_drop < 1:
            errs.append("p_drop must be in [0, 1)")
        if not 0 <= self.mission.audit_q <= 1:
            errs.append("audit_q must be in [0, 1]")
        # geometry checks against the gate clearance (T0.3, review finding 8)
        c = self.honest_clearance
        if self.gate.r_sep <= c:
            errs.append(f"r_sep={self.gate.r_sep} must exceed the honest clearance {c:.2f} m")
        lanes = w.street_min - 2 * self.gate.d_obs
        if lanes < c:
            errs.append(f"street_min leaves {lanes:.2f} m between margins, below the clearance {c:.2f} m")
        if errs:
            raise ValueError("invalid config:\n  " + "\n  ".join(errs))
        return self

    def warnings(self) -> list:
        out = []
        if self.silence_radius <= self.honest_clearance:
            out.append(f"silence radius {self.silence_radius:.1f} m <= clearance; unheard drones will hold the gate")
        return out

    # ------------------------------------------------------------------ IO and overrides
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d)
        sub = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d.pop(f.name)
            default = getattr(cls(), f.name)
            if dataclasses.is_dataclass(default):
                v = _build(type(default), v)
            kwargs[f.name] = v
        if d:
            raise KeyError(f"unknown config keys: {sorted(d)}")
        del sub
        return cls(**kwargs)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def override(self, items: Iterable[str]) -> "Config":
        """Apply dotted overrides such as ``["comms.R=30", "fleet=naive"]`` and return a new config."""
        d = self.to_dict()
        for item in items:
            key, _, raw = item.partition("=")
            try:
                val = json.loads(raw)
            except json.JSONDecodeError:
                val = raw
            node = d
            parts = key.split(".")
            for p in parts[:-1]:
                if p not in node:
                    raise KeyError(f"unknown config section '{p}'")
                node = node[p]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key '{key}'")
            node[parts[-1]] = val
        return Config.from_dict(d)


def _build(kind, values: Dict[str, Any]):
    names = {f.name for f in dataclasses.fields(kind)}
    unknown = set(values) - names
    if unknown:
        raise KeyError(f"unknown keys for {kind.__name__}: {sorted(unknown)}")
    fixed = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return kind(**fixed)
