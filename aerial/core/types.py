"""Shared value types (T0.5). Agents see only these, never ``aerial.world``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DroneState:
    pos: np.ndarray      # (3,)
    vel: np.ndarray      # (3,)
    yaw: float


@dataclass(frozen=True)
class ClaimedState:
    """What a drone puts in its beacon. Honest drones claim the truth; a liar may not."""
    pos: np.ndarray
    vel: np.ndarray
    yaw: float = 0.0


@dataclass(frozen=True)
class ReceivedBeacon:
    sender: int
    claim: ClaimedState
    seq: int             # sender's beacon sequence number (= transmit tick)
    age: int             # ticks since transmit; 0 = fresh
    uwb_range: Optional[float]   # measured at transmit tick; None when cached (age > 0)


@dataclass(frozen=True)
class StaticInfo:
    """Everything a drone knows before take-off (A4): no terrain."""
    i: int
    N: int
    F: int
    arena: float
    zone: float
    z_min: float
    z_max: float
    base: np.ndarray     # (3,) launch-base centre
    dt: float
    zone_origin: np.ndarray = None   # (2,) lower corner of the search zone (public: the zone is the mission)
    coverage_cell: float = 2.0
    terrain_cell: float = 1.0
    layer: float = 9.0               # this drone's cruise layer (3D traffic rule)
    entry: np.ndarray = None         # (3,) this drone's formation slot on the entry line
    slot: np.ndarray = None          # (3,) this drone's landing slot at the base


@dataclass
class SensorFrame:
    """One camera frame. Filled by ``aerial.sensing`` (Phase 1)."""
    t: float
    terrain_over: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (M, 3): cell x, y, ray height (H < z)
    terrain_hit: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (M, 3): cell x, y, hit height (H >= z)
    viewed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # coverage cell ids
    looks: List[Any] = field(default_factory=list)                              # SurvivorLook (exact cell)
    sightings: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))     # anonymous drone positions
    visible: Any = None                                                         # VisibleVolume


@dataclass
class Action:
    u: np.ndarray                    # (3,) commanded acceleration
    yaw_rate: float = 0.0
    outbox: List[Any] = field(default_factory=list)   # Events originated this tick
    requery: bool = False


@dataclass(frozen=True)
class AgentObservation:
    t: float
    k: int
    i: int
    own: DroneState
    frame: Optional[SensorFrame]
    beacons: Dict[int, ReceivedBeacon]
    events: Tuple[Any, ...]
    static: StaticInfo
