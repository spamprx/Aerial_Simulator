"""Relayed events (T0.5 / T1.9). Origin cannot be forged (A8): only the simulator stamps ``origin``."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EventKind(IntEnum):
    # lower value = higher relay priority
    ACCUSE = 0
    KNOWN_FAULTY = 1
    SURVIVOR = 2
    TRACK_EST = 3
    LOOKS = 4
    CLEARED = 5
    AUDITED = 6
    POSE = 7
    TERRAIN = 8
    RANGES = 9


@dataclass(frozen=True)
class Event:
    origin: int
    seq: int
    t: float
    kind: EventKind
    payload: Any

    @property
    def key(self):
        return (self.origin, self.seq)


@dataclass(frozen=True)
class Draft:
    """What an agent puts in its outbox. The simulator turns it into an ``Event`` with the true origin."""
    kind: EventKind
    payload: Any
