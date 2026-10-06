"""Message freshness and bounded per-receiver message memory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MessageMemory:
    """Cache, for each receiver, the latest message it received from each sender.

    ``pos``/``vel`` are ``(N, N, d)`` and indexed ``[receiver, sender]``, so a receiver
    whose link dropped keeps the value it last heard, not what another receiver got.
    ``ranges`` is ``(N, N)``: the distance measured when that message arrived, which goes
    stale with it.  ``payload`` caches each extra array a message carried in the same way, and reads as False
    or infinity for a sender nobody has yet heard a payload from.

    ``ttl`` is measured in simulation steps.  A TTL of zero disables reuse and
    therefore exactly matches the simulator's original packet semantics.
    """

    n: int
    dim: int
    ttl: int = 0

    def __post_init__(self) -> None:
        if self.ttl < 0:
            raise ValueError("message_ttl must be non-negative")
        self.pos = np.zeros((self.n, self.n, self.dim))
        self.vel = np.zeros((self.n, self.n, self.dim))
        self.ranges = np.full((self.n, self.n), np.inf)
        self.age = np.full((self.n, self.n), np.inf)
        self.available = np.zeros((self.n, self.n), dtype=bool)
        self.payload = {}

    def reset(self) -> None:
        self.pos.fill(0.0)
        self.vel.fill(0.0)
        self.ranges.fill(np.inf)
        self.age.fill(np.inf)
        self.available.fill(False)
        self.payload = {}

    def update(self, pos: np.ndarray, vel: np.ndarray, received: np.ndarray, ranges=None, payload=None) -> None:
        self.age[np.isfinite(self.age)] += 1
        receivers, senders = np.nonzero(received)
        self.pos[receivers, senders] = pos[senders]
        self.vel[receivers, senders] = vel[senders]
        if ranges is not None:
            self.ranges[receivers, senders] = ranges[receivers, senders]
        if self.ttl > 0:  # with no cache, a receiver reads this step's broadcast directly
            for key, value in (payload or {}).items():
                if key not in self.payload:
                    # A message can be held before any payload has arrived, as on a run's first step.  So what nobody has sent
                    # must read as nothing: False, or an infinite number, which cannot be mistaken for a measurement of zero.
                    empty = np.inf if np.issubdtype(value.dtype, np.floating) else 0
                    self.payload[key] = np.full((self.n, self.n, *value.shape[1:]), empty, dtype=value.dtype)
                self.payload[key][receivers, senders] = value[senders]
        self.age[receivers, senders] = 0
        self.available = np.isfinite(self.age) & (self.age <= self.ttl)

    def view(self, pos: np.ndarray, vel: np.ndarray, received: np.ndarray):
        """Return message arrays and availability for one observation.

        With ``ttl == 0`` the messages are this step's ``(N, d)`` broadcasts.  Otherwise
        they are the ``(N, N, d)`` per-receiver cache, indexed ``[receiver, sender]``.
        """
        if self.ttl == 0:
            return pos.copy(), vel.copy(), received.copy(), np.where(received, 0.0, np.inf)
        return (
            self.pos.copy(),
            self.vel.copy(),
            self.available.copy(),
            self.age.copy(),
        )

    def view_ranges(self, ranges: np.ndarray, received: np.ndarray) -> np.ndarray:
        """(N, N) the range each receiver holds for each sender; infinity where it holds nothing."""
        if self.ttl == 0:
            return np.where(received, ranges, np.inf)
        return np.where(self.available, self.ranges, np.inf)
