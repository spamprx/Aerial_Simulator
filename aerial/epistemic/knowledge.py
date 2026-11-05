"""Trust standards over provenance bitmasks (T3.9).

A claim (cell cleared, cell free, survivor at cell) is backed by a uint64 bitmask of reporters plus the agent's own
observation flag. A standard decides whether the agent may *act* on it:

- ``Belief`` (naive, unfiltered, and gated work allocation): anyone reported it, ignoring known-faulty reporters;
- ``Knowledge`` (gated, oracle): own observation, or every world in H_i leaves an honest reporter (S5).
"""
from __future__ import annotations

import numpy as np

from .hypotheses import Hypotheses, mask_of


class Standard:
    def __init__(self, hyp: Hypotheses):
        self.hyp = hyp

    def faulty_mask(self) -> np.uint64:
        return np.uint64(mask_of(np.flatnonzero(self.hyp.known_faulty())))

    def holds(self, own: np.ndarray, reporters: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Belief(Standard):
    def holds(self, own, reporters):
        r = np.asarray(reporters, dtype=np.uint64) & ~self.faulty_mask()
        return np.asarray(own, bool) | (r != 0)


class TrustAll(Standard):
    """Unfiltered: even proven liars are believed (it never runs detection anyway)."""

    def holds(self, own, reporters):
        return np.asarray(own, bool) | (np.asarray(reporters, dtype=np.uint64) != 0)


class Knowledge(Standard):
    def holds(self, own, reporters):
        own = np.asarray(own, bool)
        out = own.copy()
        idx = np.flatnonzero(~own & (np.asarray(reporters, dtype=np.uint64) != 0))
        if len(idx):
            out[idx] = self.hyp.outside_all(np.asarray(reporters, dtype=np.uint64)[idx])
        return out
