"""Disk graph with Bernoulli drops, re-query, and event relaying (T1.8, T1.9).

Convention: ``A[i, j] == True`` means i receives j. Every environment draw is a full (N, N) block every tick, so
naive and gated runs of one seed see the same drops (pairing).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .graph import algebraic_connectivity, disk_adjacency
from .messages import Event


class Network:
    def __init__(self, N: int, R: float, p_drop: float, rng_comm, rng_requery):
        self.N, self.R, self.p = N, R, p_drop
        self.rng_comm, self.rng_requery = rng_comm, rng_requery

    def adjacency(self, pos: np.ndarray, requery: Optional[np.ndarray] = None) -> np.ndarray:
        """Who receives whom this tick. ``requery[i]`` grants i a second, independent attempt from each in-range
        neighbour (D4)."""
        in_range = disk_adjacency(pos, self.R)
        keep = self.rng_comm.random((self.N, self.N)) >= self.p
        retry = self.rng_requery.random((self.N, self.N)) >= self.p      # drawn every tick (pairing)
        A = in_range & keep
        if requery is not None and requery.any():
            A = A | (in_range & retry & requery[:, None])
        np.fill_diagonal(A, False)
        return A

    @staticmethod
    def lambda2(A: np.ndarray) -> float:
        sym = A | A.T
        return float(algebraic_connectivity(sym))


class Gossip:
    """Flooding with dedup on (origin, seq), one hop per tick, priority bandwidth cap.

    Relaying is done on the agents' behalf. ``relay_filter[j]`` (set by a liar) may drop what j forwards; nothing can
    change an event's origin (A8).
    """

    def __init__(self, N: int, max_events: int):
        self.N, self.cap = N, max_events
        self.known: List[Set] = [set() for _ in range(N)]
        self.pending: List[List[Event]] = [[] for _ in range(N)]   # to forward next tick
        self.relay_filter: Dict[int, object] = {}
        self.sent = np.zeros(N, dtype=int)

    def originate(self, i: int, ev: Event) -> None:
        self.known[i].add(ev.key)
        self.pending[i].append(ev)

    def exchange(self, A: np.ndarray) -> List[List[Event]]:
        """Deliver every agent's pending bundle to its receivers; return newly received events per agent."""
        bundles = []
        for j in range(self.N):
            evs = sorted(self.pending[j], key=lambda e: (int(e.kind), e.t, e.origin, e.seq))
            send, keep = evs[: self.cap], evs[self.cap:]
            f = self.relay_filter.get(j)
            if f is not None:
                send = [e for e in send if e.origin == j or f(e)]
            bundles.append(send)
            self.pending[j] = keep
            self.sent[j] += len(send)
        new: List[List[Event]] = [[] for _ in range(self.N)]
        for i in range(self.N):
            for j in np.flatnonzero(A[i]):
                for e in bundles[j]:
                    if e.key not in self.known[i]:
                        self.known[i].add(e.key)
                        new[i].append(e)
                        self.pending[i].append(e)
        return new
