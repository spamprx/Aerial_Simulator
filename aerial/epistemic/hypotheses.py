"""Fault hypotheses H_i (T3.3): exact S5 knowledge of *who* is faulty.

Every faulty set S with |S| <= F and i not in S is a candidate world (10 for N=10, F=1; 191 for N=20, F=2).
Evidence adds constraints that eliminate worlds:

- ``Unit(j)``: j is in S (a proof that j lied);
- ``Clause(js)``: at least one member of ``js`` is in S (an accusation, or two claims that contradict).

K_i(j faulty) holds iff j is in every surviving S. If no S survives, the F assumption is broken: we flag it and,
instead of computing ``all()`` over an empty set (which is vacuously true), fall back to own observations only.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional

import numpy as np


def bit(j: int) -> int:
    return 1 << j


def mask_of(ids: Iterable[int]) -> int:
    m = 0
    for j in ids:
        m |= 1 << int(j)
    return m


class Hypotheses:
    def __init__(self, N: int, F: int, me: int, truth: Optional[Iterable[int]] = None):
        """``truth`` makes an oracle: only the true faulty set survives."""
        self.N, self.F, self.me = N, F, me
        others = [j for j in range(N) if j != me]
        sets = [0]
        for k in range(1, F + 1):
            sets += [mask_of(c) for c in combinations(others, k)]
        self.S = np.array(sets, dtype=np.uint64)
        self.alive = np.ones(len(self.S), bool)
        if truth is not None:
            self.alive &= self.S == np.uint64(mask_of(truth))
        self.constraints: List[tuple] = []
        self._cache = None

    # ------------------------------------------------------------------ evidence
    def add_unit(self, j: int, why: str = "") -> bool:
        """Return True if this changed the world set."""
        return self._apply(("unit", int(j), why), (self.S & np.uint64(bit(j))) != 0)

    def add_clause(self, js: Iterable[int], why: str = "") -> bool:
        js = sorted({int(j) for j in js})
        if self.me in js:
            js = [j for j in js if j != self.me]          # I know I'm honest
            if not js:
                return False
        m = np.uint64(mask_of(js))
        return self._apply(("clause", tuple(js), why), (self.S & m) != 0)

    def _apply(self, con, ok: np.ndarray) -> bool:
        before = self.alive.copy()
        self.alive &= ok
        self.constraints.append(con)
        changed = not np.array_equal(before, self.alive)
        if changed:
            self._cache = None
        return changed

    # ------------------------------------------------------------------ queries
    @property
    def assumption_broken(self) -> bool:
        return not self.alive.any()

    @property
    def worlds(self) -> np.ndarray:
        return self.S[self.alive]

    @property
    def n_hypotheses(self) -> int:
        return int(self.alive.sum())

    def _flags(self):
        if self._cache is None:
            W = self.worlds
            if len(W) == 0:
                kf = np.zeros(self.N, bool)
                kh = np.zeros(self.N, bool)
                pf = np.ones(self.N, bool)
            else:
                member = ((W[:, None] >> np.arange(self.N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
                kf = member.all(axis=0)
                kh = ~member.any(axis=0)
                pf = member.any(axis=0)
            pf[self.me] = False
            kh[self.me] = True
            self._cache = (kf, kh, pf)
        return self._cache

    def known_faulty(self) -> np.ndarray:
        return self._flags()[0]

    def known_honest(self) -> np.ndarray:
        return self._flags()[1]

    def possibly_faulty(self) -> np.ndarray:
        return self._flags()[2]

    def budget_left(self) -> int:
        return self.F - int(self.known_faulty().sum())

    def outside_all(self, reporters: np.ndarray) -> np.ndarray:
        """For each reporter bitmask, True iff in every surviving world some reporter is outside S.

        This is K_i(claim) for claims backed by ``reporters``. Empty world set -> False (fallback to own data).
        """
        reporters = np.asarray(reporters, dtype=np.uint64)
        W = self.worlds
        if len(W) == 0:
            return np.zeros(reporters.shape, bool)
        kh = mask_of(np.flatnonzero(self.known_honest() & (np.arange(self.N) != self.me)))
        out = (reporters & np.uint64(kh)) != 0                # fast path: a known-honest reporter
        rest = ~out & (reporters != 0)
        if rest.any():
            r = reporters[rest]
            out[rest] = ((r[:, None] & ~W[None, :]) != 0).all(axis=1)
        return out
