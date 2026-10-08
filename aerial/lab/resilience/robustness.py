"""(r, s)-robustness of a graph (LeBlanc et al., "Resilient asymptotic consensus in robust networks",
IEEE JSAC 2013).

A set S is r-reachable if some agent in S has at least r in-neighbours outside S, and X_S^r is the set of
agents in S that do.  A graph is (r, s)-robust if, for every pair of disjoint non-empty sets S1 and S2, at
least one of these holds:

    |X_S1^r| = |S1|,     |X_S2^r| = |S2|,     |X_S1^r| + |X_S2^r| >= s

r-robust means (r, 1)-robust.  These graph conditions are sufficient for W-MSR, not necessary:
(2F + 1)-robust under an F-local attack, and (F + 1, F + 1)-robust under an F-total one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

BRUTE_FORCE_LIMIT = 12
_BITS_IN_16 = np.array([bin(x).count("1") for x in range(1 << 16)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    return _BITS_IN_16[x & 0xFFFF] + _BITS_IN_16[x >> 16]


def _ids(mask: int) -> List[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def is_rs_robust_bruteforce(A: np.ndarray, r: int, s: int) -> bool:
    """The definition, written out: it tries every pair of disjoint subsets, about 3^N of them.

    This is the reference that ``is_rs_robust`` is tested against, so it shares no code with it.
    """
    n = len(A)
    if n > BRUTE_FORCE_LIMIT:
        raise ValueError(f"brute force is for N <= {BRUTE_FORCE_LIMIT}, use is_rs_robust")
    neighbours = [{j for j in range(n) if A[i][j]} for i in range(n)]
    reachable = {}

    def count(mask: int) -> int:
        if mask not in reachable:
            members = set(_ids(mask))
            reachable[mask] = sum(1 for i in members if len(neighbours[i] - members) >= r)
        return reachable[mask]

    everyone = (1 << n) - 1
    for first in range(1, everyone + 1):
        rest = everyone ^ first
        second = rest
        while second:
            x_first, x_second = count(first), count(second)
            satisfied = x_first == bin(first).count("1") or x_second == bin(second).count("1") or x_first + x_second >= s
            if not satisfied:
                return False
            second = (second - 1) & rest
    return True


class _Graph:
    """Bitmask tables over all 2^N subsets, so the check takes O(N * 2^N) rather than 3^N."""

    def __init__(self, A: np.ndarray):
        self.n = len(A)
        self.subsets = np.arange(1 << self.n, dtype=np.uint32)
        self.sizes = _popcount(self.subsets)
        self.complement = np.uint32((1 << self.n) - 1) ^ self.subsets
        self.in_masks = [np.uint32(sum(1 << j for j in range(self.n) if A[i][j])) for i in range(self.n)]
        self._reachable = {}

    def reachable(self, r: int) -> np.ndarray:
        """For every subset S, |X_S^r|."""
        if r not in self._reachable:
            counts = np.zeros(len(self.subsets), dtype=np.uint8)
            for i, in_mask in enumerate(self.in_masks):
                member = ((self.subsets >> i) & 1).astype(bool)
                counts += member & (_popcount(in_mask & ~self.subsets) >= r)
            self._reachable[r] = counts
        return self._reachable[r]

    def _subset_or(self, flags: np.ndarray) -> np.ndarray:
        """For every T, whether some flagged S is a subset of T: a zeta transform in O(N * 2^N)."""
        result = flags.copy()
        for bit in range(self.n):
            view = result.reshape(-1, 2, 1 << bit)  # axis 1 is whether this bit is set
            view[:, 1, :] |= view[:, 0, :]
        return result

    def witness(self, r: int, s: int) -> Optional[Tuple[List[int], List[int]]]:
        """A pair of disjoint sets violating (r, s)-robustness, or None if the graph is robust."""
        x = self.reachable(r)
        # Only non-empty sets that are not "full" can violate it, and only if x1 + x2 <= s - 1.
        open_sets = (self.sizes > 0) & (x < self.sizes)
        for x_first in range(min(s, self.n + 1)):
            firsts = open_sets & (x == x_first)
            if not firsts.any():
                continue
            seconds = open_sets & (x <= s - 1 - x_first)
            hits = self._subset_or(seconds)[self.complement[firsts]]
            if hits.any():
                first = int(self.subsets[firsts][np.argmax(hits)])
                candidates = self.subsets[seconds & ((self.subsets & np.uint32(first)) == 0)]
                return _ids(first), _ids(int(candidates[0]))
        return None


def robustness_witness(A: np.ndarray, r: int, s: int = 1) -> Optional[Tuple[List[int], List[int]]]:
    """The agent ids of two disjoint sets that break (r, s)-robustness, or None if the graph has it."""
    if r < 0 or s < 1:
        raise ValueError("(r, s)-robustness needs r >= 0 and s >= 1")
    return _Graph(A).witness(r, s)


def is_rs_robust(A: np.ndarray, r: int, s: int = 1) -> bool:
    """Whether the graph is (r, s)-robust.  About a second at N = 20, the project's limit."""
    return robustness_witness(A, r, s) is None


def _largest(low: int, high: int, holds) -> int:
    """The largest value in (low, high] for which a monotone condition ``holds``; ``low`` if none."""
    while low < high:
        middle = (low + high + 1) // 2
        if holds(middle):
            low = middle
        else:
            high = middle - 1
    return low


def max_r(A: np.ndarray) -> int:
    """The largest r for which the graph is r-robust.  No graph on N nodes exceeds ceil(N / 2)."""
    graph = _Graph(A)
    return _largest(0, (graph.n + 1) // 2, lambda r: graph.witness(r, 1) is None)


def max_s(A: np.ndarray, r: int) -> int:
    """The largest s for which the graph is (r, s)-robust, up to N; 0 if it is not even r-robust."""
    graph = _Graph(A)
    return _largest(0, graph.n, lambda s: graph.witness(r, s) is None)


def resilience_conditions(A: np.ndarray, F: int) -> Dict[str, bool]:
    """The sufficient conditions for W-MSR with bound F, under F-local and F-total attacks."""
    graph = _Graph(A)
    return {"(2F+1)-robust": graph.witness(2 * F + 1, 1) is None,
            "(F+1,F+1)-robust": graph.witness(F + 1, F + 1) is None}
