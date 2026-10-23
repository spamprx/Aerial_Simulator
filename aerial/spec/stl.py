"""Discrete-time Signal Temporal Logic: quantitative robustness over a recorded run.

A formula is a small tree over margin signals, and ``robustness`` gives rho(phi, k) for every sample
k of a run with K + 1 samples spaced dt apart.  Positive means phi holds at k with that much to spare,
negative that it fails by that much, and zero that the margin is exactly used up.

    rho(Pred(fn), k)              = fn(res)[k]
    rho(Not(phi), k)              = -rho(phi, k)
    rho(And(phi, ...), k)         = min of the children at k    (+inf with no children)
    rho(Or(phi, ...), k)          = max of the children at k    (-inf with no children)
    rho(Always(a, b, phi), k)     = min over k' in W(k) of rho(phi, k')
    rho(Eventually(a, b, phi), k) = max over k' in W(k) of rho(phi, k')
    rho(Until(a, b, phi, psi), k) = max over k' in W(k) of
                                        min(rho(psi, k'), min over k <= k'' < k' of rho(phi, k''))

W(k) holds the samples whose time lies in [t_k + a, t_k + b], with a and b in seconds.  Three
conventions decide the edge cases, and the report states them too (docs/report/specification.md):

Truncation.  W(k) stops at the end of the trace.  If no sample is left the window is empty, and the
min and max of nothing are +inf and -inf: Always holds vacuously, Eventually and Until fail.  So near
the end, rho(phi, k) describes only what was recorded.  rho(phi, 0) is the one number for a run.

Sampling.  A bound that falls between two samples is rounded inwards, so W(k) never holds a sample
outside [a, b].  In particular Eventually(0, T1, phi) never counts anything after the deadline T1.  An
interval with no sample in it at all is an error, not an empty window.  ``b = inf`` reaches the end.

Until.  phi must hold from k up to but not including k', as in LTL, so Until(0, b, phi, psi) needs
no phi at all if psi already holds at k.

A NaN margin propagates to every window that contains it.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

# 0.3 / 0.1 lands just under 3 and 0.07 / 0.01 just over 7, and neither may move a bound by a step.
_STEP_TOLERANCE = 1e-9


def _window_min(values: np.ndarray, lo: int, hi: float) -> np.ndarray:
    """(n,) minimum of values[k + lo .. k + hi] for each k, truncated at the end, +inf if empty.

    The van Herk / Gil-Werman algorithm, which costs O(n) whatever the window's width.  Cut the
    signal into blocks of that width and take running minima forwards and backwards inside each
    one.  A window then spans at most two blocks: the end of one and the start of the next.
    """
    n = len(values)
    result = np.full(n, np.inf)
    hi = min(hi, n - 1)
    if hi < lo:
        return result
    width = int(hi - lo + 1)
    count = n - lo
    # Padding with +inf is what truncates: the minimum of anything and +inf is that thing.
    padded = np.full((math.ceil((count + width - 1) / width), width), np.inf)
    padded.flat[:count] = values[lo:]
    forwards = np.minimum.accumulate(padded, axis=1).ravel()
    backwards = np.minimum.accumulate(padded[:, ::-1], axis=1)[:, ::-1].ravel()
    result[:count] = np.minimum(backwards[:count], forwards[width - 1:width - 1 + count])
    return result


def _window_max(values: np.ndarray, lo: int, hi: float) -> np.ndarray:
    return -_window_min(-values, lo, hi)


def _unbounded_until(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """(n,) rho(phi U psi) with no time bound, by one backward scan.  This is O(n), but not vectorised."""
    result = psi.copy()
    for k in range(len(psi) - 2, -1, -1):
        result[k] = np.maximum(psi[k], np.minimum(phi[k], result[k + 1]))
    return result


class _Run:
    """One run being evaluated: what is computed from it is kept, however many prefixes are asked for."""

    def __init__(self, res):
        self.res, self.dt, self.length = res, (res.dt if hasattr(res, "dt") else res.cfg.dt), len(res.t)
        self._kept = {}

    def once(self, node: "Formula", compute: Callable):
        if id(node) not in self._kept:
            self._kept[id(node)] = compute(self.res)
        return self._kept[id(node)]


class Formula:
    def _signal(self, run: _Run, n: int) -> np.ndarray:
        """(n,) rho at each sample of the trace cut down to its first ``n`` samples."""
        raise NotImplementedError


class Pred(Formula):
    """An atomic predicate.  ``fn(res)`` returns the (K+1,) margin: non-negative where it holds."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def _margins(self, res) -> np.ndarray:
        raw = np.asarray(self.fn(res))
        if raw.dtype == bool:
            raise TypeError("a predicate returns a margin, not a boolean: False would read as a margin of 0")
        margins = np.array(raw, dtype=float)
        if margins.shape != (len(res.t),):
            raise ValueError(f"predicate returned shape {margins.shape}, expected {(len(res.t),)}")
        return margins

    def _signal(self, run, n):
        return run.once(self, self._margins)[:n]


class Deferred(Formula):
    """A formula built from the run when it is evaluated, such as a conjunction over its honest agents.

    The formula itself can then be reused across runs whose compromised sets differ.
    """

    def __init__(self, build: Callable):
        self.build = build

    def _signal(self, run, n):
        return run.once(self, self.build)._signal(run, n)


class Not(Formula):
    def __init__(self, child: Formula):
        self.child = child

    def _signal(self, run, n):
        return -self.child._signal(run, n)


class And(Formula):
    def __init__(self, *children: Formula):
        self.children = children

    def _signal(self, run, n):
        result = np.full(n, np.inf)
        for child in self.children:
            result = np.minimum(result, child._signal(run, n))
        return result


class Or(Formula):
    def __init__(self, *children: Formula):
        self.children = children

    def _signal(self, run, n):
        result = np.full(n, -np.inf)
        for child in self.children:
            result = np.maximum(result, child._signal(run, n))
        return result


class _Bounded(Formula):
    """An operator with a time interval [a, b] in seconds."""

    def __init__(self, a: float, b: float):
        if not (0.0 <= a <= b and a < np.inf):
            raise ValueError(f"need 0 <= a <= b with a finite, got [{a}, {b}]")
        self.a, self.b = a, b

    def _offsets(self, dt: float):
        """The sample offsets (lo, hi) that lie inside [a, b]; hi is inf for an open-ended interval."""
        lo = math.ceil(self.a / dt - _STEP_TOLERANCE)
        hi = math.inf if self.b == np.inf else math.floor(self.b / dt + _STEP_TOLERANCE)
        if hi < lo:
            raise ValueError(f"no sample lies inside [{self.a}, {self.b}] at dt = {dt}")
        return lo, hi


class Always(_Bounded):
    def __init__(self, a: float, b: float, child: Formula):
        super().__init__(a, b)
        self.child = child

    def _signal(self, run, n):
        return _window_min(self.child._signal(run, n), *self._offsets(run.dt))


class Eventually(_Bounded):
    def __init__(self, a: float, b: float, child: Formula):
        super().__init__(a, b)
        self.child = child

    def _signal(self, run, n):
        return _window_max(self.child._signal(run, n), *self._offsets(run.dt))


class Until(_Bounded):
    """Computed as G[0, a) phi and F[a, b] psi and F[a, a](phi U psi), which is the definition above.

    The first witness of psi at or after k + a lies inside the window whenever any witness does, so
    the bound on the last term comes from the middle one.  Min and max over an ordered set preserve
    a boolean identity like this one, which makes it hold for robustness too.
    """

    def __init__(self, a: float, b: float, phi: Formula, psi: Formula):
        super().__init__(a, b)
        self.phi, self.psi = phi, psi

    def _signal(self, run, n):
        lo, hi = self._offsets(run.dt)
        phi, psi = self.phi._signal(run, n), self.psi._signal(run, n)
        from_a = np.full(n, -np.inf)
        if lo < n:
            from_a[:n - lo] = _unbounded_until(phi, psi)[lo:]
        return np.minimum.reduce([_window_min(phi, 0, lo - 1), _window_max(psi, lo, hi), from_a])


def robustness(phi: Formula, res) -> np.ndarray:
    """(K+1,) rho(phi, k) at every sample.  ``res`` needs ``t`` and ``cfg.dt``, plus what the predicates read."""
    run = _Run(res)
    return phi._signal(run, run.length)


def prefix_robustness(phi: Formula, res) -> np.ndarray:
    """(K+1,) entry t is rho(phi, 0) on the trace cut off at sample t, as if the run had stopped there.

    This is the rho(t) curve the dashboard shows.  For Always(0, inf, Pred) it is the running minimum.
    Each entry evaluates phi on its own prefix, so this costs O(K) evaluations of phi, unlike
    ``robustness``.  A nested formula's curve need not be monotone: extending the trace can give an
    inner Eventually the witness it lacked.
    """
    run = _Run(res)
    return np.array([phi._signal(run, n)[0] for n in range(1, run.length + 1)])
