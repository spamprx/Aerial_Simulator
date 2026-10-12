"""How agent i combines its neighbours' values, kept apart from every control law.

A coordination primitive asks its aggregator for "the value of the neighbourhood" and never
needs to know whether that was a plain mean or a W-MSR filter:

    result = aggregator.aggregate(own_value, received_values, sender_ids)
    u_i = k * (result.value - own_value)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..registry import aggregator_registry


@dataclass
class AggResult:
    value: np.ndarray  # (d,) aggregated value (own value if nothing trusted)
    trusted: np.ndarray  # agent ids used in every coordinate
    rejected: np.ndarray  # agent ids removed in at least one coordinate
    count: int  # |trusted|


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, own: np.ndarray, values: np.ndarray, ids: np.ndarray) -> AggResult:
        """Combine ``values`` (M, d), sent by agents ``ids`` (M,), from the view of ``own`` (d,)."""


@aggregator_registry.register("mean")
class MeanAggregator(Aggregator):
    """Naive: the mean of every received value, with no filtering."""

    def aggregate(self, own, values, ids):
        own, values, ids = np.asarray(own, dtype=float), np.asarray(values, dtype=float), np.asarray(ids, dtype=int)
        value = values.mean(axis=0) if len(ids) else own.copy()
        return AggResult(value, ids.copy(), np.empty(0, dtype=int), len(ids))


def _standard_keep(own: float, column: np.ndarray, f: int) -> np.ndarray:
    """W-MSR on one coordinate: of the values strictly above ``own`` drop the f largest (all of
    them if there are at most f), likewise below.  Values equal to ``own`` are always kept."""
    keep = np.ones(len(column), dtype=bool)
    larger = np.flatnonzero(column > own)
    larger = larger[np.argsort(column[larger], kind="stable")]
    keep[larger[len(larger) - min(f, len(larger)):]] = False
    smaller = np.flatnonzero(column < own)
    smaller = smaller[np.argsort(column[smaller], kind="stable")]
    keep[smaller[:min(f, len(smaller))]] = False
    return keep


def _trimmed_keep(own: float, column: np.ndarray, f: int) -> np.ndarray:
    """Drop the f largest and f smallest whatever ``own`` is: what ``WMSRConsensus`` does today."""
    order = np.argsort(column)
    keep = np.ones(len(column), dtype=bool)
    keep[order[:min(f, len(order))]] = False
    keep[order[max(0, len(order) - f):]] = False
    return keep


@aggregator_registry.register("wmsr")
class WMSRAggregator(Aggregator):
    """W-MSR filter, applied to each coordinate separately.

    ``variant="standard"`` follows LeBlanc et al., "Resilient asymptotic consensus in robust
    networks" (IEEE JSAC 2013), with equal weights 1/(|R|+1) over the kept values R and the
    agent's own value.  With at most f liars among the neighbours, the result stays inside the
    per-coordinate hull of the honest values and the agent's own: a liar is only ever kept if a
    dropped value lies beyond it, and at least one of those dropped values is honest.

    ``variant="trimmed"`` with ``include_self=False`` reproduces the older unconditional
    trimming, for comparison.

    ``trusted`` is the conservative set the epistemic layer reads: only ids kept in every
    coordinate.  ``value`` still uses, per coordinate, whatever survived in that coordinate.
    """

    VARIANTS = {"standard": _standard_keep, "trimmed": _trimmed_keep}

    def __init__(self, f: int, variant: str = "standard", include_self: bool = True):
        if f < 0:
            raise ValueError("f must be non-negative")
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        self.f, self.variant, self.include_self = f, variant, include_self
        self._keep = self.VARIANTS[variant]

    def aggregate(self, own, values, ids):
        own, values, ids = np.asarray(own, dtype=float), np.asarray(values, dtype=float), np.asarray(ids, dtype=int)
        if len(ids) == 0:
            return AggResult(own.copy(), ids.copy(), ids.copy(), 0)
        value, kept_everywhere = self._filter(own, values)
        return AggResult(value, ids[kept_everywhere], ids[~kept_everywhere], int(kept_everywhere.sum()))

    def _filter(self, own: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        value = own.copy()
        kept_everywhere = np.ones(len(values), dtype=bool)
        for d in range(len(own)):
            keep = self._keep(own[d], values[:, d], self.f)
            kept_everywhere &= keep
            kept = values[keep, d]
            if self.include_self:
                value[d] = (kept.sum() + own[d]) / (len(kept) + 1)
            elif len(kept):
                value[d] = kept.mean()
        return value, kept_everywhere


class FilterLog:
    """Which reports each agent used on this step, per term of its control law.

    ``accepted_pos[i, j]``: i used j's position in every coordinate.  ``rejected_pos[i, j]``: i received
    it but removed it in at least one.  Likewise ``*_vel``.  ``accepted`` means used in every term, and
    ``rejected`` means removed from at least one, which is the conservative pair the epistemic layer
    reads.  A pair with neither was not received, or that term was not computed.
    """

    TERMS = ("pos", "vel")

    def __init__(self, n: int):
        self.accepted_by_term = {term: np.zeros((n, n), dtype=bool) for term in self.TERMS}
        self.rejected_by_term = {term: np.zeros((n, n), dtype=bool) for term in self.TERMS}

    def clear(self) -> None:
        for term in self.TERMS:
            self.accepted_by_term[term].fill(False)
            self.rejected_by_term[term].fill(False)

    def record(self, term: str, i: int, result: AggResult) -> None:
        self.accepted_by_term[term][i, result.trusted] = True
        self.rejected_by_term[term][i, result.rejected] = True

    @property
    def rejected(self) -> np.ndarray:
        return self.rejected_by_term["pos"] | self.rejected_by_term["vel"]

    @property
    def accepted(self) -> np.ndarray:
        return (self.accepted_by_term["pos"] | self.accepted_by_term["vel"]) & ~self.rejected

    def diagnostics(self):
        return {"accepted_pos": self.accepted_by_term["pos"], "rejected_pos": self.rejected_by_term["pos"],
                "accepted_vel": self.accepted_by_term["vel"], "rejected_vel": self.rejected_by_term["vel"],
                "accepted": self.accepted, "rejected": self.rejected}


def build_aggregator(name: str, f: int) -> Aggregator:
    """The aggregator behind a controller's ``filter`` parameter; ``f`` applies only to W-MSR."""
    return aggregator_registry.create("wmsr", f=f) if name == "wmsr" else aggregator_registry.create(name)
