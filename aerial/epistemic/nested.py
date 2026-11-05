"""Nested and group knowledge, and the size of the abstraction (T3.14).

- ``knows_that_knows(agent_i, j, L)``: K_i K_j (L faulty), approximated soundly by "i holds j's KNOWN_FAULTY payload
  naming L, and i knows j is honest" (an honest j only names proven liars). Sound, not complete.
- ``everyone_knows_time``: ground-truth time at which E_G(L faulty) first holds for the honest group G.
- ``log10_worlds``: exact |I_i| = sum over S in H_i of 2^u(S), where u(S) counts cells still uncertain if S were the
  faulty set; computed in log space. Adding log|H_i| to the uncertain-cell count would over-count.
"""
from __future__ import annotations

import numpy as np

from .hypotheses import Hypotheses


def knows_that_knows(hyp: Hypotheses, payloads: dict, j: int, liar: int) -> bool:
    return bool(hyp.known_honest()[j] and liar in payloads.get(j, ()))


def everyone_knows_time(known_faulty: np.ndarray, honest: np.ndarray, liars: np.ndarray, dt: float) -> float:
    if len(liars) == 0:
        return np.nan
    e = known_faulty[:, honest][:, :, liars].all(axis=(1, 2))
    return float(np.argmax(e) * dt) if e.any() else np.nan


def log10_worlds(hyp: Hypotheses, own: np.ndarray, reporters: np.ndarray) -> float:
    """log10 sum_S 2^{u(S)}, u(S) = cells neither seen by me nor reported by someone outside S."""
    W = hyp.worlds
    if len(W) == 0:
        return float("nan")
    rep = np.asarray(reporters, dtype=np.uint64)[~np.asarray(own, bool)]
    u = ((rep[:, None] & ~W[None, :]) == 0).sum(axis=0).astype(float)    # (|W|,)
    l2 = u * np.log10(2.0)
    m = l2.max()
    return float(m + np.log10(np.exp((l2 - m) * np.log(10)).sum()))
