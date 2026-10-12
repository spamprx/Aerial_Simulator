"""The F-local adversary model: no honest agent hears more than F compromised agents.

This limits each neighbourhood, not the fleet, so a fleet can hold many more than F compromised agents
as long as they are spread out.
"""
from __future__ import annotations

import numpy as np


def is_f_local(A: np.ndarray, compromised: np.ndarray, F: int) -> bool:
    """Whether every honest agent has at most F compromised in-neighbours.  ``A[i, j]``: i hears j."""
    compromised = np.asarray(compromised, dtype=bool)
    return bool((np.asarray(A)[~compromised][:, compromised].sum(axis=1) <= F).all())


def pick_compromised_f_local(A: np.ndarray, n: int, F: int, rng: np.random.Generator, tries: int = 2000) -> np.ndarray:
    """Sample a (N,) bool mask of ``n`` compromised agents that is F-local on ``A``.

    Rejection sampling, so every valid set of that size is equally likely.  Raises ValueError if ``tries``
    draws find none, which on a dense graph usually means none exists.
    """
    size = len(A)
    if not 0 <= n <= size:
        raise ValueError(f"cannot pick {n} compromised agents out of {size}")
    for _ in range(tries):
        mask = np.zeros(size, dtype=bool)
        mask[rng.choice(size, n, replace=False)] = True
        if is_f_local(A, mask, F):
            return mask
    raise ValueError(f"found no F-local set of {n} compromised agents with F = {F} in {tries} draws")
