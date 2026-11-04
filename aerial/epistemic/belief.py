"""Alive/dead as a belief B_i (T2.11): log-odds from heat and motion looks, decided by an SPRT.

This is belief, not knowledge: the posterior can be wrong. The report contrasts it with the S5 knowledge of who lied.
"""
from __future__ import annotations

import math

from scipy.stats import norm

HEAT_ALIVE = (34.0, 1.0)
HEAT_DEAD = (27.5, 1.9)          # ambient 25-30 C, uniform -> approximated by a Gaussian
P_MOTION_ALIVE = 0.6 * 0.6 + 0.25 * 0.4
P_MOTION_DEAD = 0.02


def look_llr(heat: float, motion: bool) -> float:
    """log p(look | alive) / p(look | dead)."""
    llr = norm.logpdf(heat, *HEAT_ALIVE) - norm.logpdf(heat, *HEAT_DEAD)
    if motion:
        llr += math.log(P_MOTION_ALIVE / P_MOTION_DEAD)
    else:
        llr += math.log((1 - P_MOTION_ALIVE) / (1 - P_MOTION_DEAD))
    return float(max(-8.0, min(8.0, llr)))


class AliveBelief:
    def __init__(self, upper: float = 0.95, lower: float = 0.05, prior: float = 0.5):
        self.L = math.log(prior / (1 - prior))
        self.A = math.log(upper / (1 - upper))
        self.B = math.log(lower / (1 - lower))
        self.n = 0

    def update(self, heat: float, motion: bool) -> None:
        self.L += look_llr(heat, motion)
        self.n += 1

    @property
    def p_alive(self) -> float:
        return 1 / (1 + math.exp(-self.L))

    @property
    def decision(self):
        if self.L >= self.A:
            return "alive"
        if self.L <= self.B:
            return "dead"
        return None
