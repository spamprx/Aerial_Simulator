"""Second-order consensus: rendezvous and velocity alignment."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..registry import controller_registry
from ..resilience.aggregators import FilterLog, build_aggregator
from .base import Controller


def default_damping(filter: Optional[str]) -> str:
    """Own-velocity damping under W-MSR, which does not conserve mean velocity; relative otherwise."""
    return "own" if filter == "wmsr" else "relative"


@controller_registry.register("consensus")
class SecondOrderConsensus(Controller):
    """Second-order (position + velocity) consensus on the received messages:

        u_i = k_p * sum_j a_ij (p~_j - p_i)  +  k_v * sum_j a_ij (v~_j - v_i)

    ``k_p > 0, k_v = 0`` is oscillatory rendezvous; ``k_p = 0`` is pure velocity alignment;
    both positive gives rendezvous with damping.  With no ``filter`` this is the plain, summed,
    *unfiltered* baseline that W-MSR and the knowledge gate are compared against.

    With ``filter`` ("mean" or "wmsr") the sums become normalised aggregates (D11):

        u_i = k_p * (agg({p~_j}) - p_i)  +  k_v * (agg({v~_j}) - v_i)

    W-MSR's ``f`` defaults to ``cfg.F``.  ``damping="own"`` replaces the velocity term with -k_v * v_i,
    which needs no message at all, and it is the default under W-MSR (DP-MSR; Dibaji and Ishii, Systems
    & Control Letters, 2015).  W-MSR trims asymmetrically, so unlike a mean it does not conserve the
    fleet's average velocity.  With relative damping nothing brings that back to zero: the agents
    agree perfectly, and all drift away together, far outside the hull of their starting positions.
    """

    DAMPING = ("relative", "own")

    def __init__(self, k_p: float = 1.0, k_v: float = 1.5, filter: Optional[str] = None, f: Optional[int] = None,
                 damping: Optional[str] = None):
        damping = default_damping(filter) if damping is None else damping
        if damping not in self.DAMPING:
            raise ValueError(f"damping must be one of {self.DAMPING}")
        if damping == "own" and filter is None:
            raise ValueError('damping="own" needs a filter: the summed law is the unfiltered baseline')
        self.k_p, self.k_v = k_p, k_v
        self.filter, self.f, self.damping = filter, f, damping
        self.aggregator = None
        self.log = None

    def reset(self, sim) -> None:
        if self.filter is not None:
            self.aggregator = build_aggregator(self.filter, sim.cfg.F if self.f is None else self.f)
            self.log = FilterLog(sim.n)

    def normalised_gains(self):
        return None if self.filter is None else (self.k_p, self.k_v)

    def control(self, obs):
        return self._summed(obs) if self.filter is None else self._filtered(obs)

    def _summed(self, obs):
        A = obs.adj.astype(float)
        deg = A.sum(axis=1, keepdims=True)
        pos_sum, vel_sum = obs.received_sums()
        pos_term = pos_sum - deg * obs.pos
        vel_term = vel_sum - deg * obs.vel
        return self.k_p * pos_term + self.k_v * vel_term

    def _filtered(self, obs):
        u = np.zeros_like(obs.pos)
        self.log.clear()
        for i in range(obs.n):
            ids, pos, vel = obs.received(i)
            position = self.aggregator.aggregate(obs.pos[i], pos, ids)
            self.log.record("pos", i, position)
            u[i] = self.k_p * (position.value - obs.pos[i])
            if self.damping == "own":
                u[i] -= self.k_v * obs.vel[i]
            else:
                velocity = self.aggregator.aggregate(obs.vel[i], vel, ids)
                self.log.record("vel", i, velocity)
                u[i] += self.k_v * (velocity.value - obs.vel[i])
        return u

    def diagnostics(self):
        return {} if self.log is None else self.log.diagnostics()
