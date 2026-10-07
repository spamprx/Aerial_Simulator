"""Controller interface + the trivial baseline.

A controller is *decentralised by construction*: ``control`` receives an ``Observation``
holding every agent's own true state and the messages it actually received, and returns
one acceleration command per agent.  Agent i must only use row i of ``obs.adj`` and the
messages it holds (``obs.received(i)``).

Other modules plug in here:
  * coordination  -> register new controllers (rendezvous, formation, BOIDS, ...)
  * resilience    -> wrap the neighbour sums with W-MSR filtering
  * epistemic     -> wrap ``control`` with the knowledge gate (hold unless K_i(safe))
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from ..registry import controller_registry


class Controller(ABC):
    def reset(self, sim) -> None:
        """Called once per run (access to sim.cfg, sim.n, sim.dim ...)."""

    @abstractmethod
    def control(self, obs) -> np.ndarray:
        """Return u with shape (N, dim)."""

    def diagnostics(self) -> Dict[str, np.ndarray]:
        """Arrays describing the latest ``control`` call.  ``Simulator.run`` copies each one
        every step and stacks it into ``SimResult.extras[key]`` with shape (K, ...)."""
        return {}

    def filter_offsets(self) -> Optional[np.ndarray]:
        """(N, d) offsets if neighbour positions are compared in a shifted frame, as in a formation."""
        return None

    def normalised_gains(self) -> Optional[Tuple[float, float]]:
        """(position gain, velocity gain) if this controller uses mean neighbour weights (D11), which
        is what the stability check in ``aerial/stability.py`` needs to know."""
        return None

    def requery_requests(self) -> Optional[np.ndarray]:
        """(N,) bool: which agents ask for a retransmission on the next step (D4).  None if nobody ever does."""
        return None

    def broadcast_payload(self) -> Dict[str, np.ndarray]:
        """Arrays (N, ...) that each agent adds to its next broadcast, read as ``obs.payload_for``.  They arrive
        one step late: what ``control`` decides on step k goes out with step k + 1's message."""
        return {}


class SplitController(Controller):
    """A controller whose output is a gated part and a part that is never gated.

    ``advance`` is what moves the agent towards its objective; a knowledge gate may withhold
    it.  ``safety`` (separation, obstacle and geofence repulsion) always applies.
    """

    @abstractmethod
    def control_parts(self, obs) -> Dict[str, np.ndarray]:
        """Return {"advance": (N, dim), "safety": (N, dim)}."""

    def control(self, obs):
        parts = self.control_parts(obs)
        return parts["advance"] + parts["safety"]


@controller_registry.register("zero")
class ZeroController(Controller):
    """Coast: no acceleration.  Useful as a sanity baseline."""

    def control(self, obs):
        return np.zeros_like(obs.pos)
