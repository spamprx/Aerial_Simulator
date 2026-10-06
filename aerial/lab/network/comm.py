"""Communication models: turn agent positions into this step's receive-graph G(t)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import SimConfig
from .graph import disk_adjacency, make_topology, packet_keep_mask


class CommModel(ABC):
    """Produces the (N, N) boolean receive matrix for the current step."""

    # Raw keep mask from the latest ``adjacency`` call, before it meets the graph.  Models
    # without Bernoulli drops leave it None, and the run then records no ``drop_keep``.
    last_keep: Optional[np.ndarray] = None

    @abstractmethod
    def adjacency(self, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    def reachable(self, pos: np.ndarray) -> np.ndarray:
        """(N, N) the links that exist before any packet is lost, which is what a re-query can use."""
        raise NotImplementedError(f"{type(self).__name__} does not support re-queries")


class DiskComm(CommModel):
    """Distance-bounded disk graph + Bernoulli packet drops  =>  time-varying G(t)."""

    def __init__(self, radius: float, drop_prob: float = 0.0, symmetric_drops: bool = False):
        self.radius, self.drop_prob, self.symmetric = radius, drop_prob, symmetric_drops

    def adjacency(self, pos, rng):
        A = disk_adjacency(pos, self.radius)
        self.last_keep = packet_keep_mask(A.shape, self.drop_prob, rng, self.symmetric)
        return A & self.last_keep

    def reachable(self, pos):
        return disk_adjacency(pos, self.radius)


class FixedComm(CommModel):
    """A fixed topology (ring, grid, Erdos-Renyi, ...) with optional packet drops."""

    def __init__(self, base: np.ndarray, drop_prob: float = 0.0, symmetric_drops: bool = False):
        self.base, self.drop_prob, self.symmetric = base, drop_prob, symmetric_drops

    def adjacency(self, pos, rng):
        self.last_keep = packet_keep_mask(self.base.shape, self.drop_prob, rng, self.symmetric)
        return self.base & self.last_keep

    def reachable(self, pos):
        return self.base


def build_comm_model(cfg: SimConfig, rng: np.random.Generator) -> CommModel:
    if cfg.topology == "disk":
        return DiskComm(cfg.comm_radius, cfg.drop_prob, cfg.symmetric_drops)
    base = make_topology(cfg.topology, cfg.n_agents, rng, **cfg.topology_params)
    return FixedComm(base, cfg.drop_prob, cfg.symmetric_drops)
