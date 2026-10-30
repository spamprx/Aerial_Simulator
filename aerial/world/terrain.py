"""Terrain generator (T1.1): a 2.5D height map of a city district with randomly collapsed buildings.

Ground truth only: agents never import this module (PLAN §6 import guard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from ..core.config import WorldCfg


@dataclass
class Building:
    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    collapsed: bool


@dataclass
class Terrain:
    H: np.ndarray                 # (nx, ny) heights in metres, cell size `cell`
    cell: float
    arena: float
    zone_origin: np.ndarray       # (2,) lower corner of the search zone
    zone: float
    buildings: List[Building] = field(default_factory=list)

    @property
    def shape(self):
        return self.H.shape

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Height under points ``xy`` (..., 2); outside the arena counts as 0."""
        idx = np.floor(np.asarray(xy) / self.cell).astype(int)
        inside = (idx[..., 0] >= 0) & (idx[..., 1] >= 0) & (idx[..., 0] < self.H.shape[0]) & (idx[..., 1] < self.H.shape[1])
        ix = np.clip(idx[..., 0], 0, self.H.shape[0] - 1)
        iy = np.clip(idx[..., 1], 0, self.H.shape[1] - 1)
        return np.where(inside, self.H[ix, iy], 0.0)

    def free_at(self, z: float, clearance: float) -> np.ndarray:
        """Boolean map of cells where altitude ``z`` clears the terrain by ``clearance`` (horizontally inflated)."""
        blocked = self.H > z - clearance
        r = int(np.ceil(clearance / self.cell))
        if r > 0:
            blocked = ndimage.binary_dilation(blocked, iterations=r)
        return ~blocked


def generate_terrain(cfg: WorldCfg, rng: np.random.Generator) -> Terrain:
    n = int(round(cfg.arena / cfg.terrain_cell))
    H = np.zeros((n, n))
    origin = np.array([(cfg.arena - cfg.zone) / 2, cfg.arena - cfg.zone - 2.0])  # base strip at low y
    origin[1] = cfg.arena - cfg.zone
    buildings: List[Building] = []
    c = cfg.terrain_cell
    # lay out blocks inside the zone: street, block, street, block, ...
    def spans(lo: float, hi: float) -> List[Tuple[float, float]]:
        out, x = [], lo + rng.uniform(cfg.street_min, cfg.street_max)
        while x + cfg.block * 0.6 < hi:
            w = min(cfg.block, hi - x - cfg.street_min / 2)
            out.append((x, x + w))
            x += w + rng.uniform(cfg.street_min, cfg.street_max)
        return out
    zx = spans(origin[0], origin[0] + cfg.zone)
    zy = spans(origin[1], origin[1] + cfg.zone)
    for (ax, bx) in zx:
        for (ay, by) in zy:
            # each block holds 1-2 buildings split along its longer side
            if rng.random() < 0.5:
                parts = [(ax, bx, ay, by)]
            else:
                m = (ax + bx) / 2
                parts = [(ax, m - 1, ay, by), (m + 1, bx, ay, by)]
            for (x0, x1, y0, y1) in parts:
                i0, i1, j0, j1 = int(x0 / c), int(x1 / c), int(y0 / c), int(y1 / c)
                if i1 - i0 < 3 or j1 - j0 < 3:
                    continue
                h = rng.uniform(*cfg.building_h)
                collapsed = rng.random() < cfg.p_collapse
                b = Building(i0, j0, i1, j1, h, collapsed)
                buildings.append(b)
                if not collapsed:
                    H[i0:i1, j0:j1] = np.maximum(H[i0:i1, j0:j1], h)
                else:
                    _rubble(H, b, cfg, rng, c)
    return Terrain(H=H, cell=c, arena=cfg.arena, zone_origin=origin, zone=cfg.zone, buildings=buildings)


def _rubble(H: np.ndarray, b: Building, cfg: WorldCfg, rng: np.random.Generator, c: float) -> None:
    """Footprint becomes a noisy rubble mound; debris spills into the street in a random direction."""
    lo, hi = cfg.rubble_h
    fp = H[b.x0:b.x1, b.y0:b.y1]
    mound = rng.uniform(lo, hi, size=fp.shape)
    mound = ndimage.uniform_filter(mound, size=3)
    # a few taller piles that rise above the cruise layers
    if rng.random() < 0.3:
        px, py = rng.integers(0, fp.shape[0]), rng.integers(0, fp.shape[1])
        mound[max(0, px - 2):px + 2, max(0, py - 2):py + 2] = rng.uniform(13.0, 18.0)
    H[b.x0:b.x1, b.y0:b.y1] = np.maximum(fp, mound)
    # spill ellipse
    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    ang = rng.uniform(0, 2 * np.pi)
    reach = 0.5 * max(b.x1 - b.x0, b.y1 - b.y0) + rng.uniform(3, 7) / c
    ex, ey = cx + np.cos(ang) * reach * 0.7, cy + np.sin(ang) * reach * 0.7
    a, bb = reach * 0.6, reach * 0.35
    n = H.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dx, dy = X - ex, Y - ey
    u = (dx * np.cos(ang) + dy * np.sin(ang)) / a
    v = (-dx * np.sin(ang) + dy * np.cos(ang)) / bb
    inside = u * u + v * v <= 1
    spill = rng.uniform(lo, hi * 0.8, size=H.shape) * (1 - np.sqrt(np.clip(u * u + v * v, 0, 1)))
    H[inside] = np.maximum(H[inside], spill[inside])
