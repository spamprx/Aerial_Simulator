"""Geometry kernel on the height map (T1.3): ray marching, line of sight, the shared ``viewed`` test (A7).

Ray marching uses a step of a quarter cell, so a 1 m column is never skipped by a ray crossing it squarely.
One line-of-sight function is shared by coverage, survivor detection and drone sightings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import Terrain


def los_clear(terrain: Terrain, a: np.ndarray, b: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Line of sight between point arrays ``a`` (..., 3) and ``b`` (..., 3): True if no column blocks it."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    d = b - a
    L = np.linalg.norm(d, axis=-1)
    n = max(2, int(np.ceil(L.max() / step)) if L.size else 2)
    s = np.linspace(0.0, 1.0, n)[1:-1]                       # exclude endpoints
    pts = a[..., None, :] + d[..., None, :] * s[:, None]     # (..., n-2, 3)
    h = terrain.height_at(pts[..., :2])
    return np.all(pts[..., 2] > h + 1e-6, axis=-1)


def raycast(terrain: Terrain, origins: np.ndarray, dirs: np.ndarray, max_range: float, step: float = 0.25):
    """March rays; return (hit_dist (M,), hit_point (M,3), samples (M,S,3), sample_valid (M,S)).

    ``samples`` are points the ray passed above the terrain (so H < z there); the hit point proves H >= z_hit.
    Rays that go below z = 0 hit the ground.
    """
    ts = np.arange(step, max_range + 1e-9, step)
    pts = origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]       # (M, S, 3)
    h = terrain.height_at(pts[..., :2])
    below = pts[..., 2] <= np.maximum(h, 0.0)
    any_hit = below.any(axis=1)
    first = np.where(any_hit, below.argmax(axis=1), len(ts))
    hit_dist = np.where(any_hit, ts[np.minimum(first, len(ts) - 1)], np.inf)
    idx = np.minimum(first, len(ts) - 1)
    hit_point = pts[np.arange(len(origins)), idx]
    valid = np.arange(len(ts))[None, :] < first[:, None]
    return hit_dist, hit_point, pts, valid


@dataclass
class CoverageGrid:
    """2 m cells over the search zone. Cell id = a * n + b."""
    origin: np.ndarray
    size: float
    n: int
    target: np.ndarray      # (n*n, 3) target point of each cell (centre, top of 2x2 terrain block + 0.3 m)

    @property
    def n_cells(self) -> int:
        return self.n * self.n

    def cell_of(self, xy: np.ndarray) -> np.ndarray:
        ab = np.floor((np.asarray(xy)[..., :2] - self.origin) / self.size).astype(int)
        ok = np.all((ab >= 0) & (ab < self.n), axis=-1)
        return np.where(ok, ab[..., 0] * self.n + ab[..., 1], -1)

    def centres(self) -> np.ndarray:
        return self.target[:, :2]


def make_coverage_grid(terrain: Terrain, size: float) -> CoverageGrid:
    n = int(round(terrain.zone / size))
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx = terrain.zone_origin[0] + (a.ravel() + 0.5) * size
    cy = terrain.zone_origin[1] + (b.ravel() + 0.5) * size
    k = int(round(size / terrain.cell))
    i0 = np.floor((cx - size / 2) / terrain.cell).astype(int)
    j0 = np.floor((cy - size / 2) / terrain.cell).astype(int)
    hmax = np.zeros(n * n)
    for di in range(k):
        for dj in range(k):
            hmax = np.maximum(hmax, terrain.H[np.clip(i0 + di, 0, terrain.shape[0] - 1),
                                                np.clip(j0 + dj, 0, terrain.shape[1] - 1)])
    target = np.stack([cx, cy, hmax + 0.3], axis=1)
    return CoverageGrid(origin=np.array(terrain.zone_origin, float), size=size, n=n, target=target)


def in_frustum(pose_pos: np.ndarray, yaw: float, pts: np.ndarray, fov_h: float, fov_v: float, tilt: float,
               max_range: float) -> np.ndarray:
    """Points inside the camera frustum (angles in degrees; tilt = axis angle below horizontal)."""
    d = pts - pose_pos
    dist = np.linalg.norm(d, axis=-1)
    bearing = np.arctan2(d[..., 1], d[..., 0])
    dbear = np.abs((bearing - yaw + np.pi) % (2 * np.pi) - np.pi)
    depression = np.degrees(np.arctan2(-d[..., 2], np.linalg.norm(d[..., :2], axis=-1) + 1e-9))
    return ((dist <= max_range) & (np.degrees(dbear) <= fov_h / 2)
            & (np.abs(depression - tilt) <= fov_v / 2))


def viewed(terrain: Terrain, grid: CoverageGrid, pos: np.ndarray, yaw: float, sensor, candidates=None) -> np.ndarray:
    """Coverage cell ids viewed from (pos, yaw): in frustum, within r_detect, elevation >= theta_min, LOS (A7)."""
    ids = np.arange(grid.n_cells) if candidates is None else np.asarray(candidates)
    tgt = grid.target[ids]
    d = tgt - pos
    horiz = np.linalg.norm(d[:, :2], axis=1)
    near = horiz <= sensor.r_detect
    ids, tgt, d, horiz = ids[near], tgt[near], d[near], horiz[near]
    elev = np.degrees(np.arctan2(-d[:, 2], horiz + 1e-9))
    ok = (elev >= sensor.theta_min) & in_frustum(pos, yaw, tgt, sensor.fov_h, sensor.fov_v, sensor.tilt,
                                                   sensor.r_detect)
    ids, tgt = ids[ok], tgt[ok]
    if len(ids) == 0:
        return ids
    clear = los_clear(terrain, np.broadcast_to(pos, tgt.shape), tgt)
    return ids[clear]


def viewable_mask(terrain: Terrain, grid: CoverageGrid, sensor, layers, clearance: float = 1.5) -> np.ndarray:
    """Ground truth: cells viewable from some pose in the altitude band (any yaw)."""
    ok = np.zeros(grid.n_cells, bool)
    tgt = grid.target
    for z in layers:
        for r in (5.0, 7.0, 9.0):
            for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                pose = tgt.copy()
                pose[:, 0] += r * np.cos(ang)
                pose[:, 1] += r * np.sin(ang)
                pose[:, 2] = z
                d = tgt - pose
                horiz = np.linalg.norm(d[:, :2], axis=1)
                dist = np.linalg.norm(d, axis=1)
                elev = np.degrees(np.arctan2(-d[:, 2], horiz))
                cand = ((~ok) & (dist <= sensor.r_detect) & (elev >= sensor.theta_min)
                        & (elev <= sensor.tilt + sensor.fov_v / 2)
                        & (terrain.height_at(pose[:, :2]) < z - clearance))
                idx = np.flatnonzero(cand)
                if len(idx):
                    ok[idx[los_clear(terrain, pose[idx], tgt[idx])]] = True
    return ok
