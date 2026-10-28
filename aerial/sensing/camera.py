"""Camera sensing (T1.4–T1.6), vectorised in the simulator.

One gimbal camera per drone produces, per frame:
- terrain interval evidence (A18): points the rays passed *above* (H < z there) and hit points (H >= z there);
- the coverage cells it viewed (A7) and a survivor look for every survivor in them (exact cell, T1.5);
- a depth image, wrapped as a ``VisibleVolume``; drone sightings use exactly that depth image, so
  ``contains(x)`` implies a drone at x would have been sighted (T1.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.types import SensorFrame
from ..world.geometry import raycast, viewed
from ..world.gpu import gpu_terrain


@dataclass(frozen=True)
class SurvivorLook:
    cell: int
    pos_est: np.ndarray
    heat: float
    motion: bool
    t: float


@dataclass
class VisibleVolume:
    """What a frame proves empty: per ray direction, free up to the measured depth (capped at r_see)."""
    pos: np.ndarray
    yaw: float
    az: np.ndarray        # (rays_h,) bearing offsets (rad)
    el: np.ndarray        # (rays_v,) depression angles (rad, positive = below horizontal)
    depth: np.ndarray     # (rays_v, rays_h)
    r_see: float

    def contains(self, x: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where points ``x`` (..., 3) lie well inside the seen volume (by ``margin`` metres)."""
        d = np.asarray(x, float) - self.pos
        dist = np.linalg.norm(d, axis=-1)
        bearing = (np.arctan2(d[..., 1], d[..., 0]) - self.yaw + np.pi) % (2 * np.pi) - np.pi
        dep = np.arctan2(-d[..., 2], np.linalg.norm(d[..., :2], axis=-1) + 1e-9)
        ok = (bearing >= self.az[0]) & (bearing <= self.az[-1]) & (dep >= self.el[0]) & (dep <= self.el[-1])
        ih = np.clip(np.searchsorted(self.az, bearing), 0, len(self.az) - 1)
        iv = np.clip(np.searchsorted(self.el, dep), 0, len(self.el) - 1)
        # conservative: use the minimum depth over the 2x2 neighbouring rays
        ih0, iv0 = np.clip(ih - 1, 0, None), np.clip(iv - 1, 0, None)
        dmin = np.minimum.reduce([self.depth[iv, ih], self.depth[iv0, ih], self.depth[iv, ih0], self.depth[iv0, ih0]])
        return ok & (dist + margin <= np.minimum(dmin, self.r_see))


def camera_angles(sensor):
    az = np.radians(np.linspace(-sensor.fov_h / 2, sensor.fov_h / 2, sensor.rays_h))
    el = np.radians(np.linspace(sensor.tilt - sensor.fov_v / 2, sensor.tilt + sensor.fov_v / 2, sensor.rays_v))
    return az, el


def sense_all(scn, pos: np.ndarray, yaw: np.ndarray, t: float, sensor, rng) -> List[SensorFrame]:
    """Build one frame per drone from ground truth. ``rng`` is the paired ``sense`` stream."""
    N = len(pos)
    az, el = camera_angles(sensor)
    A, E = np.meshgrid(az, el)                                     # (rays_v, rays_h)
    frames = []
    surv_cells = scn.survivor_cells
    # paired draws: a fixed-size block per tick, whatever happens
    noise = rng.standard_normal((N, max(1, len(scn.survivors)), 5))
    nr = A.size
    b = (yaw[:, None] + A.ravel()[None, :]).ravel()
    e = np.tile(E.ravel(), N)
    dirs = np.stack([np.cos(e) * np.cos(b), np.cos(e) * np.sin(b), -np.sin(e)], axis=1)
    origins = np.repeat(pos, nr, axis=0)
    g = gpu_terrain(scn.terrain)
    if g is not None:
        HD, HP, SM, VA = g.raycast(origins, dirs, sensor.r_depth, step=0.4)
    else:
        HD, HP, SM, VA = raycast(scn.terrain, origins, dirs, sensor.r_depth, step=0.4)
    for i in range(N):
        sl = slice(i * nr, (i + 1) * nr)
        hit_d, hit_p, samples, valid = HD[sl], HP[sl], SM[sl], VA[sl]
        depth = np.minimum(hit_d, sensor.r_depth).reshape(E.shape)
        over = samples[valid][::3]
        hits = hit_p[np.isfinite(hit_d)]
        vis = VisibleVolume(pos=pos[i].copy(), yaw=float(yaw[i]), az=az, el=el, depth=depth, r_see=sensor.r_see)
        cells = viewed(scn.terrain, scn.grid, pos[i], float(yaw[i]), sensor)
        looks = []
        if len(cells) and len(surv_cells):
            for k, s in enumerate(scn.survivors):
                if s.cell in set(cells.tolist()):
                    rng_d = np.linalg.norm(s.pos - pos[i])
                    sig = 0.2 + 0.05 * rng_d
                    est = s.pos + noise[i, k, :3] * sig
                    heat = s.heat_mean + noise[i, k, 3]
                    motion = bool(np.abs(noise[i, k, 4]) < _z_for(s.p_motion))
                    looks.append(SurvivorLook(cell=s.cell, pos_est=est, heat=float(heat), motion=motion, t=t))
        others = np.delete(pos, i, axis=0)
        seen = others[vis.contains(others)]
        frames.append(SensorFrame(t=t, terrain_over=over, terrain_hit=hits, viewed=cells, looks=looks,
                                  sightings=seen, visible=vis))
    return frames


def _z_for(p: float) -> float:
    """|Z| < z has probability p for a standard normal Z (so the motion bit has the right rate)."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + p / 2))
