"""Optional CUDA backend for camera ray casting (the GPU-friendly third of a run).

Set ``AERIAL_DEVICE=cuda`` (and pin a card with ``CUDA_VISIBLE_DEVICES``) to cast every drone's rays in one batched
torch call on the GPU. Arithmetic is float64, mirroring ``geometry.raycast`` operation for operation, so results
match the numpy path (tests/world/test_gpu.py checks this when a GPU is present). With the variable unset, torch
is never imported and nothing changes.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_DEVICE = os.environ.get("AERIAL_DEVICE", "cpu").lower()


def enabled() -> bool:
    return _DEVICE.startswith("cuda")


class GpuTerrain:
    """The height map resident on the GPU, uploaded once per scenario."""

    def __init__(self, terrain):
        import torch
        self.torch = torch
        if _DEVICE == "cuda:auto":      # spread worker processes over the visible GPUs
            self.dev = torch.device(f"cuda:{os.getpid() % max(1, torch.cuda.device_count())}")
        else:
            self.dev = torch.device(_DEVICE)
        self.H = torch.as_tensor(terrain.H, dtype=torch.float64, device=self.dev)
        self.cell = float(terrain.cell)
        self.n0, self.n1 = terrain.H.shape

    def height_at(self, xy):
        t = self.torch
        idx = t.floor(xy / self.cell).to(t.int64)
        ix, iy = idx[..., 0], idx[..., 1]
        inside = (ix >= 0) & (iy >= 0) & (ix < self.n0) & (iy < self.n1)
        h = self.H[ix.clamp(0, self.n0 - 1), iy.clamp(0, self.n1 - 1)]
        return t.where(inside, h, t.zeros_like(h))

    def raycast(self, origins: np.ndarray, dirs: np.ndarray, max_range: float, step: float = 0.25):
        """Same contract as ``geometry.raycast``; returns numpy arrays."""
        t = self.torch
        ts_np = np.arange(step, max_range + 1e-9, step)
        ts = t.as_tensor(ts_np, dtype=t.float64, device=self.dev)
        o = t.as_tensor(origins, dtype=t.float64, device=self.dev)
        d = t.as_tensor(dirs, dtype=t.float64, device=self.dev)
        pts = o[:, None, :] + d[:, None, :] * ts[None, :, None]
        h = self.height_at(pts[..., :2])
        below = pts[..., 2] <= t.clamp(h, min=0.0)
        any_hit = below.any(dim=1)
        first = t.where(any_hit, below.to(t.int8).argmax(dim=1), t.full_like(any_hit, len(ts_np), dtype=t.int64))
        idx = first.clamp(max=len(ts_np) - 1)
        hit_dist = t.where(any_hit, ts[idx], t.full_like(ts[idx], float("inf")))
        hit_point = pts[t.arange(len(o), device=self.dev), idx]
        valid = t.arange(len(ts_np), device=self.dev)[None, :] < first[:, None]
        return hit_dist.cpu().numpy(), hit_point.cpu().numpy(), pts.cpu().numpy(), valid.cpu().numpy()


def gpu_terrain(terrain) -> Optional[GpuTerrain]:
    """The cached GPU copy of ``terrain`` when the CUDA backend is on, else None."""
    if not enabled():
        return None
    g = getattr(terrain, "_gpu", None)
    if g is None:
        g = GpuTerrain(terrain)
        terrain._gpu = g
    return g
