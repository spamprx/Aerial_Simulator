"""The coarse, finite abstraction of space the epistemic layer reasons over (proposal section 3).

Space is cut into closed cubes of side h, and a cube is named by integer indices: cell c covers
[c * h, (c + 1) * h] in each coordinate, wherever the arena lies.  ``Grid`` knows the cells of the arena plus
``margin`` on every side, and a set of cells is an (M, dim) integer array.

Every set is an over-approximation: it holds each cell that touches the region, so a point in the region is
always in a cell of the set.  That is the whole of the abstraction's soundness, and it holds for regions
inside the grid.  The cells are clipped to the grid, so a region beyond it is simply not represented.  That is
harmless as long as ``margin`` is wide enough that nothing out there could touch a plan inside the arena.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_TOLERANCE = 1e-9  # rounding must never drop a cell that touches a region's edge


class Grid:
    def __init__(self, h: float, arena_size: float, dim: int, margin: float = 0.0):
        if h <= 0:
            raise ValueError("cell side h must be positive")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self.h, self.arena_size, self.dim, self.margin = float(h), float(arena_size), int(dim), float(margin)
        self.low = int(np.floor(-margin / h))
        self.high = int(np.ceil((arena_size + margin) / h)) - 1
        self.side = self.high - self.low + 1
        self.n_cells = self.side ** self.dim
        self._all_cells = None

    # ---- single cells -------------------------------------------------------------------------
    def cell_of(self, p: np.ndarray) -> np.ndarray:
        """(..., dim) the index of the cell holding each point, in or out of the grid."""
        return np.floor(np.asarray(p, dtype=float) / self.h).astype(int)

    def center(self, cells: np.ndarray) -> np.ndarray:
        return (np.asarray(cells) + 0.5) * self.h

    def contains(self, cells: np.ndarray) -> np.ndarray:
        """(M,) which of these cells are inside the grid."""
        return ((cells >= self.low) & (cells <= self.high)).all(axis=-1)

    def all_cells(self) -> np.ndarray:
        if self._all_cells is None:
            self._all_cells = self._box(np.full(self.dim, self.low), np.full(self.dim, self.high))
        return self._all_cells

    # ---- distances ----------------------------------------------------------------------------
    def distance_range(self, cells: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(nearest, farthest), each (P, M): the distance from each of P points to the near and far
        side of each of M cells."""
        points = np.asarray(points, dtype=float)[:, None, :]
        low = cells[None, :, :] * self.h - points
        high = low + self.h
        nearest = np.maximum(np.maximum(low, -high), 0.0)
        farthest = np.maximum(np.abs(low), np.abs(high))
        return np.sqrt((nearest ** 2).sum(axis=-1)), np.sqrt((farthest ** 2).sum(axis=-1))

    def _box(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Every cell of the grid between these corners, inclusive."""
        low, high = np.maximum(low, self.low), np.minimum(high, self.high)
        if np.any(high < low):
            return np.empty((0, self.dim), dtype=int)
        axes = np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(low, high)), indexing="ij")
        return np.stack([axis.ravel() for axis in axes], axis=1)

    def _box_around(self, p: np.ndarray, r: float) -> np.ndarray:
        """Cells closed cubes can touch within ``r``.  One extra on each side, since a point on a face
        belongs to the cells on both sides of it, though ``cell_of`` names only one."""
        return self._box(self.cell_of(p - r) - 1, self.cell_of(p + r) + 1)

    # ---- regions ------------------------------------------------------------------------------
    def cells_in_ball(self, p: np.ndarray, r: float) -> np.ndarray:
        """The cells that touch the ball of radius ``r`` around ``p``."""
        p = np.asarray(p, dtype=float)
        box = self._box_around(p, r)
        nearest, _ = self.distance_range(box, p[None])
        return box[nearest[0] <= r + _TOLERANCE]

    def cells_in_shell(self, p: np.ndarray, r: float, eps: float) -> np.ndarray:
        """The cells that touch the set of points whose distance from ``p`` is within ``eps`` of ``r``."""
        p = np.asarray(p, dtype=float)
        box = self._box_around(p, r + eps)
        nearest, farthest = self.distance_range(box, p[None])
        return box[(nearest[0] <= r + eps + _TOLERANCE) & (farthest[0] >= r - eps - _TOLERANCE)]

    def dilate(self, cells: np.ndarray, r: float) -> np.ndarray:
        """Every cell within ``r`` of one of these: it covers all points within ``r`` of a point in them."""
        if len(cells) == 0:
            return cells
        reach = int(np.ceil(r / self.h)) + 1
        offsets = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * self.dim, indexing="ij"), axis=-1)
        offsets = offsets.reshape(-1, self.dim)
        # Two cells whose indices differ by o are h * max(|o| - 1, 0) apart in each coordinate.
        gap = np.maximum(np.abs(offsets) - 1, 0) * self.h
        offsets = offsets[np.sqrt((gap ** 2).sum(axis=1)) <= r + _TOLERANCE]
        reached = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, self.dim)
        return np.unique(reached[self.contains(reached)], axis=0)

    def cells_near(self, points: np.ndarray, r: float) -> np.ndarray:
        """The cells of the grid whose centre is less than ``r`` from one of these points."""
        low = np.ceil((points.min(axis=0) - r) / self.h - 0.5).astype(int)
        high = np.floor((points.max(axis=0) + r) / self.h - 0.5).astype(int)
        box = self._box(low, high)
        if len(box) == 0:
            return box
        distance = np.linalg.norm(self.center(box)[:, None, :] - points[None, :, :], axis=-1)
        return box[(distance < r).any(axis=1)]

    # ---- counting a region's cells without listing them ---------------------------------------
    def count_in_balls(self, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """(P,) how many cells ``cells_in_ball`` would return for each point.  Each axis is worked out once."""
        if len(points) == 0:
            return np.zeros(0, dtype=int)
        reach = min(int(np.ceil(radii.max() / self.h)) + 2, self.side)
        first = np.clip(self.cell_of(points) - reach, self.low, self.high + 1 - min(2 * reach + 1, self.side))
        index = first[:, :, None] + np.arange(min(2 * reach + 1, self.side))  # (P, d, width)
        low = index * self.h - points[:, :, None]
        gap = np.maximum(np.maximum(low, -(low + self.h)), 0.0) ** 2
        total = gap[:, 0]
        for axis in range(1, self.dim):
            total = total[..., None] + gap[:, axis].reshape(len(points), *([1] * axis), -1)
        limit = (radii + _TOLERANCE) ** 2
        return np.count_nonzero(total <= limit.reshape(-1, *([1] * self.dim)), axis=tuple(range(1, self.dim + 1)))

    def count_in_shells(self, origin: np.ndarray, ranges: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """(P,) how many cells ``cells_in_shell`` would return for these shells around one origin."""
        if len(ranges) == 0:
            return np.zeros(0, dtype=int)
        nearest, farthest = self.distance_range(self.all_cells(), origin[None])
        inside = (nearest <= (ranges + widths + _TOLERANCE)[:, None]) & (farthest >= (ranges - widths - _TOLERANCE)[:, None])
        return np.count_nonzero(inside, axis=1)
