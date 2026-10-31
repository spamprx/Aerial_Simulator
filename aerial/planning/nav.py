"""Navigation grid, path planning and the dynamic partition (T2.4, T2.5).

The navigation grid has 2 m cells aligned with the coverage grid. Paths use Dijkstra from scipy's csgraph (C speed)
over cells *free under the fleet's standard*; unknown cells are blocked for flying. The partition is a geodesic
Voronoi labelling from multi-source Dijkstra where unknown cells count as passable and only known-blocked cells block
(review finding 9).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def coarsen(fine: np.ndarray, k: int, how=np.all) -> np.ndarray:
    n = fine.shape[0] // k
    return how(fine[: n * k, : n * k].reshape(n, k, n, k), axis=(1, 3))


def grid_graph(passable: np.ndarray, cell: float) -> sparse.csr_matrix:
    n = passable.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, w = [], [], []
    for di, dj in _NB:
        a = passable[max(0, -di): n - max(0, di), max(0, -dj): n - max(0, dj)]
        b = passable[max(0, di): n - max(0, -di) or None, max(0, dj): n - max(0, -dj) or None]
        ok = a & b
        if di and dj:   # no corner cutting
            ok &= passable[max(0, -di): n - max(0, di), max(0, dj): n - max(0, -dj) or None]
            ok &= passable[max(0, di): n - max(0, -di) or None, max(0, -dj): n - max(0, dj)]
        src = idx[max(0, -di): n - max(0, di), max(0, -dj): n - max(0, dj)][ok]
        dst = idx[max(0, di): n - max(0, -di) or None, max(0, dj): n - max(0, -dj) or None][ok]
        rows.append(src); cols.append(dst); w.append(np.full(len(src), cell * np.hypot(di, dj)))
    rows, cols, w = map(np.concatenate, (rows, cols, w))
    return sparse.csr_matrix((w, (rows, cols)), shape=(n * n, n * n))


class NavGrid:
    def __init__(self, passable: np.ndarray, cell: float):
        self.passable = passable
        self.cell = cell
        self.n = passable.shape[0]
        self.G = grid_graph(passable, cell)

    def cell_of(self, xy) -> int:
        i, j = (np.asarray(xy)[:2] // self.cell).astype(int)
        i, j = np.clip(i, 0, self.n - 1), np.clip(j, 0, self.n - 1)
        return int(i * self.n + j)

    def centre(self, c) -> np.ndarray:
        c = np.asarray(c)
        return np.stack([(c // self.n + 0.5) * self.cell, (c % self.n + 0.5) * self.cell], axis=-1)

    def distances(self, start: int) -> Tuple[np.ndarray, np.ndarray]:
        return csgraph.dijkstra(self.G, indices=start, return_predecessors=True)

    @staticmethod
    def path(pred: np.ndarray, goal: int) -> Optional[list]:
        if goal < 0:
            return None
        out = [goal]
        while pred[out[-1]] >= 0:
            out.append(int(pred[out[-1]]))
        return out[::-1]

    def shortcut(self, cells: list) -> list:
        """Keep only turning points whose straight segments stay on passable cells."""
        if len(cells) <= 2:
            return cells
        pts = self.centre(np.array(cells))
        keep = [0]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._segment_ok(pts[i], pts[j]):
                j -= 1
            keep.append(j)
            i = j
        return [cells[k] for k in keep]

    def _segment_ok(self, a, b) -> bool:
        L = np.linalg.norm(b - a)
        s = np.linspace(0, 1, max(2, int(L / (self.cell * 0.4))))
        p = a + (b - a) * s[:, None]
        ij = np.clip((p // self.cell).astype(int), 0, self.n - 1)
        return bool(self.passable[ij[:, 0], ij[:, 1]].all())


def partition_labels(blocked: np.ndarray, cell: float, sources_xy: np.ndarray, source_ids: np.ndarray,
                     cov_ids_to_nav: np.ndarray) -> np.ndarray:
    """Geodesic Voronoi: owner id for every coverage cell (-1 if unreachable from all sources)."""
    nav = NavGrid(~blocked, cell)
    src = np.array([nav.cell_of(p) for p in sources_xy])
    _, _, which = csgraph.dijkstra(nav.G, indices=src, min_only=True, return_predecessors=True)
    # map source nav cell -> agent id robustly (several agents may share a cell)
    lut = {}
    for s, a in zip(src, source_ids):
        lut.setdefault(int(s), int(a))
    owner_nav = np.array([lut.get(int(w), -1) if w >= 0 else -1 for w in which])
    return owner_nav[cov_ids_to_nav]
