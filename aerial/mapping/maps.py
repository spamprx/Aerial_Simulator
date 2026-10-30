"""Per-drone maps with provenance (T2.1–T2.3).

Terrain (A18): own height intervals on the 1 m grid. A ray passing over a cell proves H < z there (lowers ``hi``); a
hit proves H >= z (raises ``lo``); flying through a column proves H < own altitude. For every cruise layer a cell is
*free* if hi + clearance <= layer. Others' free claims arrive with a reporter bitmask, so the fleet's trust standard
decides whether to use them.

Coverage: own viewed cells, others' CLEARED claims (bitmask), survivor sightings (bitmask), abandoned cells.
"""
from __future__ import annotations

import numpy as np


class TerrainMap:
    def __init__(self, arena: float, cell: float, layers, clearance: float):
        self.n = int(round(arena / cell))
        self.cell = cell
        self.layers = np.asarray(layers, float)
        self.clear = clearance
        self.hi = np.full((self.n, self.n), np.inf)
        self.lo = np.zeros((self.n, self.n))
        self.rep_free = np.zeros((len(self.layers), self.n, self.n), dtype=np.uint64)
        self._sent_free = np.zeros((len(self.layers), self.n, self.n), bool)
        self.rep_wall = np.zeros((self.n, self.n), dtype=np.uint64)   # reporters of "H >= 6 m here"
        self._sent_wall = np.zeros((self.n, self.n), bool)

    def idx(self, xy: np.ndarray):
        ij = np.floor(np.asarray(xy)[..., :2] / self.cell).astype(int)
        ok = np.all((ij >= 0) & (ij < self.n), axis=-1)
        return ij[ok], ok

    def mark_free_rect(self, x0, y0, x1, y1, z=0.0):
        i0, j0 = int(max(0, x0 // self.cell)), int(max(0, y0 // self.cell))
        i1, j1 = int(min(self.n, np.ceil(x1 / self.cell))), int(min(self.n, np.ceil(y1 / self.cell)))
        self.hi[i0:i1, j0:j1] = np.minimum(self.hi[i0:i1, j0:j1], z)

    def observe(self, over: np.ndarray, hits: np.ndarray, own_pos: np.ndarray) -> None:
        if len(over):
            ij, ok = self.idx(over)
            np.minimum.at(self.hi, (ij[:, 0], ij[:, 1]), over[ok, 2])
        if len(hits):
            ij, ok = self.idx(hits)
            np.maximum.at(self.lo, (ij[:, 0], ij[:, 1]), hits[ok, 2])
        ij, ok = self.idx(own_pos[None])
        if ok.all():
            self.hi[ij[0, 0], ij[0, 1]] = min(self.hi[ij[0, 0], ij[0, 1]], own_pos[2] - 0.5)

    def own_free(self) -> np.ndarray:
        """(layers, n, n): own evidence that the layer clears the terrain."""
        return self.hi[None] + self.clear <= self.layers[:, None, None]

    def known_blocked(self) -> np.ndarray:
        return self.lo[None] + self.clear > self.layers[:, None, None]

    def add_report(self, reporter: int, layer: int, cells: np.ndarray) -> None:
        cells = np.asarray(cells, dtype=int)
        self.rep_free[layer].ravel()[cells] |= np.uint64(1 << reporter)

    def add_wall_report(self, reporter: int, cells: np.ndarray) -> None:
        self.rep_wall.ravel()[np.asarray(cells, dtype=int)] |= np.uint64(1 << reporter)

    def new_wall_cells(self) -> np.ndarray:
        own = self.lo >= 6.0
        new = own & ~self._sent_wall
        self._sent_wall |= own
        return np.flatnonzero(new)

    def walls(self, standard) -> np.ndarray:
        return standard.holds((self.lo >= 6.0).ravel(), self.rep_wall.ravel()).reshape(self.lo.shape)

    def new_free_cells(self):
        """Cells this drone newly knows free (for TERRAIN events): list of (layer, flat ids)."""
        own = self.own_free()
        new = own & ~self._sent_free
        self._sent_free |= own
        return [(l, np.flatnonzero(new[l])) for l in range(len(self.layers)) if new[l].any()]

    def free(self, standard, layer=None) -> np.ndarray:
        """Free cells under ``standard``: all layers (L, n, n), or one layer (n, n) if ``layer`` is given."""
        layers = range(len(self.layers)) if layer is None else [layer]
        out = []
        for l in layers:
            z = self.layers[l]
            own = self.hi + self.clear <= z
            f = standard.holds(own.ravel(), self.rep_free[l].ravel()).reshape(own.shape)
            out.append(f & ~(self.lo + self.clear > z))
        return np.stack(out) if layer is None else out[0]


class CoverageMap:
    def __init__(self, origin: np.ndarray, size: float, zone: float):
        self.origin = np.asarray(origin, float)
        self.size = size
        self.n = int(round(zone / size))
        a, b = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="ij")
        self.centres = np.stack([self.origin[0] + (a.ravel() + 0.5) * size,
                                 self.origin[1] + (b.ravel() + 0.5) * size], axis=1)
        nc = self.n * self.n
        self.own = np.zeros(nc, bool)
        self.cleared_rep = np.zeros(nc, dtype=np.uint64)
        self.survivor_rep = np.zeros(nc, dtype=np.uint64)
        self.survivor_own = np.zeros(nc, bool)
        self.abandoned = np.zeros(nc, bool)
        self.roof = np.zeros(nc, bool)          # known to be under a standing roof (not coverable)
        self.attempts = np.zeros(nc, int)

    @property
    def n_cells(self) -> int:
        return self.n * self.n

    def cell_of(self, xy) -> np.ndarray:
        ab = np.floor((np.asarray(xy)[..., :2] - self.origin) / self.size).astype(int)
        ok = np.all((ab >= 0) & (ab < self.n), axis=-1)
        return np.where(ok, ab[..., 0] * self.n + ab[..., 1], -1)

    def cleared(self, standard) -> np.ndarray:
        """Cells done under ``standard``: seen empty (own or claims) or a survivor was seen there."""
        own = self.own | self.survivor_own
        rep = self.cleared_rep | self.survivor_rep
        return standard.holds(own, rep)

    def done(self, standard) -> np.ndarray:
        return self.cleared(standard) | self.abandoned | self.roof

    def update_roofs(self, terrain: TerrainMap, standard=None) -> None:
        """Mark coverage cells whose terrain is known to rise above 6 m (standing roofs)."""
        k = int(round(self.size / terrain.cell))
        i0 = np.floor(self.centres[:, 0] / terrain.cell).astype(int) - k // 2
        j0 = np.floor(self.centres[:, 1] / terrain.cell).astype(int) - k // 2
        wall = terrain.lo >= 6.0 if standard is None else terrain.walls(standard)
        hit = np.zeros(self.n_cells, bool)
        for di in range(k):
            for dj in range(k):
                hit |= wall[np.clip(i0 + di, 0, terrain.n - 1), np.clip(j0 + dj, 0, terrain.n - 1)]
        roof = hit.reshape(self.n, self.n)
        # interiors enclosed by known walls cannot be seen from any pose in the band
        from scipy import ndimage
        self.roof |= ndimage.binary_fill_holes(roof).ravel()
