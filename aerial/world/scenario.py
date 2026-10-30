"""Survivors (T1.2) and the scenario generator (T1.10)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.config import Config
from .geometry import CoverageGrid, make_coverage_grid, viewable_mask
from .terrain import Terrain, generate_terrain

SCENARIOS = ("open_field", "district", "dense_rubble", "demo_district")
DEMO_SEED = 7


@dataclass
class Survivor:
    cell: int
    pos: np.ndarray          # (3,) on the ground at the cell centre
    alive: bool
    conscious: bool
    heat_mean: float         # deg C
    p_motion: float


@dataclass
class Scenario:
    name: str
    seed: int
    terrain: Terrain
    grid: CoverageGrid
    viewable: np.ndarray             # (n_cells,) bool ground truth
    coverable: np.ndarray            # (n_cells,) bool: not under a standing roof
    survivors: List[Survivor]
    init_pos: np.ndarray             # (N, 3)
    init_yaw: np.ndarray
    base: np.ndarray                 # (3,)
    liars: np.ndarray                # (N,) bool
    claim_region: np.ndarray         # coverage cell ids claimed by any liar (union, used by the metrics)
    entry_y: float                   # y of the zone's entry line (end of transit)
    extra: dict = field(default_factory=dict)
    claim_regions: list = field(default_factory=list)   # one region per liar, in liar-id order

    @property
    def survivor_cells(self) -> np.ndarray:
        return np.array([s.cell for s in self.survivors], dtype=int)


def make_survivor(cell: int, grid: CoverageGrid, cfg, rng) -> Survivor:
    alive = rng.random() < cfg.world.p_alive
    conscious = bool(alive and rng.random() < 0.6)
    if alive:
        heat, pm = rng.normal(34.0, 1.0), (0.6 if conscious else 0.25)
    else:
        heat, pm = rng.uniform(25.0, 30.0), 0.02
    pos = grid.target[cell].copy()
    pos[2] -= 0.3
    return Survivor(cell=int(cell), pos=pos, alive=bool(alive), conscious=conscious, heat_mean=float(heat),
                    p_motion=pm)


def generate_scenario(cfg: Config, name: Optional[str] = None, seed: Optional[int] = None) -> Scenario:
    name = name or cfg.world.scenario
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}; choose from {SCENARIOS}")
    seed = (DEMO_SEED if name == "demo_district" else cfg.sim.seed) if seed is None else seed
    wcfg = cfg.world
    if name == "open_field":
        wcfg = type(wcfg)(**{**wcfg.__dict__, "p_collapse": 0.0, "building_h": (0.0, 0.0)})
    elif name == "dense_rubble":
        wcfg = type(wcfg)(**{**wcfg.__dict__, "p_collapse": 0.7, "street_min": 8.0, "street_max": 9.0})
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    for _ in range(20):
        terrain = generate_terrain(wcfg, rng)
        if name == "open_field":
            terrain.H[:] = 0.0
        grid = make_coverage_grid(terrain, wcfg.coverage_cell)
        coverable = grid.target[:, 2] - 0.3 < 6.0
        viewable = viewable_mask(terrain, grid, cfg.sensor, wcfg.layers) & coverable
        if viewable.mean() > 0.5:
            break
    N = cfg.N
    # launch base: a row in the strip below the zone, at the lowest layer
    base = np.array([wcfg.arena / 2, (wcfg.arena - wcfg.zone) / 2, 0.0])
    xs = base[0] + (np.arange(N) - (N - 1) / 2) * min(8.0, (wcfg.arena - 8) / max(1, N - 1))
    init_pos = np.stack([xs, np.full(N, base[1]), np.full(N, wcfg.z_min)], axis=1)
    init_yaw = np.full(N, np.pi / 2)
    liars = np.zeros(N, bool)
    n_c = cfg.attack.n_compromised
    if n_c:
        liars[rng.choice(N, size=n_c, replace=False)] = True
    # claim region: a square of cells far from the entry, chosen to contain survivors for the demo
    n = grid.n
    side = n // 2
    a0 = 0 if rng.random() < 0.5 else n - side           # a far corner of the zone
    b0 = n - side
    region = np.array([a * n + b for a in range(a0, a0 + side) for b in range(b0, b0 + side)])
    # a second liar claims the opposite far corner (no extra random draws, so one-liar scenarios are unchanged)
    a1 = n - side if a0 == 0 else 0
    region2 = np.array([a * n + b for a in range(a1, a1 + side) for b in range(b0, b0 + side)])
    claim_regions = [region, region2][:max(1, n_c)]
    # cells deep inside the region (>= 14 m from any edge shared with the rest of the zone) for the demo survivors
    m = int(np.ceil(14.0 / wcfg.coverage_cell))
    lo_a = a0 + (m if a0 > 0 else 0)
    hi_a = a0 + side - (m if a0 == 0 else 0)
    deep = np.array([a * n + b for a in range(lo_a, hi_a) for b in range(b0 + m, n)])
    cand = np.flatnonzero(viewable)
    survivors: List[Survivor] = []
    n_s = wcfg.n_survivors
    if name == "demo_district":
        in_reg = np.intersect1d(cand, deep)
        for c in rng.choice(in_reg, size=min(3, len(in_reg)), replace=False):
            survivors.append(make_survivor(c, grid, cfg, rng))
    taken = {s.cell for s in survivors}
    rest = np.array([c for c in cand if c not in taken])
    for c in rng.choice(rest, size=max(0, min(n_s - len(survivors), len(rest))), replace=False):
        survivors.append(make_survivor(c, grid, cfg, rng))
    return Scenario(name=name, seed=seed, terrain=terrain, grid=grid, viewable=viewable, coverable=coverable,
                    survivors=survivors, init_pos=init_pos, init_yaw=init_yaw, base=base, liars=liars,
                    claim_region=np.unique(np.concatenate(claim_regions)), entry_y=float(terrain.zone_origin[1]),
                    claim_regions=claim_regions)
