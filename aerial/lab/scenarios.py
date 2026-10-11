"""Disaster-response scenarios: where a fleet starts, what is in its way, and what each UAV must sweep.

``generate_scenario(name, seed, **params)`` returns a whole ``SimConfig``, the same one for the same seed:

    the arena         cut into k x k sectors, listed in ``cfg.mission["sectors"]``
    obstacles         collapsed buildings: spheres in ``world.obstacles``, never over a start or a survivor
    survivors         small targets in ``world.targets``, placed in the sectors to sweep
    the launch base   a cluster of start positions in ``init_pos``, clear of everything
    the assignment    which sectors each UAV sweeps, in ``cfg.mission["assignment"]``
    the routes        a lawnmower over those sectors, in ``cfg.mission["agent_waypoints"]``

The named scenarios are ``open_field`` (no obstacles), ``urban`` (a wall of buildings with one corridor through it,
plus scattered buildings) and ``demo_corridor`` (a liar hovering in the corridor every UAV has to take).  The
config is the naive fleet.  ``fleet_overrides`` gives the changes that make it another fleet of D1, and
``no_attack_overrides`` the changes that make it the baseline with nobody lying.

Everything lies in the x-y plane.  In 3-D that plane is at half the arena's height and buildings are centred on it,
so a 3-D scenario is the same kind of map with room above and below.  ``coarse_path`` is the reachability check, and the
router uses it too: a route is a set of straight legs, because a UAV seeking a point behind a wall would stop at the
wall.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SimConfig
from .core.world import Obstacle, World, breadth_first_steps
from .safety import gate_clearance, obstacle_margins

SCENARIOS = ("open_field", "urban", "demo_corridor")
# The seed behind each file in configs/scenarios/, which scripts/make_scenarios.py writes.  The demo's is frozen for T5.4:
# scripts/validate_demo.py shows that it is typical, and T8.1 will assert on it.
FROZEN_SEEDS = {"open_field": 0, "urban": 0, "demo_corridor": 0}

_SALT = 5  # keeps the layout's random numbers apart from the simulator's, which spawn from the same seed
_MAX_SPEED, _MAX_ACCEL, _DT = 5.0, 4.0, 0.05
_EDGE_MARGIN = 3.0  # waypoints stay this far from the arena's walls, outside the geofence's repulsion
_ROUTE_CELL = 1.5  # metres: the router's grid
_SURVIVOR_RADIUS = 0.5
_WALL_RADIUS = 6.0
_WALL_STEP = 1.5  # sphere centres this many radii apart overlap, so nothing fits between them
_CORRIDOR_HALF_LENGTH = _WALL_RADIUS + 4.0  # a UAV is in its lane this far either side of the wall
_SECTOR_SIDE = 20.0  # metres: how big an arena is, by default, for its number of sectors
_DEFAULTS = {"open_field": {"n_agents": 8, "k": 3, "comm_radius": 30.0}, "urban": {"n_agents": 8, "k": 3, "comm_radius": 30.0},
             # Everyone stays in touch to the far corner, or the last UAV through the corridor has nobody to locate the liar with.
             "demo_corridor": {"n_agents": 9, "k": 4, "comm_radius": 60.0}}
_PATIENCE = 1.0  # seconds a gated UAV holds and re-queries before it looks for a way round


# ---- reachability ------------------------------------------------------------------------------
def _free_cells(world: World, cell: float, margin: float) -> Tuple[np.ndarray, float]:
    """(side,) * d bool for each cell whose centre is at least ``margin`` clear of every obstacle, and the cell's side."""
    side = max(int(np.ceil(world.arena_size / cell)), 1)
    size = world.arena_size / side
    axis = (np.arange(side) + 0.5) * size
    centres = np.stack(np.meshgrid(*[axis] * world.dim, indexing="ij"), axis=-1)
    return obstacle_margins(centres, world) >= margin, size


def _join_grid(point: np.ndarray, free: np.ndarray, size: float, world: World) -> Optional[Tuple[int, ...]]:
    """The free cell nearest ``point`` that a straight, clear leg reaches.  The point's own cell may not be free, since
    its centre can be closer to an obstacle than the point is."""
    cells = np.argwhere(free)
    if len(cells) == 0:
        return None
    distance = np.linalg.norm((cells + 0.5) * size - point, axis=1)
    for index in np.argsort(distance, kind="stable")[:8]:
        if world.path_is_clear(point, (cells[index] + 0.5) * size):
            return tuple(cells[index].tolist())
    return None


def _search_from(world: World, start: np.ndarray, cell: float, margin: float):
    """(free cells, each one's side, steps from ``start`` to each), or a distance of None if ``start`` has no way in."""
    free, size = _free_cells(world, cell, margin)
    if margin <= size / 2.0:
        raise ValueError(f"margin {margin} must exceed half a cell, {size / 2.0}, or a leg could clip an obstacle")
    source = _join_grid(start, free, size, world)
    return free, size, None if source is None else breadth_first_steps(free, source)


def coarse_path(world: World, start: Sequence[float], goal: Sequence[float], cell: float = 1.5,
                margin: float = 1.0) -> Optional[np.ndarray]:
    """(M, d) points from ``start`` to ``goal`` with every leg clear of every obstacle, or None if the grid has no way.

    A breadth-first search over cells at least ``margin`` clear.  Neighbouring centres are one cell apart, so a leg
    between them is clear as long as ``margin`` is over half a cell, which this checks.  The search only proves that
    a way exists, so a None can mean the way is narrower than this grid can see.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    free, size, distance = _search_from(world, start, cell, margin)
    target = None if distance is None else _join_grid(goal, free, size, world)
    if target is None or distance[target] < 0:
        return None
    cells = [np.array(target)]
    while distance[tuple(cells[-1])] > 0:
        here = cells[-1]
        for step in np.vstack([np.eye(world.dim, dtype=int), -np.eye(world.dim, dtype=int)]):
            near = here + step
            if ((near >= 0) & (near < free.shape[0])).all() and distance[tuple(near)] == distance[tuple(here)] - 1:
                cells.append(near)
                break
    return np.vstack([start, (np.array(cells[::-1]) + 0.5) * size, goal])


def survivors_reachable(cfg: SimConfig, cell: float = 1.5) -> np.ndarray:
    """(S,) bool: whether each survivor has a collision-free path from a start position, keeping d_safe from obstacles.
    One search from one honest UAV's start answers for all of them, since a way there is a way back."""
    world = World.from_dict(cfg.world, cfg.dim, cfg.arena_size)
    starts = np.array(cfg.init_pos)
    start = starts[[i for i in range(cfg.n_agents) if i not in (cfg.compromised_ids or [])][0]]
    free, size, distance = _search_from(world, start, cell, cfg.d_safe)
    joined = [None if distance is None else _join_grid(np.array(target.center), free, size, world) for target in world.targets]
    return np.array([where is not None and distance[where] >= 0 for where in joined], dtype=bool)


# ---- routes ---------------------------------------------------------------------------------------
def _shortcut(path: np.ndarray, world: World, margin: float) -> List[np.ndarray]:
    """The points after the first, leaving out any that a straight, clear leg can skip."""
    kept, here = [], 0
    while here < len(path) - 1:
        furthest = here + 1
        for candidate in range(len(path) - 1, here + 1, -1):
            if world.path_is_clear(path[here], path[candidate], margin):
                furthest = candidate
                break
        kept.append(path[furthest])
        here = furthest
    return kept


def _legs_to(start: np.ndarray, goal: np.ndarray, world: World, margin: float) -> List[np.ndarray]:
    """The waypoints that take a UAV from ``start`` to ``goal`` on straight, clear legs, ending at ``goal``."""
    if world.path_is_clear(start, goal, margin):
        return [goal]
    path = coarse_path(world, start, goal, _ROUTE_CELL, margin)
    if path is None:
        raise ValueError(f"no way from {start.tolist()} to {goal.tolist()}")
    return _shortcut(path, world, margin)


def lawnmower(low: Sequence[float], high: Sequence[float], r_detect: float, entry: Sequence[float],
              edge_margin: float = 0.0) -> np.ndarray:
    """(M, 2) the ends of each pass over the box, starting from the corner nearest ``entry``.

    A pass stops r / 2 short of the box's ends, and the outer passes run sqrt(3) r / 2 in from its sides, which puts a
    corner exactly r from the nearest pass.  Passes sqrt(3) r apart do the same for the points between two of them, out
    at the end where no leg joins them, so a sensor of radius r sees all of the box.  ``edge_margin`` keeps the passes
    that far inside the box's edges as well, at the cost of what lies beyond it.
    """
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    half_gap = np.sqrt(3.0) * r_detect / 2.0
    inset = np.maximum([r_detect / 2.0, half_gap], edge_margin)
    near, far = low + inset, high - inset
    near, far = np.where(near <= far, near, (low + high) / 2.0), np.where(near <= far, far, (low + high) / 2.0)
    passes = int(np.ceil((far[1] - near[1]) / (2.0 * half_gap) - 1e-9)) + 1
    rows = np.linspace(near[1], far[1], passes) if passes > 1 else np.array([near[1]])
    if abs(entry[1] - rows[-1]) < abs(entry[1] - rows[0]):
        rows = rows[::-1]
    ends = (near[0], far[0]) if abs(entry[0] - near[0]) <= abs(entry[0] - far[0]) else (far[0], near[0])
    points = []
    for index, y in enumerate(rows):
        first, second = ends if index % 2 == 0 else ends[::-1]
        points += [[first, y], [second, y]]
    return np.array(points)


def _clear_end(end: np.ndarray, other: np.ndarray, world: World, margin: float) -> Optional[np.ndarray]:
    """The point nearest ``end`` on the way to ``other`` that is clear of every building, or None if the whole pass is covered."""
    for fraction in np.linspace(0.0, 1.0, 41):
        point = end + fraction * (other - end)
        if world.is_clear(point, margin):
            return point
    return None


def _sweep_route(start: np.ndarray, boxes: Sequence[np.ndarray], world: World, margin: float, r_detect: float,
                 first_legs: Sequence[np.ndarray] = ()) -> np.ndarray:
    """(M, 2) a route from ``start`` over each box in turn.  A pass with an end too close to a building stops short of it."""
    route, here = [], start
    for point in first_legs:
        route += _legs_to(here, point, world, margin)
        here = point
    for box in boxes:
        passes = lawnmower(box[0], box[1], r_detect, here, _EDGE_MARGIN).reshape(-1, 2, 2)
        for first, second in passes:
            ends = _clear_end(first, second, world, margin), _clear_end(second, first, world, margin)
            if ends[0] is None:
                continue
            for point in ends:
                if np.linalg.norm(point - here) > 1e-9:
                    route += _legs_to(here, point, world, margin)
                    here = point
    return np.array(route).reshape(-1, 2)


def route_length(start: np.ndarray, route: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([start, np.reshape(route, (-1, len(start)))]), axis=0), axis=1).sum())


# ---- layout ---------------------------------------------------------------------------------------
def _sector_boxes(arena_size: float, k: int) -> np.ndarray:
    """(k * k, 2, 2) each sector's low and high corner, column by column."""
    size = arena_size / k
    return np.array([[[a * size, b * size], [(a + 1) * size, (b + 1) * size]] for a in range(k) for b in range(k)])


def _launch_cluster(centre: np.ndarray, n: int, rows: int, spacing: float, rng: np.random.Generator, arena_size: float) -> np.ndarray:
    """(n, 2) a jittered grid around ``centre``, nudged inside the arena, ``spacing`` apart less the jitter.  Rows fill
    from the middle outwards, so that a short row is an outer one."""
    columns = int(np.ceil(n / rows))
    outwards = sorted(range(rows), key=lambda row: abs(row - (rows - 1) / 2.0))
    spots = np.array([[a, b] for a in range(columns) for b in outwards], dtype=float)[:n]
    spots[:, 1] -= (rows - 1) / 2.0
    spots[:, 0] -= (columns - 1) / 2.0
    spots *= spacing
    width = np.ptp(spots, axis=0) + 2.0 * _EDGE_MARGIN
    if np.any(width > arena_size):
        raise ValueError(f"{n} UAVs {spacing} m apart do not fit a launch base in an arena of {arena_size} m")
    low = _EDGE_MARGIN - spots.min(axis=0)
    high = arena_size - _EDGE_MARGIN - spots.max(axis=0)
    return np.clip(centre, low, high) + spots + rng.uniform(-0.25, 0.25, size=spots.shape)


def _wall(x: float, gap_centre: float, gap: float, arena_size: float) -> List[Tuple[np.ndarray, float]]:
    """Spheres from one side of the arena to the other at this x, leaving a gap ``gap`` wide around ``gap_centre``."""
    step = _WALL_STEP * _WALL_RADIUS
    spheres = []
    for start, sign, end in ((gap_centre - gap / 2.0 - _WALL_RADIUS, -1.0, 0.0),
                             (gap_centre + gap / 2.0 + _WALL_RADIUS, 1.0, arena_size)):
        y = start
        while (y - end) * sign < 0.0:
            spheres.append((np.array([x, y]), _WALL_RADIUS))
            y += sign * step
        spheres.append((np.array([x, end]), _WALL_RADIUS))
    return spheres


def _scatter_buildings(count: int, radius_range: Tuple[float, float], gap: float, keep_clear: np.ndarray, clear_by: float,
                       existing: List[Tuple[np.ndarray, float]], arena_size: float, rng: np.random.Generator):
    """Up to ``count`` more spheres, ``gap`` from each other and ``clear_by`` from each point of ``keep_clear``.  One placed
    badly can leave no room for the next, so a layout that gets stuck starts again, and the fullest one is kept."""
    best = list(existing)
    for _layout in range(100):
        buildings = list(existing)
        for _attempt in range(200):
            if len(buildings) == len(existing) + count:
                return buildings
            radius = float(rng.uniform(*radius_range))
            centre = rng.uniform(radius + _EDGE_MARGIN, arena_size - radius - _EDGE_MARGIN, size=2)
            near_point = len(keep_clear) and np.linalg.norm(keep_clear - centre, axis=1).min() < radius + clear_by
            near_building = any(np.linalg.norm(other - centre) < radius + r + gap for other, r in buildings)
            if not (near_point or near_building):
                buildings.append((centre, radius))
        if len(buildings) > len(best):
            best = buildings
    return best


def _distance_to_routes(point: np.ndarray, legs_from: np.ndarray, legs_to: np.ndarray) -> float:
    """How far ``point`` is from the nearest of these straight legs."""
    along = legs_to - legs_from
    fraction = np.clip(((point - legs_from) * along).sum(axis=1) / np.maximum((along ** 2).sum(axis=1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(legs_from + fraction[:, None] * along - point, axis=1).min())


def _place_survivors(count: int, boxes: np.ndarray, buildings_world: World, margin: float, routes: Sequence[np.ndarray],
                     starts: np.ndarray, r_detect: float, rng: np.random.Generator) -> np.ndarray:
    """(count, 2) one survivor per box in turn, clear of the buildings by ``margin``, and where a route passes close
    enough to find it: a UAV that flies its whole route should find everyone."""
    legs_from = np.vstack([np.vstack([starts[i], route])[:-1] for i, route in enumerate(routes) if len(route)])
    legs_to = np.vstack([route for route in routes if len(route)])
    order = rng.permutation(len(boxes))
    survivors = []
    for index in range(count):
        low, high = boxes[order[index % len(boxes)]]
        for _attempt in range(500):
            point = rng.uniform(np.maximum(low, _EDGE_MARGIN), np.minimum(high, buildings_world.arena_size - _EDGE_MARGIN))
            if buildings_world.is_clear(point, margin) and _distance_to_routes(point, legs_from, legs_to) < 0.9 * r_detect:
                survivors.append(point)
                break
        else:
            raise ValueError(f"no room for a survivor in sector {low.tolist()} to {high.tolist()}")
    return np.array(survivors).reshape(-1, 2)


def _assign_sectors(starts: np.ndarray, boxes: np.ndarray, sweepers: Sequence[int], spacing: float) -> Dict[int, List[int]]:
    """Sector indices per UAV.  Both are taken from the bottom row up and, in a row, from the furthest ahead.  So routes
    do not cross: each row of the base goes to a row of sectors, and in a row the UAV in front has the furthest to go."""
    if len(boxes) < len(sweepers):
        raise ValueError(f"{len(sweepers)} UAVs need at least that many sectors to sweep, and there are {len(boxes)}")
    centres = boxes.mean(axis=1)
    sectors = sorted(range(len(boxes)), key=lambda b: (centres[b, 1], -centres[b, 0]))
    bottom = float(starts[list(sweepers), 1].min())
    in_rows = sorted(sweepers, key=lambda i: (round((float(starts[i, 1]) - bottom) / spacing), -float(starts[i, 0])))
    assignment = {}
    for rank, i in enumerate(in_rows):
        first, last = rank * len(sectors) // len(in_rows), (rank + 1) * len(sectors) // len(in_rows)
        assignment[i] = [int(index) for index in sectors[first:last]]
    return assignment


def _in_plane(points, dim: int, height: float) -> List[List[float]]:
    """Points rounded for a readable JSON file, in the flight plane if the arena is 3-D."""
    points = np.round(np.asarray(points, dtype=float).reshape(-1, 2), 3)
    if dim == 3:
        points = np.hstack([points, np.full((len(points), 1), height)])
    return points.tolist()


def _spheres_dict(items, dim: int, height: float, name: str) -> List[Dict]:
    return [{"center": _in_plane(centre, dim, height)[0], "radius": round(float(radius), 3), "name": name}
            for centre, radius in items]


def _clearances(d_safe: float, dim: int) -> Tuple[float, float]:
    """(r_sep, launch spacing).  Neighbours inside the gate's clearance hold each other, and only separation can part
    them, so it has to reach beyond that clearance.  The launch base stands clear of it too, so that nobody starts held."""
    clearance = gate_clearance(d_safe, d_safe, dim, _MAX_SPEED, 5.0 * _DT, 0.2)
    r_sep = round(float(clearance) + 0.85, 2)
    return r_sep, r_sep + 1.0


def generate_scenario(name: str, seed: int = 0, n_agents: Optional[int] = None, dim: int = 2, arena_size: Optional[float] = None,
                      k: Optional[int] = None, n_buildings: Optional[int] = None, n_survivors: Optional[int] = None,
                      r_detect: float = 4.0, gap: Optional[float] = None, d_safe: float = 1.0, comm_radius: Optional[float] = None,
                      drop_prob: float = 0.1, v_ref: float = 2.5, t_final: Optional[float] = None,
                      offset_mag: float = 8.0, n_liars: int = 1) -> SimConfig:
    """A whole, valid config for this scenario, the same one every time for the same arguments.

    ``k`` is how many sectors along each side, which is also how many rows the launch base has and how many lanes the
    corridor has.  ``gap`` is the corridor's width, by default what those lanes need.  ``n_buildings`` is how many
    ``urban`` scatters: 3 if they fit by default, and exactly that many or an error if asked for.  ``r_detect`` is the
    survivor sensor's radius, and ``offset_mag`` how far ``demo_corridor``'s liar reports itself from where it is.
    ``n_liars`` is how many liars ``demo_corridor`` has, on its outer lanes, and also its F.  A second one is added to the
    fleet, and the rest of the map stays exactly as it was.
    ``t_final`` defaults to what the longest route needs at ``v_ref``, with room to spare for a fleet that hesitates.
    Other defaults depend on the scenario, in ``_DEFAULTS``.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Available: {list(SCENARIOS)}")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if name != "demo_corridor" and n_liars != 1:
        raise ValueError(f"only demo_corridor has liars, and '{name}' does not")
    if n_liars not in (1, 2):
        raise ValueError("demo_corridor has 1 or 2 liars: F is at most 2")
    defaults = _DEFAULTS[name]
    n_agents = defaults["n_agents"] + n_liars - 1 if n_agents is None else n_agents
    n_liars = n_liars if name == "demo_corridor" else 0
    if k is None:  # enough for a sector each, and the demo sweeps only the far half
        k = max(defaults["k"], int(np.ceil(np.sqrt(n_agents))))
        while name == "demo_corridor" and k * (k // 2) < n_agents - n_liars:
            k += 1
    arena_size = _SECTOR_SIDE * k if arena_size is None else arena_size
    comm_radius = defaults["comm_radius"] if comm_radius is None else comm_radius
    if name == "open_field" and n_buildings:
        raise ValueError("an open field has no buildings")
    rng = np.random.default_rng([_SALT, seed])
    height, middle = arena_size / 2.0, arena_size / 2.0
    route_margin = d_safe + 1.5
    r_sep, spacing = _clearances(d_safe, dim)
    lane_offsets = (np.arange(k) - (k - 1) / 2.0) * spacing

    gap_centre = middle + float(rng.uniform(-2.0, 2.0))
    base = np.array([arena_size / 6.0, gap_centre])
    buildings: List[Tuple[np.ndarray, float]] = []
    corridor = None  # (the wall's x, the x a UAV enters at, the x it leaves at)
    honest_count = n_agents - n_liars
    if name != "open_field":
        needed = 2.0 * (lane_offsets.max() + route_margin + 1.0)
        gap = needed if gap is None else gap
        if gap < needed:
            raise ValueError(f"a corridor {gap} m wide has no room for {k} lanes {spacing} m apart: it needs {needed}")
        # The demo's wall stands so that a UAV leaves the corridor just short of the first sector to sweep.
        wall_x = middle + 3.0 - _CORRIDOR_HALF_LENGTH if name == "demo_corridor" else arena_size * 2.0 / 3.0
        buildings = _wall(wall_x, gap_centre, gap, arena_size)
        corridor = (wall_x, wall_x - _CORRIDOR_HALF_LENGTH, wall_x + _CORRIDOR_HALF_LENGTH)
        # All of the base stands short of the corridor, or a UAV would start by flying back into those behind it.
        columns = np.ceil(honest_count / k)
        base[0] = min(base[0], corridor[1] - _EDGE_MARGIN - spacing * (columns - 1) / 2.0)

    starts = _launch_cluster(base, honest_count, k, spacing, rng, arena_size)
    if name == "demo_corridor":
        # On an outer lane, whose UAVs have the longest way to go, so that what the liar costs them shows in the fleet's time.
        liars = [[wall_x + float(rng.uniform(-1.0, 1.0)), gap_centre + lane_offsets[-1] + float(rng.uniform(-0.5, 0.5))]]
        if n_liars == 2:  # with numbers of its own, so that the rest of the map does not change
            second = np.random.default_rng([_SALT, seed, 2])
            liars.append([wall_x + float(second.uniform(-1.0, 1.0)), gap_centre + lane_offsets[0] + float(second.uniform(-0.5, 0.5))])
        starts = np.vstack([starts, liars])
    if name == "urban":
        count = 3 if n_buildings is None else n_buildings
        lane_ends = [[x, gap_centre + offset] for x in corridor[1:] for offset in lane_offsets]
        wall_only = len(buildings)
        buildings = _scatter_buildings(count, (2.0, 4.0), 2.0 * route_margin + r_sep, np.vstack([starts, lane_ends]),
                                       r_sep + route_margin, buildings, arena_size, rng)
        if n_buildings is not None and len(buildings) < wall_only + n_buildings:
            raise ValueError(f"no room for {n_buildings} scattered buildings in an arena of {arena_size} m")
    plan_world = World(2, arena_size, tuple(Obstacle(tuple(centre.tolist()), radius) for centre, radius in buildings))

    all_boxes = _sector_boxes(arena_size, k)
    # In the demo, every sector left to sweep lies through the corridor.
    to_sweep = np.flatnonzero(all_boxes[:, 0, 0] >= middle) if name == "demo_corridor" else np.arange(len(all_boxes))
    sweepers = list(range(honest_count))
    assignment = _assign_sectors(starts, all_boxes[to_sweep], sweepers, spacing)
    assignment = {i: [int(to_sweep[index]) for index in chosen] for i, chosen in assignment.items()}

    # A lane for each row of the base, so that a lane's UAVs reach it in turn.  With one shared point they would reach it
    # together and hold each other off it for ever, since separation forbids them all from being there.
    lanes = {}
    if corridor is not None:
        for i in sweepers:
            if all_boxes[assignment[i]].mean(axis=(0, 1))[0] > corridor[0]:
                y = gap_centre + lane_offsets[np.argmin(np.abs(gap_centre + lane_offsets - starts[i, 1]))]
                lanes[i] = (np.array([corridor[1], y]), np.array([corridor[2], y]))

    routes = []
    for i in sweepers:
        boxes = all_boxes[assignment[i]]
        entry = lanes[i][1] if i in lanes else starts[i]
        boxes = boxes[np.argsort(np.linalg.norm(boxes.mean(axis=1) - entry, axis=1), kind="stable")]
        routes.append(_sweep_route(starts[i], boxes, plan_world, route_margin, r_detect, lanes.get(i, ())))
    for i in range(honest_count, n_agents):  # a liar never flies a route: it only hovers and lies
        assignment[i] = []
        routes.append(np.empty((0, 2)))

    count = 2 * len(to_sweep) if n_survivors is None else n_survivors
    survivors = _place_survivors(count, all_boxes[to_sweep], plan_world, route_margin, routes[:honest_count], starts, r_detect, rng)

    longest = max(route_length(starts[i], routes[i]) for i in range(n_agents))
    duration = float(np.ceil((1.5 * longest / v_ref + 30.0) / 5.0) * 5.0) if t_final is None else t_final
    settings = dict(
        n_agents=n_agents, dim=dim, dt=_DT, t_final=duration, arena_size=arena_size,
        init_pos=_in_plane(starts, dim, height),
        world={"obstacles": _spheres_dict(buildings, dim, height, "building"),
               "targets": _spheres_dict([(point, _SURVIVOR_RADIUS) for point in survivors], dim, height, "survivor")},
        comm_radius=comm_radius, drop_prob=drop_prob, max_speed=_MAX_SPEED, max_accel=_MAX_ACCEL,
        sensing={"ranging": True, "sigma": 0.05},
        # Counting worlds scans every cell, which in 3-D is what a run spends nearly all of its time on: 631 s against 9 s.
        gate={"share_ranges": True, "patience": _PATIENCE, **({"size_every": 0} if dim == 3 else {})},
        controller="mission",
        controller_params=dict(behaviour="boids", filter="wmsr", r_sep=r_sep, v_ref=v_ref, speed_limit=v_ref + 0.5),
        mission=dict(agent_waypoints=[_in_plane(route, dim, height) for route in routes],
                     sectors=np.round(all_boxes, 3).tolist(), assignment=[assignment[i] for i in range(n_agents)],
                     r_detect=r_detect),
        d_safe=d_safe, seed=seed, record_lambda2=False,
    )
    if n_liars:
        settings.update(F=n_liars, compromised_ids=list(range(honest_count, n_agents)), attack="covert", adversary_motion="hover",
                        attack_params=dict(offset_mag=offset_mag, direction="away_from_path"))
    return SimConfig(**settings).validate()


# ---- the fleets of D1 -----------------------------------------------------------------------------
def fleet_overrides(cfg: SimConfig, fleet: str) -> Dict:
    """The overrides that turn a scenario into one of D1's fleets.  Only the controller changes, which is what makes
    them comparable: ``naive`` flies W-MSR with the gate off, ``gated`` adds the gate, ``oracle`` tells the gate who is
    faulty, and ``unfiltered`` drops W-MSR."""
    base = cfg.controller_params.get("inner_params", cfg.controller_params)
    if fleet == "naive":
        return {"controller": "mission", "controller_params": {**base, "filter": "wmsr"}}
    if fleet == "unfiltered":
        return {"controller": "mission", "controller_params": {**base, "filter": "mean"}}
    if fleet in ("gated", "oracle"):
        return {"controller": fleet, "controller_params": {"inner": "mission", "inner_params": {**base, "filter": "wmsr"}}}
    raise KeyError(f"unknown fleet '{fleet}'. Available: ['naive', 'gated', 'oracle', 'unfiltered']")


def no_attack_overrides(cfg: SimConfig) -> Dict:
    """The overrides for a baseline run of this scenario: nobody lying, and nobody assumed to be (T5.3).

    A UAV that exists only to lie, as in ``demo_corridor``, has no mission of its own, and it is left out.  Leaving it
    hovering honestly in the corridor would be a different experiment: an obstacle that says where it is.
    """
    overrides = {"F": 0, "n_compromised": 0, "compromised_ids": None, "attack": "none", "attack_params": {}, "adversary_motion": "same"}
    remove = set(cfg.compromised_ids or [])
    if cfg.adversary_motion == "hover" and remove:
        keep = [i for i in range(cfg.n_agents) if i not in remove]
        mission = dict(cfg.mission)
        for key in ("agent_waypoints", "assignment"):
            mission[key] = [cfg.mission[key][i] for i in keep]
        overrides.update(n_agents=len(keep), init_pos=[cfg.init_pos[i] for i in keep], mission=mission)
    return overrides
