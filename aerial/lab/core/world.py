"""World geometry used by scenarios, safety checks, and mission monitors."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def breadth_first_steps(free: np.ndarray, source: Tuple[int, ...]) -> np.ndarray:
    """Steps from ``source`` to each free cell of a boolean grid, moving one cell along one axis at a time; -1 if out of reach."""
    distance = np.full(free.shape, -1, dtype=int)
    distance[source] = 0
    frontier = np.zeros_like(free)
    frontier[source] = True
    steps = 0
    while frontier.any():
        steps += 1
        reached = np.zeros_like(free)
        for axis in range(free.ndim):
            low, high = [slice(None)] * free.ndim, [slice(None)] * free.ndim
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            reached[tuple(low)] |= frontier[tuple(high)]
            reached[tuple(high)] |= frontier[tuple(low)]
        frontier = reached & free & (distance < 0)
        distance[frontier] = steps
    return distance


def _vector(value: Sequence[float], dim: int, name: str) -> Tuple[float, ...]:
    result = tuple(float(x) for x in value)
    if len(result) != dim:
        raise ValueError(f"{name} must have exactly {dim} coordinates")
    return result


@dataclass(frozen=True)
class Obstacle:
    """A closed spherical obstacle (a disk when ``dim == 2``)."""

    center: Tuple[float, ...]
    radius: float
    name: str = "obstacle"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("obstacle radius must be positive")

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        return float(np.linalg.norm(np.asarray(point) - self.center)) <= self.radius + margin

    def to_dict(self) -> Dict[str, Any]:
        return {"center": list(self.center), "radius": self.radius, "name": self.name}


@dataclass(frozen=True)
class Target:
    """A spherical target region (a disk when ``dim == 2``)."""

    center: Tuple[float, ...]
    radius: float
    name: str = "target"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("target radius must be positive")

    def contains(self, point: Sequence[float]) -> bool:
        return float(np.linalg.norm(np.asarray(point) - self.center)) <= self.radius

    def to_dict(self) -> Dict[str, Any]:
        return {"center": list(self.center), "radius": self.radius, "name": self.name}


@dataclass(frozen=True)
class World:
    """Static geometry for one simulation run."""

    dim: int
    arena_size: float
    obstacles: Tuple[Obstacle, ...] = ()
    targets: Tuple[Target, ...] = ()

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("world dimension must be 2 or 3")
        if self.arena_size <= 0:
            raise ValueError("arena_size must be positive")
        for item in (*self.obstacles, *self.targets):
            if len(item.center) != self.dim:
                raise ValueError(f"{type(item).__name__} center must have {self.dim} coordinates")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], dim: int, arena_size: float) -> "World":
        data = data or {}
        unknown = set(data) - {"obstacles", "targets"}
        if unknown:
            raise KeyError(f"unknown world keys: {sorted(unknown)}")
        obstacles = tuple(
            Obstacle(_vector(item["center"], dim, "obstacle center"),
                     float(item["radius"]), item.get("name", "obstacle"))
            for item in data.get("obstacles", [])
        )
        targets = tuple(
            Target(_vector(item["center"], dim, "target center"),
                   float(item["radius"]), item.get("name", "target"))
            for item in data.get("targets", [])
        )
        return cls(dim, float(arena_size), obstacles, targets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "targets": [target.to_dict() for target in self.targets],
        }

    @cached_property
    def obstacle_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """(O, d) centres and (O,) radii, for working on all the obstacles at once."""
        centres = np.array([obstacle.center for obstacle in self.obstacles], dtype=float).reshape(-1, self.dim)
        return centres, np.array([obstacle.radius for obstacle in self.obstacles], dtype=float)

    def is_clear(self, point: Sequence[float], margin: float = 0.0) -> bool:
        """Return whether a point is outside every obstacle."""
        return not any(obstacle.contains(point, margin) for obstacle in self.obstacles)

    def path_is_clear(
        self, start: Sequence[float], end: Sequence[float], margin: float = 0.0,
        samples: int = 0,
    ) -> bool:
        """Return whether a straight path avoids obstacles.

        ``samples == 0`` uses the exact closest-point test for spherical obstacles.
        A positive value additionally checks evenly spaced points, useful for a
        conservative discretised sensor model.
        """
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if a.shape != (self.dim,) or b.shape != (self.dim,):
            raise ValueError(f"path endpoints must have shape ({self.dim},)")
        direction = b - a
        length_sq = float(np.dot(direction, direction))
        for obstacle in self.obstacles:
            if length_sq == 0:
                closest = a
            else:
                fraction = np.clip(np.dot(np.asarray(obstacle.center) - a, direction) / length_sq, 0.0, 1.0)
                closest = a + fraction * direction
            if obstacle.contains(closest, margin):
                return False
        if samples > 0:
            for point in np.linspace(a, b, samples + 2)[1:-1]:
                if not self.is_clear(point, margin):
                    return False
        return True

    def clear_ahead(
        self, position: Sequence[float], direction: Sequence[float],
        lookahead: float, margin: float = 0.0,
    ) -> bool:
        """Return whether the ray segment ahead of ``position`` is obstacle-free."""
        if lookahead < 0:
            raise ValueError("lookahead must be non-negative")
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        end = np.asarray(position, dtype=float) if norm == 0 else np.asarray(position) + vector / norm * lookahead
        return self.path_is_clear(position, end, margin)

    def target_contains(self, point: Sequence[float]) -> bool:
        return any(target.contains(point) for target in self.targets)
