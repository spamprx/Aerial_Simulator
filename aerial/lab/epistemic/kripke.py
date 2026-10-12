"""A generic S5 Kripke engine: knowledge as truth in every world an agent cannot tell apart.

Two worlds look the same to agent i when its local state is the same in both.  That is an equivalence
relation, so T, 4 and 5 hold by construction.  Formulas are built from ``Atom``, ``Not``, ``And``, ``Or``,
``K(i, phi)``, ``E(group, phi)`` and ``C(group, phi)``:

    K(i, phi)      phi holds in every world i cannot tell from this one
    E(group, phi)  everyone in the group knows phi
    C(group, phi)  phi holds in every world reachable by any chain of the group's indistinguishabilities,
                   which is what "everyone knows that everyone knows ..." comes to

The gate does not run on this engine, since its worlds are far too many to list (see ``worlds.knows_safe``).
This is the reference that the gate's shortcuts are checked against, on instances small enough to enumerate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Tuple


class Formula:
    """Formulas compare by structure, so a repeated subformula is computed once."""


@dataclass(frozen=True)
class Atom(Formula):
    name: str


@dataclass(frozen=True)
class Not(Formula):
    child: Formula


@dataclass(frozen=True, init=False)
class And(Formula):
    children: Tuple[Formula, ...]

    def __init__(self, *children: Formula):
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, init=False)
class Or(Formula):
    children: Tuple[Formula, ...]

    def __init__(self, *children: Formula):
        object.__setattr__(self, "children", children)


@dataclass(frozen=True)
class K(Formula):
    agent: Hashable
    child: Formula


def _group(group: Iterable[Hashable]) -> FrozenSet[Hashable]:
    members = frozenset(group)
    if not members:
        raise ValueError("a group of agents must not be empty")
    return members


@dataclass(frozen=True, init=False)
class E(Formula):
    group: FrozenSet[Hashable]
    child: Formula

    def __init__(self, group: Iterable[Hashable], child: Formula):
        object.__setattr__(self, "group", _group(group))
        object.__setattr__(self, "child", child)


@dataclass(frozen=True, init=False)
class C(Formula):
    group: FrozenSet[Hashable]
    child: Formula

    def __init__(self, group: Iterable[Hashable], child: Formula):
        object.__setattr__(self, "group", _group(group))
        object.__setattr__(self, "child", child)


def implies(premise: Formula, conclusion: Formula) -> Formula:
    return Or(Not(premise), conclusion)


class KripkeModel:
    """``worlds`` must be hashable, and ``valuation`` maps each atom's name to a test on a world."""

    def __init__(self, worlds: Iterable[Hashable], local_state: Callable[[Hashable, Hashable], Hashable],
                 valuation: Dict[str, Callable[[Hashable], bool]]):
        self.worlds = frozenset(worlds)
        self.local_state, self.valuation = local_state, valuation
        self._classes: Dict[Hashable, List[FrozenSet]] = {}
        self._extensions: Dict[Formula, FrozenSet] = {}

    def classes(self, agent: Hashable) -> List[FrozenSet]:
        """The sets of worlds this agent cannot tell apart."""
        if agent not in self._classes:
            grouped: Dict[Hashable, set] = {}
            for world in self.worlds:
                grouped.setdefault(self.local_state(agent, world), set()).add(world)
            self._classes[agent] = [frozenset(members) for members in grouped.values()]
        return self._classes[agent]

    def possible(self, agent: Hashable, world: Hashable) -> FrozenSet:
        """I_i(w): the worlds this agent considers possible at ``world``."""
        self._check(world)
        return next(members for members in self.classes(agent) if world in members)

    def extension(self, phi: Formula) -> FrozenSet:
        """The worlds where ``phi`` holds."""
        if phi not in self._extensions:
            self._extensions[phi] = self._compute(phi)
        return self._extensions[phi]

    def holds(self, phi: Formula, world: Hashable) -> bool:
        self._check(world)
        return world in self.extension(phi)

    def restrict(self, phi: Formula) -> "KripkeModel":
        """The model after ``phi`` is publicly announced: only the worlds where it holds remain."""
        return KripkeModel(self.extension(phi), self.local_state, self.valuation)

    def _check(self, world: Hashable) -> None:
        if world not in self.worlds:
            raise KeyError(f"{world!r} is not a world of this model")

    def _compute(self, phi: Formula) -> FrozenSet:
        if isinstance(phi, Atom):
            if phi.name not in self.valuation:
                raise KeyError(f"unknown atom '{phi.name}'. Available: {sorted(self.valuation)}")
            test = self.valuation[phi.name]
            return frozenset(world for world in self.worlds if test(world))
        if isinstance(phi, Not):
            return self.worlds - self.extension(phi.child)
        if isinstance(phi, And):
            result = self.worlds
            for child in phi.children:
                result = result & self.extension(child)
            return result
        if isinstance(phi, Or):
            return frozenset().union(*(self.extension(child) for child in phi.children))
        if isinstance(phi, K):
            return self._whole_classes(self.classes(phi.agent), self.extension(phi.child))
        if isinstance(phi, E):
            result = self.worlds
            for agent in phi.group:
                result = result & self.extension(K(agent, phi.child))
            return result
        if isinstance(phi, C):
            return self._whole_classes(self._reachable_sets(phi.group), self.extension(phi.child))
        raise TypeError(f"not a formula: {phi!r}")

    @staticmethod
    def _whole_classes(classes: Iterable[FrozenSet], inside: FrozenSet) -> FrozenSet:
        return frozenset().union(*(members for members in classes if members <= inside))

    def _reachable_sets(self, group: FrozenSet) -> List[FrozenSet]:
        """Connected components of the union of the group's relations, by union-find."""
        parent = {world: world for world in self.worlds}

        def find(world):
            while parent[world] != world:
                parent[world] = parent[parent[world]]
                world = parent[world]
            return world

        for agent in group:
            for members in self.classes(agent):
                first, *rest = members
                for world in rest:
                    parent[find(world)] = find(first)
        components: Dict[Hashable, set] = {}
        for world in self.worlds:
            components.setdefault(find(world), set()).add(world)
        return [frozenset(members) for members in components.values()]
