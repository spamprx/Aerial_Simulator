"""Tiny name -> class registry so configs can refer to components by string.

Controllers, attacks (and later: gates, specs) all register themselves here, which is
what lets a JSON config say  {"controller": "consensus"}  without any import wiring.
"""
from __future__ import annotations

from typing import Callable, Dict, Type


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def deco(cls: Type) -> Type:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            cls.registry_name = name
            return cls

        return deco

    def create(self, name: str, **params):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](**params)

    def names(self):
        return sorted(self._items)


agent_registry = Registry("agent")
liar_registry = Registry("liar")
aggregator_registry = Registry("aggregator")
