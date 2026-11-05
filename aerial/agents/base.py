"""DroneAgent interface (T0.5) and trivial agents for tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..core.types import Action, AgentObservation, ClaimedState, DroneState, StaticInfo
from ..registry import agent_registry


@dataclass
class AgentContext:
    static: StaticInfo
    cfg: Any                  # Config
    rng: np.random.Generator  # this agent's private ctrl stream


class DroneAgent(ABC):
    """One drone. It holds private state and sees only its own observation (A2)."""

    is_liar: bool = False

    def reset(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.i = ctx.static.i

    def beacon(self, own: DroneState, k: int) -> ClaimedState:
        """The claim broadcast this tick. Honest: the truth."""
        return ClaimedState(own.pos.copy(), own.vel.copy(), own.yaw)

    @abstractmethod
    def step(self, obs: AgentObservation) -> Action: ...

    def diagnostics(self) -> Dict[str, Any]:
        return {}


@agent_registry.register("zero")
class ZeroAgent(DroneAgent):
    def step(self, obs: AgentObservation) -> Action:
        return Action(u=np.zeros(3))


@agent_registry.register("waypoint")
class WaypointAgent(DroneAgent):
    """Toy PD agent that flies to a fixed point; used by simulator tests."""

    def __init__(self, target=(50.0, 50.0, 9.0), k_p: float = 1.0, k_v: float = 1.8):
        self.target = np.asarray(target, float)
        self.k_p, self.k_v = k_p, k_v

    def step(self, obs: AgentObservation) -> Action:
        return Action(u=self.k_p * (self.target - obs.own.pos) - self.k_v * obs.own.vel)
