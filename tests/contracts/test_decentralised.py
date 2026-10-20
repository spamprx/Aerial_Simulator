"""Agent i's action must not depend on anything outside its own sensors and radio (A2)."""
import numpy as np
import pytest

from aerial.registry import agent_registry
import aerial.agents.base  # noqa: F401  (registers agents)
from aerial.core.simulator import Simulator


@pytest.mark.parametrize("name", agent_registry.names())
def test_far_agent_does_not_affect_me(cfg, name):
    cfg.comms.R = 20.0
    base = np.array([[10, 10, 9], [15, 10, 9], [110, 110, 9], [20, 10, 9]], float)
    moved = base.copy()
    moved[2] = [100, 115, 12]           # out of range of everyone, before and after
    def actions(p):
        sim = Simulator(cfg, [agent_registry.create(name) for _ in range(4)], p)
        sim.run(steps=5)
        return sim.pos[[0, 1, 3]]
    assert np.allclose(actions(base), actions(moved))
