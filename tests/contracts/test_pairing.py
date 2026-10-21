import numpy as np

from aerial.agents.base import WaypointAgent, ZeroAgent
from aerial.core.simulator import Simulator
from tests.conftest import line_positions


def _sig(cfg, agent_cls):
    sim = Simulator(cfg, [agent_cls() for _ in range(cfg.N)], line_positions(cfg.N))
    sim.run()
    return sim.rng.env_signature()


def test_env_draws_identical_across_controllers(cfg):
    assert _sig(cfg, ZeroAgent) == _sig(cfg, WaypointAgent)


def test_requery_does_not_change_draws(cfg):
    class Requerier(ZeroAgent):
        def step(self, obs):
            a = super().step(obs)
            a.requery = obs.k % 2 == 0
            return a
    assert _sig(cfg, ZeroAgent) == _sig(cfg, Requerier)
