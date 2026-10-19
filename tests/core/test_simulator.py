import numpy as np

from aerial.agents.base import WaypointAgent, ZeroAgent
from aerial.core.result import SimResult
from aerial.core.simulator import Simulator
from tests.conftest import line_positions


def test_zero_agents_round_trip(cfg, tmp_path):
    res = Simulator(cfg, [ZeroAgent() for _ in range(4)], line_positions(4)).run()
    res.save(tmp_path / "run")
    back = SimResult.load(tmp_path / "run")
    assert np.array_equal(back.pos, res.pos) and back.config == cfg
    assert np.allclose(res.pos[-1], res.pos[0])


def test_deterministic_and_pinned(cfg):
    def run():
        return Simulator(cfg, [WaypointAgent() for _ in range(4)], line_positions(4)).run()
    a, b = run(), run()
    assert np.array_equal(a.pos, b.pos)
    # regression pin: drones converge towards the waypoint
    assert np.linalg.norm(a.pos[-1] - [50, 50, 9], axis=1).max() < np.linalg.norm(a.pos[0] - [50, 50, 9], axis=1).max()


def test_events_relay_one_hop_per_tick():
    from aerial.comms.messages import Event, EventKind
    from aerial.comms.network import Gossip
    g = Gossip(4, 64)
    A = np.zeros((4, 4), bool)
    for i in range(3):          # line 0-1-2-3, both directions
        A[i, i + 1] = A[i + 1, i] = True
    g.originate(0, Event(0, 1, 0.0, EventKind.CLEARED, None))
    hops = [sorted(i for i, evs in enumerate(g.exchange(A)) if evs) for _ in range(3)]
    assert hops == [[1], [2], [3]]


def test_bandwidth_cap_and_priority():
    from aerial.comms.messages import Event, EventKind
    from aerial.comms.network import Gossip
    g = Gossip(2, 2)
    A = np.array([[False, True], [True, False]])
    for s, kind in enumerate([EventKind.RANGES, EventKind.CLEARED, EventKind.ACCUSE]):
        g.originate(0, Event(0, s, 0.0, kind, None))
    got = g.exchange(A)[1]
    assert [e.kind for e in got] == [EventKind.ACCUSE, EventKind.CLEARED]
