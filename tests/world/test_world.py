import numpy as np
import pytest

from aerial.core.config import Config
from aerial.world.geometry import los_clear, raycast
from aerial.world.scenario import SCENARIOS, generate_scenario


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenarios_valid_and_deterministic(name):
    c = Config()
    c.attack.n_compromised = 1
    a, b = generate_scenario(c, name), generate_scenario(c, name)
    assert np.array_equal(a.terrain.H, b.terrain.H)
    assert a.viewable.mean() > 0.5
    cells = a.survivor_cells
    assert len(cells) == c.world.n_survivors and a.viewable[cells].all() and a.coverable[cells].all()
    assert a.liars.sum() == 1


def test_demo_claim_region_holds_survivors():
    c = Config()
    c.attack.n_compromised = 1
    s = generate_scenario(c, "demo_district")
    assert np.isin(s.survivor_cells, s.claim_region).sum() >= 3


def test_thin_column_never_skipped():
    c = Config()
    s = generate_scenario(c, "open_field")
    s.terrain.H[:] = 0
    s.terrain.H[50, 50] = 20.0                         # one 1 m column
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = 50 + rng.uniform(0.05, 0.95)
        o = np.array([[40.0, y, 9.0]])
        d, _, _, _ = raycast(s.terrain, o, np.array([[1.0, 0, 0]]), 25.0)
        assert 9.9 <= d[0] <= 11.0
        assert not los_clear(s.terrain, o[0], np.array([60.0, y, 9.0]))
