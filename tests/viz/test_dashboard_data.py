import numpy as np
import pytest

from aerial.core.config import Config
from aerial.viz import dashboard_data as dd


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    c = Config(N=4).override(["world.scenario=open_field", "sim.T=6", "world.n_survivors=3",
                              "attack.n_compromised=1", "attack.start_time=1"])
    return dd.run_pair_cached(c, ("naive", "gated"), cache=tmp_path_factory.mktemp("cache"))


def test_cache_round_trip(pair, tmp_path):
    runs, scn = pair
    assert set(runs) == {"naive", "gated"}
    for res in runs.values():
        assert dd.knowledge_matrix(res, 3).shape == (4, 4)
        assert len(dd.fraction_knowing(res)) == len(res.t) - 1
        assert all(len(v) == len(res.t) - 1 for v in dd.cleared_curves(res).values())


def test_si_fit_recovers_rate():
    t = np.linspace(0, 20, 200)
    i0, beta = 0.1, 0.8
    frac = 1 / (1 + (1 / i0 - 1) * np.exp(-beta * t))
    b, curve = dd.si_fit(t, frac)
    assert abs(b - beta) < 0.05


def test_scene_builds(pair):
    pytest.importorskip("plotly")
    from aerial.viz.scene3d import scene_figure
    runs, scn = pair
    fig = scene_figure(runs["gated"], scn, n_frames=5)
    assert len(fig.frames) == 5
