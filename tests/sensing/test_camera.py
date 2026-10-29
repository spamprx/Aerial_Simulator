import numpy as np

from aerial.core.config import Config
from aerial.sensing.camera import sense_all
from aerial.world.scenario import generate_scenario


def _frames(seed=0):
    c = Config()
    s = generate_scenario(c, "district")
    rng = np.random.default_rng(seed)
    pos = np.stack([rng.uniform(15, 110, 10), rng.uniform(25, 115, 10), rng.uniform(5, 14, 10)], axis=1)
    pos = pos[s.terrain.height_at(pos[:, :2]) < pos[:, 2] - 1]
    yaw = rng.uniform(-np.pi, np.pi, len(pos))
    return c, s, pos, sense_all(s, pos, yaw, 0.0, c.sensor, rng)


def test_terrain_intervals_are_sound():
    c, s, pos, frames = _frames()
    for f in frames:
        if len(f.terrain_over):
            assert (f.terrain_over[:, 2] > s.terrain.height_at(f.terrain_over[:, :2]) - 1e-9).all()
        if len(f.terrain_hit):
            assert (f.terrain_hit[:, 2] <= np.maximum(s.terrain.height_at(f.terrain_hit[:, :2]), 0) + 1e-9).all()


def test_detected_iff_viewed():
    c, s, pos, frames = _frames(1)
    for f in frames:
        viewed = set(f.viewed.tolist())
        assert {lk.cell for lk in f.looks} == {x.cell for x in s.survivors if x.cell in viewed}


def test_contains_implies_sighted():
    c, s, pos, frames = _frames(2)
    for i, f in enumerate(frames):
        others = np.delete(pos, i, axis=0)
        inside = f.visible.contains(others, margin=0.0)
        for p in others[inside]:
            assert np.min(np.linalg.norm(f.sightings - p, axis=1)) < 1e-9
