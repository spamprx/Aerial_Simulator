"""The CUDA ray caster must match the numpy one exactly (runs only where a GPU and torch exist)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import aerial.world.gpu as gpu
from aerial.core.config import Config
from aerial.world.geometry import raycast
from aerial.world.scenario import generate_scenario


def test_gpu_raycast_matches_numpy(monkeypatch):
    monkeypatch.setattr(gpu, "_DEVICE", "cuda")
    s = generate_scenario(Config(), "district")
    rng = np.random.default_rng(0)
    o = np.stack([rng.uniform(10, 110, 500), rng.uniform(20, 120, 500), rng.uniform(4, 15, 500)], axis=1)
    d = rng.normal(size=(500, 3))
    d[:, 2] = -np.abs(d[:, 2])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a = raycast(s.terrain, o, d, 25.0, step=0.4)
    b = gpu.GpuTerrain(s.terrain).raycast(o, d, 25.0, step=0.4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
