import numpy as np
import pytest

from aerial.core.config import Config


@pytest.fixture
def cfg():
    c = Config(N=4)
    c.sim.T = 2.0
    return c


def line_positions(n, spacing=5.0, z=9.0):
    return np.array([[10.0 + spacing * i, 10.0, z] for i in range(n)])
