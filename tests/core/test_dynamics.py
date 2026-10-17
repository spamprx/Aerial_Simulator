import numpy as np

from aerial.core.dynamics import step_double_integrator, step_yaw


def test_exact_when_unsaturated():
    p, v, u = np.zeros((1, 3)), np.array([[1.0, 0, 0]]), np.array([[0.5, 0, 0.2]])
    p2, v2, _ = step_double_integrator(p, v, u, 0.1, 4.0, 5.0, 2.0)
    assert np.allclose(v2, v + u * 0.1)
    assert np.allclose(p2, p + v * 0.1 + 0.5 * u * 0.01)


def test_saturation_separates_horizontal_and_vertical():
    p, v = np.zeros((1, 3)), np.array([[5.0, 0, 2.0]])
    _, v2, u = step_double_integrator(p, v, np.array([[10.0, 0, 10.0]]), 0.1, 4.0, 5.0, 2.0)
    assert np.linalg.norm(u) <= 4.0 + 1e-9
    assert np.linalg.norm(v2[0, :2]) <= 5.0 + 1e-9 and abs(v2[0, 2]) <= 2.0 + 1e-9


def test_yaw_wraps_and_is_rate_limited():
    y = step_yaw(np.array([3.1]), np.array([100.0]), 0.1, np.pi / 2)
    assert -np.pi <= y[0] < np.pi
    assert np.isclose(y[0], (3.1 + np.pi / 20 + np.pi) % (2 * np.pi) - np.pi)
