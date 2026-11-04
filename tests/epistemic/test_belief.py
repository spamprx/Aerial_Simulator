import numpy as np

from aerial.epistemic.belief import AliveBelief


def simulate(alive, rng):
    b = AliveBelief()
    while b.decision is None and b.n < 30:
        if alive:
            heat, motion = rng.normal(34, 1), rng.random() < 0.46
        else:
            heat, motion = rng.uniform(25, 30), rng.random() < 0.02
        b.update(heat, motion)
    return b


def test_sprt_accuracy_and_speed():
    rng = np.random.default_rng(0)
    res = [(alive, simulate(alive, rng)) for alive in rng.random(500) < 0.7]
    acc = np.mean([b.decision == ("alive" if a else "dead") for a, b in res])
    assert acc >= 0.95
    assert np.mean([b.n for _, b in res]) <= 4
