import numpy as np

from aerial.core.types import AgentObservation, ClaimedState, DroneState, ReceivedBeacon, StaticInfo
from aerial.epistemic.evidence import check_audit, check_beacons, check_shared_range

ST = StaticInfo(i=0, N=3, F=1, arena=120, zone=100, z_min=4, z_max=15, base=np.zeros(3), dt=0.1)


def obs_with(beacons):
    return AgentObservation(t=0, k=0, i=0, own=DroneState(np.zeros(3), np.zeros(3), 0.0), frame=None,
                            beacons=beacons, events=(), static=ST)


def b(j, pos, rng, vel=(0, 0, 0), seq=0, age=0):
    return ReceivedBeacon(j, ClaimedState(np.array(pos, float), np.array(vel, float)), seq, age, rng)


def test_r1_tolerates_noise_up_to_eps_exactly():
    eps = 0.2
    assert check_beacons(obs_with({1: b(1, [10, 0, 0], 10.2)}), eps, 35, 5, 2, {}) == []
    assert check_beacons(obs_with({1: b(1, [10, 0, 0], 10.4)}), eps, 35, 5, 2, {}) == [("R1", (1,))]


def test_r2_uses_separate_limits_so_climbing_is_fine():
    assert check_beacons(obs_with({1: b(1, [10, 0, 0], 10.0, vel=(5, 0, 2))}), 0.2, 35, 5, 2, {}) == []
    assert ("R2", (1,)) in check_beacons(obs_with({1: b(1, [10, 0, 0], 10.0, vel=(6, 0, 0))}), 0.2, 35, 5, 2, {})


def test_r3_radio_horizon():
    assert ("R3", (1,)) in check_beacons(obs_with({1: b(1, [50, 0, 0], 50.0)}), 0.2, 35, 5, 2, {})


def test_r5_audit_accuses_every_reporter_of_a_survivor_cell():
    rep = np.zeros(10, dtype=np.uint64)
    rep[4] = (1 << 2) | (1 << 0)
    assert check_audit(np.array([4]), rep, me=0) == [("R5", (2,))]


def test_r6_is_a_clause():
    hist = {2: {7: np.array([10.0, 0, 0])}}
    assert check_shared_range(1, 2, 7, 3.0, np.zeros(3), hist, 0.2) == [("R6", (2, 1))]
    assert check_shared_range(1, 2, 7, 10.0, np.zeros(3), hist, 0.2) == []
