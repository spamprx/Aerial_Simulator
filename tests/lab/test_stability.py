import warnings

import numpy as np
import pytest

from aerial.lab import DiskComm, FixedComm, SimConfig, Simulator, algebraic_connectivity, make_topology
from aerial.lab.metrics import decay_rate, position_spread_series
from aerial.lab.stability import (fleet_rate, is_stable, is_stable_normalised, max_stable_dt, predicted_rate,
                              spectral_radius, update_matrix, worst_case_lambda)

DT = 0.05


def consensus(n=20, topology="complete", k_p=1.0, k_v=1.5, dt=DT, t_final=20.0, **kw):
    params = dict(n_agents=n, dim=2, topology=topology, dt=dt, t_final=t_final, record_extras=False, record_lambda2=False,
                  controller_params={"k_p": k_p, "k_v": k_v})
    return SimConfig(**{**params, **kw})


def unwarned(cfg):
    """Run with the stability warning out of the way, for tests about the outcome only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Simulator(cfg).run()


# ---- the update matrix --------------------------------------------------------------------
def test_update_matrix_is_exactly_what_the_simulator_does():
    # Two agents on a complete graph: the difference p0 - p1 is the mode with lam = 2.
    k_p, k_v, dt = 0.7, 1.3, 0.1
    res = Simulator(consensus(n=2, k_p=k_p, k_v=k_v, dt=dt, t_final=1.0, init_pos=[[0, 0], [3, 0]],
                              init_vel=[[1, 0], [-1, 0]])).run()
    state = np.array([res.pos[0, 0, 0] - res.pos[0, 1, 0], res.vel[0, 0, 0] - res.vel[0, 1, 0]])
    M = update_matrix(k_p, k_v, dt, 2.0)
    for k in range(1, len(res.t)):
        state = M @ state
        assert state == pytest.approx([res.pos[k, 0, 0] - res.pos[k, 1, 0], res.vel[k, 0, 0] - res.vel[k, 1, 0]], abs=1e-12)


def test_spectral_radius_on_simple_cases():
    assert spectral_radius(1.0, 1.5, DT, 0.0) == pytest.approx(1.0)  # the agreement mode never decays
    # k_p = 0 makes M triangular, with eigenvalues 1 and 1 - k_v * lam * dt.
    assert spectral_radius(0.0, 2.0, 0.5, 3.0) == pytest.approx(2.0)


# ---- the observed result the card names ---------------------------------------------------
def test_predicts_the_observed_blow_up_at_full_fleet_size():
    lam_max = worst_case_lambda(FixedComm(make_topology("complete", 20)), 20)
    assert lam_max == pytest.approx(20.0)
    assert not is_stable(1.0, 3.0, DT, lam_max)
    assert is_stable(1.0, 1.5, DT, lam_max)


def test_a_simulation_confirms_both_predictions():
    assert position_spread_series(unwarned(consensus(k_v=3.0)))[-1] > 1e50
    assert position_spread_series(unwarned(consensus(k_v=1.5)))[-1] < 1e-3


@pytest.mark.parametrize("k_v, stable", [(3.8, True), (4.2, False)])
def test_prediction_is_sharp_on_either_side_of_the_boundary(k_v, stable):
    # A star on 10 nodes has lam_max = 10, so the boundary is k_v * 10 * 0.05 = 2, at k_v = 4.
    lam_max = worst_case_lambda(FixedComm(make_topology("star", 10)), 10)
    assert lam_max == pytest.approx(10.0)
    assert is_stable(1.0, k_v, DT, lam_max) is stable
    spread = position_spread_series(unwarned(consensus(n=10, topology="star", k_v=k_v, t_final=40.0)))
    assert bool(spread[-1] < spread[0]) is stable


def test_too_little_damping_is_unstable_at_every_lambda():
    # k_p * dt / 2 > k_v: no eigenvalue is safe, so it grows on any graph.  Slowly, unlike the above.
    assert not is_stable(10.0, 0.2, DT, 1.0)
    assert is_stable(10.0, 0.3, DT, 1.0)
    spread = position_spread_series(unwarned(consensus(n=4, k_p=10.0, k_v=0.2, t_final=100.0)))
    assert spread[-1] > 10.0 * spread[0]


def test_no_damping_is_unstable_at_any_step_size():
    assert not is_stable(1.0, 0.0, DT, 2.0) and not is_stable(1.0, 0.0, 1e-3, 2.0)
    assert max_stable_dt(1.0, 0.0, 2.0) == 0.0


def test_velocity_alignment_alone_counts_as_stable_until_its_own_limit():
    # k_p = 0: positions hold still rather than converge, and that is not a divergence.
    assert is_stable(0.0, 2.0, DT, 8.0)
    assert not is_stable(0.0, 2.0, DT, 21.0)  # 2 * 21 * 0.05 > 2
    assert max_stable_dt(0.0, 2.0, 8.0) == pytest.approx(0.125)


def test_a_lone_agent_has_nothing_to_diverge():
    assert is_stable(1.0, 0.0, DT, 0.0) and max_stable_dt(1.0, 1.5, 0.0) == float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(SimConfig(n_agents=1, dt=0.5, init_pos=[[1.0, 1.0, 1.0]]))


def test_no_gains_at_all_are_not_a_divergence():
    assert is_stable(0.0, 0.0, DT, 5.0) and max_stable_dt(0.0, 0.0, 5.0) == float("inf")


# ---- the closed form ----------------------------------------------------------------------
@pytest.mark.parametrize("k_p, k_v, lam_max", [(1.0, 1.5, 20.0), (1.0, 3.0, 20.0), (4.0, 0.5, 2.0), (10.0, 0.2, 1.0),
                                               (0.5, 1.0, 38.0), (0.0, 2.0, 8.0)])
def test_max_stable_dt_is_where_the_grid_check_flips(k_p, k_v, lam_max):
    dt = max_stable_dt(k_p, k_v, lam_max)
    assert is_stable(k_p, k_v, dt * 0.999, lam_max)
    assert not is_stable(k_p, k_v, dt * 1.001, lam_max)


def test_closed_form_agrees_with_the_eigenvalues_on_random_gains():
    rng = np.random.default_rng(1)
    for _ in range(500):
        k_p, k_v, dt, lam = rng.uniform(0.01, 8.0), rng.uniform(0.01, 8.0), rng.uniform(0.005, 0.5), rng.uniform(0.01, 40.0)
        closed_form = k_v * lam * dt < 2.0 and k_p * dt / 2.0 < k_v
        if abs(k_v * lam * dt - 2.0) > 1e-3 and abs(k_p * dt / 2.0 - k_v) > 1e-3:  # not on the boundary
            assert (spectral_radius(k_p, k_v, dt, lam) < 1.0) == closed_form


# ---- the worst case -----------------------------------------------------------------------
@pytest.mark.parametrize("topology, n, expected", [("complete", 20, 20.0), ("star", 7, 7.0), ("ring", 10, 4.0),
                                                   ("line", 2, 2.0)])
def test_worst_case_on_a_fixed_graph_with_no_loss_is_exact(topology, n, expected):
    assert worst_case_lambda(FixedComm(make_topology(topology, n)), n) == pytest.approx(expected)


def test_worst_case_with_packet_loss_or_a_disk_graph_is_the_smaller_bound():
    assert worst_case_lambda(FixedComm(make_topology("ring", 10), drop_prob=0.2), 10) == 4.0  # 2 * d_max
    assert worst_case_lambda(FixedComm(make_topology("complete", 10), drop_prob=0.2), 10) == 10.0  # n
    assert worst_case_lambda(DiskComm(5.0), 20) == 20.0
    assert worst_case_lambda(DiskComm(5.0), 1) == 0.0


def test_loss_only_ever_removes_links_so_the_bound_holds_on_symmetric_subgraphs():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = rng.integers(2, 21)
        upper = np.triu(rng.random((n, n)) < rng.uniform(0.1, 1.0), 1)
        A = upper | upper.T
        L = np.diag(A.sum(axis=1)) - A.astype(float)
        assert np.linalg.eigvalsh(L).max() <= worst_case_lambda(DiskComm(1.0), n) + 1e-9


# ---- normalised controllers (D11) ---------------------------------------------------------
def test_normalised_check_uses_lambda_up_to_two():
    assert is_stable_normalised(1.0, 3.0, DT)  # the gain that destroys a summed fleet of 20
    assert not is_stable_normalised(1.0, 25.0, DT)  # 25 * 2 * 0.05 > 2


def test_normalised_eigenvalues_really_are_within_two():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = rng.integers(2, 21)
        upper = np.triu(rng.random((n, n)) < 0.5, 1)
        A = (upper | upper.T).astype(float)
        degree = np.maximum(A.sum(axis=1), 1.0)
        assert np.abs(np.linalg.eigvals(np.eye(n) - A / degree[:, None])).max() <= 2.0 + 1e-9


@pytest.mark.parametrize("k_p, k_v, dt, lam", [(1.0, 1.5, DT, 2.0), (1.0, 30.0, DT, 2.0), (1.0, 45.0, DT, 2.0),
                                               (50.0, 0.5, DT, 2.0), (1.0, 1.5, 0.5, 1.0)])
def test_own_damping_stability_has_the_stated_closed_form(k_p, k_v, dt, lam):
    closed_form = k_v * dt < 2.0 and k_p * lam * dt / 2.0 < k_v
    assert (spectral_radius(k_p, k_v, dt, lam, own_damping=True) < 1.0) == closed_form


def test_own_damping_prediction_is_confirmed_by_a_simulation():
    def spread(k_v):
        cfg = SimConfig(n_agents=6, dim=2, topology="star", t_final=30.0, record_extras=False, record_lambda2=False,
                        controller="consensus", controller_params={"filter": "mean", "damping": "own", "k_v": k_v})
        return position_spread_series(unwarned(cfg))[-1]

    # The gain that breaks relative damping on this graph is fine here.
    assert is_stable(1.0, 25.0, DT, 2.0, own_damping=True) and spread(25.0) < 100.0
    assert not is_stable(1.0, 45.0, DT, 2.0, own_damping=True) and spread(45.0) > 1e6


def test_normalised_prediction_is_confirmed_by_a_simulation():
    # A star has the normalised eigenvalue 2 exactly, so the check is tight on it.
    def spread(k_v):
        cfg = SimConfig(n_agents=6, dim=2, topology="star", t_final=30.0, controller="formation", record_extras=False,
                        record_lambda2=False, controller_params={"k_v": k_v})
        return position_spread_series(unwarned(cfg))[-1]

    assert is_stable_normalised(1.0, 15.0, DT) and spread(15.0) < 100.0
    assert not is_stable_normalised(1.0, 25.0, DT) and spread(25.0) > 1e6


# ---- the warning --------------------------------------------------------------------------
def test_simulator_warns_about_unstable_summed_gains():
    with pytest.warns(UserWarning, match="SecondOrderConsensus .* unstable"):
        Simulator(consensus(k_v=3.0))


@pytest.mark.parametrize("cfg", [
    consensus(k_v=1.5),                                       # the card's stable case, and on a disk:
    consensus(topology="disk", comm_radius=100.0, k_v=1.5),
    consensus(n=10, topology="ring", k_v=3.0),                # k_v * 4 * dt is well below 2 on a ring
    SimConfig(controller="formation"),
    SimConfig(controller="boids"),
    SimConfig(controller="mission"),
    SimConfig(controller="zero"),
    SimConfig.load("configs/lab_default.json"),
])
def test_simulator_is_quiet_about_stable_setups(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(cfg)


def test_warning_depends_on_the_graph_not_just_the_gain():
    with pytest.warns(UserWarning, match="unstable"):
        Simulator(consensus(k_v=3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(consensus(topology="ring", k_v=3.0))


@pytest.mark.parametrize("controller, params", [
    ("formation", {"k_v": 25.0}),
    ("leader_follower", {"k_v": 25.0}),
    ("boids", {"k_ali": 25.0}),
    ("wmsr_consensus", {"k_v": 45.0}),  # own damping: this needs k_v * dt > 2, whatever lambda is
    ("mission", {"behaviour_params": {"k_v": 25.0}}),
])
def test_simulator_warns_about_unstable_normalised_gains(controller, params):
    with pytest.warns(UserWarning, match="normalised gains .* unstable"):
        Simulator(SimConfig(n_agents=4, controller=controller, controller_params=params))


def test_warning_says_what_step_size_would_work():
    with pytest.warns(UserWarning, match=r"needs dt < 0\.03333"):
        Simulator(consensus(k_v=3.0))


# ---- how fast ------------------------------------------------------------------------------
def measured_rate(res):
    return decay_rate(position_spread_series(res), res.t)


@pytest.mark.parametrize("topology, n, t_final", [("ring", 8, 60.0), ("line", 6, 120.0), ("complete", 8, 40.0),
                                                  ("star", 8, 60.0), ("grid", 9, 60.0)])
def test_predicted_decay_rate_matches_a_simulation(topology, n, t_final):
    # fleet_rate is the asymptotic rate.  A slowest mode that starts with little weight may not have
    # taken over yet inside the window, so the measured slope can sit a little above it.
    res = unwarned(consensus(n=n, topology=topology, t_final=t_final))
    assert measured_rate(res) == pytest.approx(fleet_rate(1.0, 1.5, DT, res.adj[0]), rel=0.06)


def test_lambda_2_alone_predicts_sparse_graphs_but_not_dense_ones():
    # On a ring lambda_2 is the slowest mode.  On a complete graph, with k_v^2 * lam > 4 * k_p, it
    # is overdamped, and the slow root heads towards k_p / k_v = 2/3 however large lam gets.
    ring = make_topology("ring", 12)
    assert predicted_rate(1.0, 1.5, DT, algebraic_connectivity(ring)) == pytest.approx(fleet_rate(1.0, 1.5, DT, ring))
    assert predicted_rate(1.0, 1.5, DT, 1.78) > predicted_rate(1.0, 1.5, DT, 20.0) > 2.0 / 3.0 - 0.05


def test_decay_rate_recovers_a_known_exponential():
    t = np.arange(0.0, 100.0, 0.05)
    assert decay_rate(3.0 * np.exp(-0.4 * t), t) == pytest.approx(0.4)


def test_decay_rate_ignores_the_transient_and_the_numerical_floor():
    t = np.arange(0.0, 100.0, 0.05)
    series = np.maximum(50.0 * np.exp(-5.0 * t) + np.exp(-0.4 * t), 1e-14)
    assert decay_rate(series, t) == pytest.approx(0.4, rel=0.01)


def test_decay_rate_is_nan_if_the_series_never_falls_far_enough():
    t = np.arange(0.0, 1.0, 0.05)
    assert np.isnan(decay_rate(np.ones_like(t), t))


def test_a_disconnected_fleet_has_no_decay_rate():
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = True
    assert fleet_rate(1.0, 1.5, DT, A) == 0.0
    assert fleet_rate(1.0, 1.5, DT, np.zeros((1, 1), dtype=bool)) == 0.0
    assert predicted_rate(1.0, 1.5, DT, 0.0) == pytest.approx(0.0)
