"""T2.1: W-MSR in every coordination primitive.  One liar (F = 1) on a complete graph of 8, which is
4-robust and so more than the 3-robust the card asks for; test_robustness.py checks that claim."""
import warnings

import numpy as np
import pytest

from aerial.lab import SimConfig, Simulator, controller_registry
from aerial.lab.metrics import formation_error_series, min_separation_series, position_spread_series, velocity_spread_series
from aerial.lab.specifications import completion_time, obstacle_clearance_series

N, LIAR = 8, 7
OFFSET = [20.0, 0.0]
# The liar is not among the pinned agents here, so the only thing it can do is lie.
ATTACK = dict(compromised_ids=[LIAR], attack="constant_offset", attack_params={"pos_offset": OFFSET})
LIMITS = dict(max_speed=5.0, max_accel=4.0)


def fleet(controller, seed=0, t_final=40.0, attack=ATTACK, **kw):
    params = kw.pop("controller_params", {})
    return SimConfig(n_agents=N, dim=2, topology="complete", seed=seed, t_final=t_final, record_lambda2=False,
                     controller=controller, controller_params=params, **attack, **kw)


def run(cfg):
    sim = Simulator(cfg)
    return sim, sim.run()


def hull_overshoot(res):
    """How far any honest agent ever got outside the box around where the honest agents started."""
    honest = res.pos[:, res.honest]
    low, high = honest[0].min(axis=0), honest[0].max(axis=0)
    return max((low - honest).max(), (honest - high).max())


def drift(res, offsets=0.0):
    corrected = res.pos[:, res.honest] - (offsets[res.honest] if np.ndim(offsets) else 0.0)
    return np.linalg.norm(corrected[-1].mean(axis=0) - corrected[0].mean(axis=0))


# ---- the card's "done when", one primitive at a time ---------------------------------------
@pytest.mark.parametrize("unfiltered", [{}, {"filter": "mean"}], ids=["summed", "mean"])
def test_consensus_is_dragged_off_by_a_liar_unless_filtered(unfiltered):
    _, dragged = run(fleet("consensus", controller_params=unfiltered))
    _, filtered = run(fleet("consensus", controller_params={"filter": "wmsr"}))
    assert drift(dragged) > 100.0
    assert position_spread_series(filtered)[-1] < 1e-3
    assert hull_overshoot(filtered) < 1e-9  # the safety condition of MSR algorithms


def test_formation_is_dragged_off_by_a_liar_unless_filtered():
    params = {"offsets": "circle"}
    sim, dragged = run(fleet("formation", controller_params=params))
    assert drift(dragged, sim.controller.offsets) > 100.0
    sim, filtered = run(fleet("formation", controller_params={**params, "filter": "wmsr"}))
    assert formation_error_series(filtered, sim.controller.offsets)[-1] < 1e-3
    assert drift(filtered, sim.controller.offsets) < 5.0


def track_error(sim, res):
    offsets = sim.controller.offsets
    return np.linalg.norm((res.pos[-1, res.honest] - offsets[res.honest]).mean(axis=0) - sim.controller.reference(res.t[-1])[0])


def test_leader_follower_is_dragged_off_a_moving_reference_unless_filtered():
    params = {"offsets": "circle", "start": [5, 5], "velocity": [0.3, 0.2]}
    sim, dragged = run(fleet("leader_follower", t_final=40.0, controller_params={**params, "pinned": [0, 1, 2]}))
    assert track_error(sim, dragged) > 5.0 and formation_error_series(dragged, sim.controller.offsets)[-1] > 1.0
    sim, filtered = run(fleet("leader_follower", t_final=40.0, controller_params={**params, "filter": "wmsr"}))
    assert track_error(sim, filtered) < 0.1 and formation_error_series(filtered, sim.controller.offsets)[-1] < 0.1


def test_boids_velocity_is_hijacked_by_a_liar_unless_filtered():
    # A free flock has no place to be, so what is at stake is its velocity.
    def flock(flt, claimed):
        lie = dict(compromised_ids=[LIAR], attack="constant_offset",
                   attack_params={"pos_offset": OFFSET, "vel_offset": [claimed, 0.0]})
        return run(fleet("boids", attack=lie, seed=2, init_speed=1.0, min_init_separation=1.5, t_final=30.0,
                         controller_params={"filter": flt}, **LIMITS))[1]

    def speed(res):
        return np.linalg.norm(res.vel[-1, res.honest].mean(axis=0))

    assert speed(flock("mean", 4.0)) > 3.0  # it follows the lie
    modest, huge = flock("wmsr", 4.0), flock("wmsr", 20.0)
    assert speed(modest) < 1.5
    assert velocity_spread_series(modest)[-1] < 0.05
    assert min_separation_series(modest, agents="honest").min() >= 1.0
    assert modest.vel[-1, modest.honest] == pytest.approx(huge.vel[-1, huge.honest])  # the size of the lie is irrelevant


def launch(seed, n=N, low=2.0, high=10.0):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        candidate = rng.uniform(low, high, size=2)
        if all(np.linalg.norm(candidate - p) >= 2.0 for p in points):
            points.append(candidate)
    return np.array(points).tolist()


WORLD = {"obstacles": [{"center": [15.0, 15.0], "radius": 2.0}]}
HEAVY_LIE = dict(compromised_ids=[LIAR], attack="constant_offset",
                 attack_params={"pos_offset": OFFSET, "vel_offset": [-20.0, 0.0]})


def mission(behaviour, flt, seed=0, attack=HEAVY_LIE):
    return SimConfig(n_agents=N, dim=2, seed=seed, t_final=45.0, arena_size=32.0, comm_radius=200.0, record_lambda2=False,
                     init_pos=launch(seed), world=WORLD, mission={"waypoints": [[24.0, 24.0]]}, controller="mission",
                     controller_params={"behaviour": behaviour, "filter": flt}, **attack, **LIMITS)


@pytest.mark.parametrize("behaviour", ["formation", pytest.param("boids", marks=pytest.mark.slow)])
def test_mission_fails_under_a_liar_unless_filtered(behaviour):
    _, dragged = run(mission(behaviour, "mean"))
    assert not dragged.extras["mission_done"][-1][dragged.honest].all()
    assert completion_time(dragged) == float("inf")

    _, filtered = run(mission(behaviour, "wmsr"))
    assert filtered.extras["mission_done"][-1][filtered.honest].all() and np.isfinite(completion_time(filtered))
    assert min_separation_series(filtered, agents="honest").min() >= 1.0
    assert obstacle_clearance_series(filtered).min() > 0.0


@pytest.mark.slow
def test_switching_needs_the_filter_to_finish_a_sortie_under_a_liar():
    def sortie(flt):
        sweeps = [[[20.0 + 8.0 * (i % 3), 20.0 + 8.0 * (i // 3)]] for i in range(N)]
        return SimConfig(n_agents=N, dim=2, t_final=140.0, arena_size=50.0, comm_radius=200.0, record_lambda2=False,
                         init_pos=launch(1), controller="switching", controller_params={"filter": flt},
                         mission={"waypoints": [[6.0, 6.0], [26.0, 26.0]], "agent_waypoints": sweeps, "home": [6.0, 6.0]},
                         **HEAVY_LIE, **LIMITS)

    _, dragged = run(sortie("mean"))
    _, filtered = run(sortie("wmsr"))
    assert not dragged.extras["mission_done"][-1][dragged.honest].all()
    assert filtered.extras["mission_done"][-1][filtered.honest].all()
    assert np.diff(filtered.extras["mode"], axis=0).min() == 0
    assert min_separation_series(filtered, agents="honest").min() >= 1.0


# ---- the alias and old configs ---------------------------------------------------------------
def test_wmsr_consensus_is_an_alias_for_consensus_with_the_filter():
    _, alias = run(fleet("wmsr_consensus", t_final=5.0))
    _, spelled_out = run(fleet("consensus", t_final=5.0, controller_params={"filter": "wmsr"}))
    assert np.array_equal(alias.pos, spelled_out.pos) and np.array_equal(alias.extras["rejected"], spelled_out.extras["rejected"])


def test_old_wmsr_config_still_loads_and_runs():
    res = Simulator(SimConfig.from_dict({"n_agents": 6, "t_final": 2.0, "n_compromised": 1, "attack": "constant_offset",
                                         "attack_params": {"pos_offset": [5, 0, 0]}, "controller": "wmsr_consensus",
                                         "controller_params": {"f": 1}})).run()
    assert res.extras["rejected"].any()


def test_f_comes_from_the_config_unless_given():
    assert Simulator(fleet("wmsr_consensus", attack={}, F=2)).controller.aggregator.f == 2
    assert Simulator(fleet("wmsr_consensus", attack={}, F=2, controller_params={"f": 1})).controller.aggregator.f == 1


def test_plain_consensus_keeps_its_summed_law():
    sim = Simulator(fleet("consensus", attack={}, t_final=1.0))
    assert sim.controller.filter is None and sim.controller.normalised_gains() is None and sim.controller.diagnostics() == {}


# ---- diagnostics per term --------------------------------------------------------------------
PRIMITIVES = {
    "consensus": {"filter": "wmsr", "damping": "relative"},
    "formation": {"filter": "wmsr", "damping": "relative"},
    "leader_follower": {"filter": "wmsr"},
    "boids": {"filter": "wmsr"},
    "mission": {"filter": "wmsr"},
}


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_every_primitive_records_who_it_used_per_term(name):
    cfg = fleet(name, t_final=1.0, controller_params=PRIMITIVES[name], **LIMITS)
    res = Simulator(cfg).run()
    K = cfg.n_steps
    for key in ("accepted_pos", "rejected_pos", "accepted_vel", "rejected_vel", "accepted", "rejected"):
        assert res.extras[key].shape == (K, N, N) and res.extras[key].dtype == bool
    for term in ("pos", "vel"):
        assert np.array_equal(res.extras[f"accepted_{term}"] | res.extras[f"rejected_{term}"], res.adj)  # all it received
        assert not (res.extras[f"accepted_{term}"] & res.extras[f"rejected_{term}"]).any()
    # "rejected" means removed from any term; "accepted" means used in every one.
    assert np.array_equal(res.extras["rejected"], res.extras["rejected_pos"] | res.extras["rejected_vel"])
    assert np.array_equal(res.extras["accepted"], res.extras["accepted_pos"] & res.extras["accepted_vel"])


def test_an_obvious_liar_is_rejected_nearly_always():
    res = Simulator(fleet("consensus", controller_params={"filter": "wmsr"})).run()
    assert res.extras["rejected_pos"][:, res.honest, LIAR].mean() > 0.99


def test_a_plain_mean_rejects_nobody():
    res = Simulator(fleet("formation", t_final=1.0)).run()
    assert not res.extras["rejected"].any() and np.array_equal(res.extras["accepted"], res.adj)


def test_own_damping_has_no_velocity_term_to_record():
    res = Simulator(fleet("consensus", t_final=1.0, controller_params={"filter": "wmsr"})).run()
    assert not res.extras["accepted_vel"].any() and not res.extras["rejected_vel"].any()
    assert np.array_equal(res.extras["accepted"], res.extras["accepted_pos"])


def test_a_neighbour_kept_in_x_but_dropped_in_y_counts_as_rejected():
    # Bug B4(d): the old controller counted this as accepted, and rejected nobody on a 3-D complete graph.
    params = dict(n_agents=4, dim=2, t_final=1, comm_radius=1000.0, arena_size=1000.0, controller="wmsr_consensus",
                  init_pos=[[0, 0], [1, 1], [2, 2], [1.5, 900]])
    sim = Simulator(SimConfig(**params))
    sim.controller.control(sim.sense())
    assert sim.controller.rejected[0, 3] and not sim.controller.accepted[0, 3]
    three_d = SimConfig(n_agents=8, dim=3, topology="complete", t_final=1.0, controller="wmsr_consensus")
    assert Simulator(three_d).run().extras["rejected"].any()


def test_diagnostics_are_rebuilt_each_time_control_is_called():
    sim = Simulator(fleet("formation", t_final=1.0, controller_params={"filter": "wmsr"}))
    obs = sim.sense()
    sim.controller.control(obs)
    first = sim.controller.diagnostics()["rejected"].copy()
    sim.controller.control(obs)
    assert np.array_equal(sim.controller.diagnostics()["rejected"], first)


# ---- why the velocity term is what it is ------------------------------------------------------
@pytest.mark.slow
def test_relative_damping_under_wmsr_lets_the_whole_fleet_sail_off():
    # The agents agree perfectly, so spread alone would call this a success.  Why "own" is the default.
    params = {"filter": "wmsr", "damping": "relative"}
    _, sailing = run(fleet("consensus", controller_params=params))
    assert position_spread_series(sailing)[-1] < 1e-3 and hull_overshoot(sailing) > 10.0


def test_damping_defaults_follow_the_filter():
    assert controller_registry.create("consensus").damping == "relative"
    assert controller_registry.create("consensus", filter="mean").damping == "relative"
    assert controller_registry.create("consensus", filter="wmsr").damping == "own"
    assert controller_registry.create("wmsr_consensus").damping == "own"
    assert controller_registry.create("formation").damping == "relative"
    assert controller_registry.create("formation", filter="wmsr").damping == "own"
    assert controller_registry.create("leader_follower", filter="wmsr").damping == "relative"  # it has to move


@pytest.mark.parametrize("name, params", [("consensus", {"damping": "own"}), ("consensus", {"filter": "wmsr", "damping": "x"}),
                                          ("formation", {"damping": "x"})])
def test_bad_damping_settings_are_rejected(name, params):
    with pytest.raises(ValueError):
        controller_registry.create(name, **params)


def test_own_damping_law_on_a_hand_computed_case():
    cfg = SimConfig(n_agents=3, dim=2, t_final=1, comm_radius=100, controller="consensus", init_vel=[[2, 0], [0, 0], [0, 0]],
                    controller_params={"filter": "mean", "damping": "own", "k_p": 2.0, "k_v": 0.5},
                    init_pos=[[0, 0], [4, 0], [8, 0]])
    sim = Simulator(cfg)
    # 2 * (mean(4, 8) - 0) - 0.5 * 2
    assert sim.controller.control(sim.sense())[0].tolist() == [11.0, 0.0]


# ---- pinning under W-MSR ----------------------------------------------------------------------
@pytest.mark.parametrize("flt, F, n, expected", [("mean", 1, 8, [0]), ("wmsr", 1, 8, [0, 1, 2]), ("wmsr", 2, 8, [0, 1, 2, 3, 4]),
                                                 ("wmsr", 2, 3, [0, 1, 2]), ("wmsr", 0, 8, [0])])
def test_default_pins_are_enough_for_the_filter(flt, F, n, expected):
    cfg = SimConfig(n_agents=n, dim=2, t_final=1, F=F, controller="leader_follower", controller_params={"filter": flt})
    assert Simulator(cfg).controller.pinned == expected


@pytest.mark.slow
def test_too_few_pins_warn_and_really_do_cut_the_followers_off():
    params = {"filter": "wmsr", "pinned": [0], "offsets": "circle", "start": [5, 5], "velocity": [0.5, 0.25]}
    with pytest.warns(UserWarning, match="needs 3 pinned"):
        sim = Simulator(fleet("leader_follower", attack={}, t_final=80.0, controller_params=params))
    res = sim.run()
    assert formation_error_series(res, sim.controller.offsets)[-1] > 1.0  # the leader has flown off alone


@pytest.mark.slow
def test_enough_pins_do_not_warn_and_the_fleet_follows_even_when_one_of_them_lies():
    params = {"filter": "wmsr", "offsets": "circle", "start": [5, 5], "velocity": [0.3, 0.2]}
    lying_leader = dict(compromised_ids=[1], attack="constant_offset", attack_params={"pos_offset": [-20.0, 0.0]})
    cfg = fleet("leader_follower", t_final=50.0, attack=lying_leader, controller_params=params)
    with pytest.warns(UserWarning, match="compromised agent is pinned"):
        sim = Simulator(cfg)
    res = sim.run()
    assert track_error(sim, res) < 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulator(fleet("leader_follower", attack={}, t_final=1.0, controller_params=params))


def test_the_pinned_reference_itself_is_not_filtered():
    params = {"filter": "wmsr", "k_p": 0.0, "k_v": 0.0, "k_pin": 2.0, "k_pin_v": 0.0, "start": [10, 0], "offsets": [[0, 0]] * N}
    sim = Simulator(fleet("leader_follower", attack={}, t_final=1, controller_params=params, comm_radius=1000.0))
    # A pinned agent's pull towards the reference is the same whatever anyone reports.
    expected = 2.0 * (np.array([10.0, 0.0]) - sim.pos[0])
    assert sim.controller.control(sim.sense())[0] == pytest.approx(expected)


# ---- a phantom threat -----------------------------------------------------------------------
def test_a_false_velocity_report_cannot_change_the_safety_part():
    # Separation is unfiltered, so a liar claiming "I am coming at you at 20 m/s" must have no effect.
    def safety_with(vel_offset):
        attack = dict(compromised_ids=[LIAR], attack="constant_offset",
                      attack_params={"pos_offset": OFFSET, "vel_offset": vel_offset})
        sim = Simulator(mission("formation", "wmsr", attack=attack))
        return sim.controller.control_parts(sim.sense())["safety"]

    assert np.array_equal(safety_with([0.0, 0.0]), safety_with([-20.0, 0.0]))


def test_a_false_position_close_by_is_still_avoided():
    # The other side of the same choice: the term is unfiltered, so a claim to be close is believed.
    def safety_with(offset):
        cfg = SimConfig(n_agents=4, dim=2, t_final=1, comm_radius=100.0, arena_size=40.0, controller="mission",
                        init_pos=[[10, 10], [20, 20], [25, 25], [14, 14]], controller_params={"filter": "wmsr"},
                        compromised_ids=[3], attack="constant_offset", attack_params={"pos_offset": offset}, **LIMITS)
        sim = Simulator(cfg)
        return sim.controller.control_parts(sim.sense())["safety"][0]

    assert not safety_with([0.0, 0.0]).any()
    assert safety_with([-3.0, -4.0])[0] < 0.0  # it claims to be at (11, 10), so agent 0 backs off along x
