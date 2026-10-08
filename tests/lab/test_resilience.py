import numpy as np
import pytest

from aerial.lab import (MeanAggregator, SimConfig, Simulator, WMSRAggregator, aggregator_registry,
                    attack_registry, build_aggregator, controller_registry)
from aerial.lab.metrics import position_spread_series


def test_scope_limits():
    with pytest.raises(ValueError):
        SimConfig(n_agents=21).validate()
    with pytest.raises(ValueError):
        SimConfig(n_agents=3, n_compromised=3).validate()


def test_registered_attacks_change_only_compromised_broadcasts():
    cfg = SimConfig(
        n_agents=3, dim=2, t_final=1, n_compromised=1,
        attack="constant_offset", attack_params={"pos_offset": [2, 0]},
    )
    sim = Simulator(cfg)
    obs = sim.sense()
    assert np.allclose(obs.msg_pos[~sim.compromised], obs.pos[~sim.compromised])
    assert np.allclose(obs.msg_pos[sim.compromised], obs.pos[sim.compromised] + [2, 0])


@pytest.mark.filterwarnings("ignore:.*is unstable")  # gains chosen for easy arithmetic, not to fly
def test_wmsr_rejects_extreme_neighbor():
    controller = controller_registry.create("wmsr_consensus", k_p=1.0, k_v=0.0, f=1)
    cfg = SimConfig(n_agents=4, dim=2, t_final=1, comm_radius=100)
    sim = Simulator(cfg, controller=controller,
                    init_pos=np.array([[0, 0], [1, 0], [2, 0], [100, 0]], dtype=float))
    obs = sim.sense()
    controller.control(obs)
    assert controller.rejected[0, 3]
    assert not controller.accepted[0, 3]


@pytest.mark.filterwarnings("ignore:.*is unstable")  # gains chosen for easy arithmetic, not to fly
def test_wmsr_command_uses_the_kept_value_not_the_outlier():
    controller = controller_registry.create("wmsr_consensus", k_p=1.0, k_v=0.0, f=1)
    cfg = SimConfig(n_agents=4, dim=2, t_final=1, comm_radius=100)
    sim = Simulator(cfg, controller=controller,
                    init_pos=np.array([[0, 0], [1, 0], [2, 0], [100, 0]], dtype=float))
    u = controller.control(sim.sense())
    # Agent 0 at x = 0 hears 1, 2, 100.  All are larger, so W-MSR drops only the largest, then averages
    # what is left with its own value: (1 + 2 + 0) / 3.  It would be 100 if the indexing bug came back.
    assert u[0, 0] == pytest.approx(1.0)
    assert u[0, 1] == 0.0
    assert controller.accepted[0, 1] and controller.accepted[0, 2] and controller.rejected[0, 3]


def test_wmsr_converges_when_neighbour_ids_exceed_the_degree():
    cfg = SimConfig(n_agents=10, topology="complete", controller="wmsr_consensus")
    res = Simulator(cfg).run()
    assert position_spread_series(res)[-1] < 1e-3


# ---- aggregators ------------------------------------------------------------------------
def scalar(aggregator, own, values, ids=None):
    """Run a 1-D aggregation; ids default to 10, 11, 12, ... so they differ from positions."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    ids = np.arange(10, 10 + len(values)) if ids is None else np.asarray(ids)
    return aggregator.aggregate(np.array([float(own)]), values, ids)


def wmsr(f, **kw):
    return aggregator_registry.create("wmsr", f=f, **kw)


def test_both_aggregators_are_registered():
    assert {"mean", "wmsr"} <= set(aggregator_registry.names())
    assert isinstance(build_aggregator("mean", 1), MeanAggregator)
    assert isinstance(build_aggregator("wmsr", 2), WMSRAggregator) and build_aggregator("wmsr", 2).f == 2
    with pytest.raises(KeyError):
        build_aggregator("median", 1)
    with pytest.raises(ValueError):
        wmsr(-1)
    with pytest.raises(ValueError):
        wmsr(1, variant="nonsense")


def test_mean_aggregator_averages_everything_received():
    r = aggregator_registry.create("mean").aggregate(
        np.array([0.0, 0.0]), np.array([[1.0, 2.0], [3.0, 4.0], [100.0, -6.0]]), np.array([4, 7, 9]))
    assert r.value.tolist() == [104.0 / 3, 0.0]
    assert r.trusted.tolist() == [4, 7, 9] and r.rejected.size == 0 and r.count == 3


@pytest.mark.parametrize("name, build", [("mean", lambda: aggregator_registry.create("mean")),
                                         ("standard", lambda: wmsr(1)),
                                         ("trimmed", lambda: wmsr(1, variant="trimmed"))])
def test_empty_neighbourhood_returns_own_value(name, build):
    r = build().aggregate(np.array([3.0, -1.0]), np.empty((0, 2)), np.empty(0, dtype=int))
    assert r.value.tolist() == [3.0, -1.0]
    assert r.trusted.size == 0 and r.rejected.size == 0 and r.count == 0


def test_wmsr_drops_all_larger_values_when_there_are_at_most_f():
    r = scalar(wmsr(2), own=0.0, values=[1.0, 5.0, 0.0])  # two above, and f = 2
    assert r.rejected.tolist() == [10, 11] and r.trusted.tolist() == [12]
    assert r.value[0] == 0.0  # the mean of own and the one kept value


def test_wmsr_drops_only_the_f_largest_when_there_are_more():
    r = scalar(wmsr(1), own=0.0, values=[1.0, 5.0, 2.0])
    assert r.rejected.tolist() == [11] and r.trusted.tolist() == [10, 12]
    assert r.value[0] == pytest.approx((1.0 + 2.0 + 0.0) / 3)


def test_wmsr_is_symmetric_below_own_value():
    r = scalar(wmsr(1), own=0.0, values=[-1.0, -5.0, -2.0])
    assert r.rejected.tolist() == [11] and r.value[0] == pytest.approx(-1.0)


def test_wmsr_filters_each_side_relative_to_own_value_not_to_the_whole_list():
    # Unconditional trimming would also drop the 1, the smallest value, although it is above own.
    r = scalar(wmsr(1), own=0.0, values=[1.0, 2.0, 3.0])
    assert r.rejected.tolist() == [12] and r.trusted.tolist() == [10, 11]
    trimmed = scalar(wmsr(1, variant="trimmed"), own=0.0, values=[1.0, 2.0, 3.0])
    assert trimmed.rejected.tolist() == [10, 12] and trimmed.trusted.tolist() == [11]


def test_wmsr_always_keeps_values_equal_to_own():
    r = scalar(wmsr(1), own=2.0, values=[2.0, 2.0, 2.0])
    assert r.rejected.size == 0 and r.count == 3 and r.value[0] == 2.0


def test_wmsr_with_ties_at_the_extreme_drops_exactly_f_of_them():
    r = scalar(wmsr(1), own=0.0, values=[3.0, 3.0, 3.0])
    assert len(r.rejected) == 1 and r.count == 2
    assert r.value[0] == pytest.approx((3.0 + 3.0 + 0.0) / 3)


def test_wmsr_with_f_zero_keeps_everything():
    r = scalar(wmsr(0), own=0.0, values=[-9.0, 1.0, 50.0])
    assert r.rejected.size == 0 and r.value[0] == pytest.approx((-9.0 + 1.0 + 50.0 + 0.0) / 4)


def test_wmsr_can_leave_out_own_value():
    r = scalar(wmsr(1, include_self=False), own=0.0, values=[1.0, 5.0, 2.0])
    assert r.value[0] == pytest.approx(1.5)
    r = scalar(wmsr(1, include_self=False), own=7.0, values=[9.0])  # nothing survives
    assert r.value[0] == 7.0 and r.count == 0


def test_attacker_accepted_in_x_but_rejected_in_y_counts_as_rejected():
    own = np.array([0.0, 0.0])
    values = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 900.0]])  # agent 8 is mid-range in x only
    r = wmsr(1).aggregate(own, values, np.array([3, 5, 8]))
    assert 8 in r.rejected and 8 not in r.trusted
    assert r.value[0] == pytest.approx((1.0 + 1.5 + 0.0) / 3)  # x still uses agent 8's value
    assert r.value[1] == pytest.approx((1.0 + 2.0 + 0.0) / 3)
    assert r.count == len(r.trusted)
    assert sorted([*r.trusted.tolist(), *r.rejected.tolist()]) == [3, 5, 8]


@pytest.mark.parametrize("f", [1, 2])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_wmsr_output_stays_in_the_honest_hull_with_at_most_f_liars(f, dim):
    rng = np.random.default_rng(100 * f + dim)
    for _ in range(300):
        own = rng.uniform(-5, 5, size=dim)
        n_liars = rng.integers(0, f + 1)
        n_honest = rng.integers(2 * f, 2 * f + 6)
        honest = rng.uniform(-5, 5, size=(n_honest, dim))
        # Liars far outside, at a covert in-range value, or sitting exactly on an honest value.
        liars = [rng.uniform(-1e3, 1e3, size=(n_liars, dim)), rng.uniform(-6, 6, size=(n_liars, dim)),
                 honest[:n_liars]][rng.integers(3)]
        values = np.vstack([honest, liars])
        order = rng.permutation(len(values))
        value = wmsr(f).aggregate(own, values[order], order).value
        low = np.minimum(honest.min(axis=0), own)
        high = np.maximum(honest.max(axis=0), own)
        assert np.all(value >= low - 1e-12) and np.all(value <= high + 1e-12)


def test_plain_mean_leaves_the_honest_hull_with_one_liar():
    # The contrast that makes the hull test above mean something.
    value = aggregator_registry.create("mean").aggregate(
        np.zeros(1), np.array([[1.0], [2.0], [1e3]]), np.arange(3)).value
    assert value[0] > 2.0


def test_trimmed_variant_reproduces_the_original_controllers_filter():
    def original(own, values, ids, f=1):  # as WMSRConsensus._filter was before T2.1
        order = np.argsort(values)
        remove = set(order[:min(f, len(order))])
        remove.update(order[max(0, len(order) - f):])
        keep = [k for k in range(len(ids)) if k not in remove]
        return (values[keep].mean(), set(ids[keep].tolist())) if keep else (own, set())

    rng = np.random.default_rng(8)
    aggregator = wmsr(1, variant="trimmed", include_self=False)
    for _ in range(200):
        n = rng.integers(0, 7)
        own = rng.uniform(-5, 5)
        values = rng.uniform(-5, 5, size=n)
        ids = rng.permutation(20)[:n]
        old_value, old_kept = original(own, values, ids)
        r = aggregator.aggregate(np.array([own]), values.reshape(-1, 1), ids)
        assert r.value[0] == pytest.approx(old_value)
        assert set(r.trusted.tolist()) == old_kept
