"""The STL engine on short signals worked out by hand, then against the definition itself on random ones."""
import math
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest

from aerial.spec.stl import (Always, And, Deferred, Eventually, Not, Or, Pred, Until, _window_min, prefix_robustness,
                        robustness)

INF = float("inf")
X = [3.0, -1.0, 2.0, 0.0, 5.0, -2.0, 4.0]
Y = [-4.0, 1.0, -3.0, 2.0, -1.0, 6.0, 0.0]


def trace(n=7, dt=1.0):
    return SimpleNamespace(t=np.arange(n) * dt, cfg=SimpleNamespace(dt=dt))


def signal(values):
    return Pred(lambda _res: np.array(values, dtype=float))


def rho(phi, n=7, dt=1.0):
    return robustness(phi, trace(n, dt)).tolist()


def prefix(phi, n=7, dt=1.0):
    return prefix_robustness(phi, trace(n, dt)).tolist()


x, y = signal(X), signal(Y)


# ---- every operator, by hand -------------------------------------------------------------------
def test_pred_returns_the_margin_itself():
    assert rho(x) == X


def test_not_negates():
    assert rho(Not(x)) == [-3.0, 1.0, -2.0, 0.0, -5.0, 2.0, -4.0]


def test_and_takes_the_minimum_and_or_the_maximum():
    assert rho(And(x, y)) == [-4.0, -1.0, -3.0, 0.0, -1.0, -2.0, 0.0]
    assert rho(Or(x, y)) == [3.0, 1.0, 2.0, 2.0, 5.0, 6.0, 4.0]


def test_a_conjunction_of_nothing_holds_and_a_disjunction_of_nothing_fails():
    assert rho(And()) == [INF] * 7
    assert rho(Or()) == [-INF] * 7


def test_always_is_a_sliding_minimum_truncated_at_the_end():
    # min(3, -1, 2), min(-1, 2, 0), min(2, 0, 5), min(0, 5, -2), min(5, -2, 4), min(-2, 4), min(4)
    assert rho(Always(0, 2, x)) == [-1.0, -1.0, 0.0, -2.0, -2.0, -2.0, 4.0]


def test_always_with_a_delay_holds_vacuously_once_no_sample_is_left():
    # x[1..3], x[2..4], x[3..5], x[4..6], x[5..6], x[6], nothing
    assert rho(Always(1, 3, x)) == [-1.0, 0.0, -2.0, -2.0, -2.0, 4.0, INF]


def test_eventually_is_a_sliding_maximum_truncated_at_the_end():
    # max(3, -1, 2), max(-1, 2, 0), max(2, 0, 5), max(0, 5, -2), max(5, -2, 4), max(-2, 4), max(4)
    assert rho(Eventually(0, 2, x)) == [3.0, 2.0, 5.0, 5.0, 5.0, 4.0, 4.0]


def test_eventually_with_a_delay_fails_once_no_sample_is_left():
    # x[2..3], x[3..4], x[4..5], x[5..6], x[6], nothing, nothing
    assert rho(Eventually(2, 3, x)) == [2.0, 5.0, 5.0, 4.0, 4.0, -INF, -INF]


def test_until_by_hand():
    # k = 0: psi at 0 is -4; psi at 1 with phi at 0 gives min(1, 3) = 1; psi at 2 with phi at 0, 1
    # gives min(-3, 3, -1) = -3.  The best of the three is 1.  The other samples go the same way.
    assert rho(Until(0, 2, x, y)) == [1.0, 1.0, 2.0, 2.0, 5.0, 6.0, 0.0]


def test_until_with_a_delay_by_hand():
    # k = 1: psi at 2 with phi at 1 gives min(-3, -1) = -3; psi at 3 with phi at 1, 2 gives
    # min(2, -1, 2) = -1.  The last sample has no candidate left.
    assert rho(Until(1, 2, x, y)) == [1.0, -1.0, 2.0, 0.0, 5.0, -2.0, -INF]


def test_until_does_not_need_phi_at_the_witness_itself():
    phi, psi = signal([1.0, -5.0]), signal([-1.0, 2.0])
    assert rho(Until(0, 1, phi, psi), n=2)[0] == 1.0  # with phi required at the witness, it would be -5


def test_until_needs_phi_all_the_way_to_the_witness_even_during_the_delay():
    phi, psi = signal([1.0, -5.0, 1.0, 1.0, 1.0]), signal([-1.0, -1.0, -1.0, 2.0, -1.0])
    # psi holds at 3, inside the window [2, 4], but phi's -5 at 1 comes before every witness.
    assert rho(Until(2, 4, phi, psi), n=5)[0] == -5.0
    assert rho(Until(2, 4, signal([1.0] * 5), psi), n=5)[0] == 1.0


def test_nesting_by_hand():
    # F[0,1] x is [3, 2, 2, 5, 5, 4, 4], and G[0,1] of that is the smaller of each neighbouring pair.
    assert rho(Always(0, 1, Eventually(0, 1, x))) == [2.0, 2.0, 2.0, 5.0, 4.0, 4.0, 4.0]
    assert rho(Not(Always(0, 1, x))) == [1.0, 1.0, 0.0, 0.0, 2.0, 2.0, -4.0]


# ---- bounds in seconds -------------------------------------------------------------------------
def test_bounds_are_seconds_not_steps():
    assert rho(Always(0, 1, x), dt=0.5) == rho(Always(0, 2, x), dt=1.0)
    assert rho(Eventually(1.0, 1.5, x), dt=0.5) == rho(Eventually(2, 3, x), dt=1.0)


@pytest.mark.parametrize("arrival", [2, 3, 7, 29])
@pytest.mark.parametrize("dt", [1.0, 0.1, 0.05, 0.01])
def test_the_deadline_in_eventually_is_respected(arrival, dt):
    """Reached at step ``arrival``: exactly on the deadline counts, one step late does not."""
    margins = [-1.0] * arrival + [1.0] + [-1.0] * 2
    reach = signal(margins)
    t = arrival * dt  # for these values, t / dt lands on both sides of the whole number
    assert rho(Eventually(0, t, reach), n=len(margins), dt=dt)[0] == 1.0
    assert rho(Eventually(0, (arrival - 1) * dt, reach), n=len(margins), dt=dt)[0] == -1.0


def test_a_bound_between_samples_is_rounded_inwards():
    assert rho(Eventually(0, 2.9, x)) == rho(Eventually(0, 2, x))
    assert rho(Eventually(1.1, 3, x)) == rho(Eventually(2, 3, x))
    assert rho(Always(0.5, 2.5, x)) == rho(Always(1, 2, x))


def test_a_whole_number_of_steps_is_not_moved_by_floating_point():
    assert 0.3 / 0.1 < 3.0  # rounding down, a deadline of 0.3 s would miss step 3
    assert rho(Eventually(0, 0.3, signal([-1.0, -1.0, -1.0, 1.0, -1.0])), n=5, dt=0.1)[0] == 1.0
    assert rho(Eventually(0.3, 0.3, x), dt=0.1) == [0.0, 5.0, -2.0, 4.0, -INF, -INF, -INF]
    assert 0.07 / 0.01 > 7.0  # rounding up, a start of 0.07 s would skip step 7
    margins = [-1.0] * 7 + [1.0, -1.0]
    assert rho(Eventually(0.07, 0.07, signal(margins)), n=9, dt=0.01)[0] == 1.0
    assert rho(Until(0.07, 0.07, signal([1.0] * 9), signal(margins)), n=9, dt=0.01)[0] == 1.0


def test_an_infinite_bound_reaches_the_end_of_the_trace():
    assert rho(Always(0, INF, x)) == [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 4.0]
    assert rho(Eventually(0, INF, x)) == [5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0]
    assert rho(Always(0, 1000.0, x)) == rho(Always(0, INF, x))  # longer than the trace does the same


@pytest.mark.parametrize("a, b", [(-1, 2), (3, 2), (INF, INF), (float("nan"), 1), (0, float("nan"))])
def test_a_bad_interval_is_rejected(a, b):
    for build in (lambda: Always(a, b, x), lambda: Eventually(a, b, x), lambda: Until(a, b, x, y)):
        with pytest.raises(ValueError, match="0 <= a <= b"):
            build()


def test_an_interval_with_no_sample_in_it_is_an_error_not_an_empty_window():
    with pytest.raises(ValueError, match="no sample"):
        rho(Eventually(0.2, 0.8, x))


# ---- predicates --------------------------------------------------------------------------------
def test_a_predicate_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        rho(Pred(lambda _res: np.zeros((7, 2))))
    with pytest.raises(ValueError, match="shape"):
        rho(Pred(lambda _res: np.zeros(6)))


def test_a_boolean_predicate_is_rejected_rather_than_read_as_margins():
    with pytest.raises(TypeError, match="margin"):
        rho(Pred(lambda _res: np.zeros(7, dtype=bool)))


def test_a_predicate_is_computed_once_per_evaluation():
    calls = []
    counted = Pred(lambda _res: calls.append(1) or np.array(X))
    prefix(And(counted, Always(0, 2, counted)))
    assert len(calls) == 1


def test_the_result_does_not_alias_the_run():
    res = SimpleNamespace(t=np.arange(3.0), cfg=SimpleNamespace(dt=1.0), data=np.array([1.0, 2.0, 3.0]))
    robustness(Pred(lambda r: r.data), res)[0] = 99.0
    assert res.data[0] == 1.0


def test_nan_stays_in_the_windows_that_contain_it():
    gap = signal([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0])
    reaches_it = [True, True, True, False, False, False, False]  # only k <= 2 has sample 2 in [k, k + 2]
    assert np.isnan(robustness(Always(0, 2, gap), trace())).tolist() == reaches_it
    assert np.isnan(robustness(Eventually(0, 2, gap), trace())).tolist() == reaches_it
    assert np.isnan(robustness(Until(0, 2, gap, y), trace())[2])


def test_deferred_builds_from_the_run_being_evaluated():
    size = Deferred(lambda res: signal([float(len(res.t))] * len(res.t)))
    assert rho(size, n=3) == [3.0, 3.0, 3.0]
    assert rho(size, n=5) == [5.0] * 5  # one formula, used again on a different run


# ---- prefix robustness -------------------------------------------------------------------------
def test_prefix_of_always_is_the_running_minimum():
    assert prefix(Always(0, INF, x)) == [3.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0]
    margins = np.random.default_rng(0).normal(size=200)
    assert np.array_equal(prefix(Always(0, INF, signal(margins)), n=200), np.minimum.accumulate(margins))


def test_prefix_of_eventually_is_the_running_maximum_up_to_the_deadline():
    assert prefix(Eventually(0, INF, y)) == [-4.0, 1.0, 1.0, 2.0, 2.0, 6.0, 6.0]
    assert prefix(Eventually(0, 2, y)) == [-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 2 and 6 come too late


def test_prefix_before_a_window_opens():
    assert prefix(Always(1, 2, x)) == [INF, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    assert prefix(Eventually(1, 2, x)) == [-INF, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0]


def test_prefix_of_until_by_hand():
    assert prefix(Until(0, 1, x, y)) == [-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_prefix_of_a_nested_formula_evaluates_the_inner_window_on_the_prefix_too():
    # t = 0: F[0,1] at 0 sees only x0 = 3.  t = 1: G needs F at 1 too, which sees only x1 = -1.
    # t = 2: F at 1 now also sees x2 = 2, so rho rises again: a nested curve need not be monotone.
    assert prefix(Always(0, 1, Eventually(0, 1, x))) == [3.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0]


# ---- against the definition, on random signals -----------------------------------------------
def definition_always(v, lo, hi, k):
    window = v[k + lo:min(k + hi, len(v) - 1) + 1]
    return window.min() if len(window) else INF


def definition_eventually(v, lo, hi, k):
    window = v[k + lo:min(k + hi, len(v) - 1) + 1]
    return window.max() if len(window) else -INF


def definition_until(phi, psi, lo, hi, k):
    best = -INF
    for witness in range(k + lo, min(k + hi, len(psi) - 1) + 1):
        before = phi[k:witness].min() if witness > k else INF
        best = max(best, min(psi[witness], before))
    return best


def random_cases(count=400, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(1, 30))
        lo = int(rng.integers(0, n + 5))
        hi = lo + int(rng.integers(0, n + 10))
        yield rng.normal(size=n).round(2), rng.normal(size=n).round(2), lo, hi


def test_always_and_eventually_match_the_definition():
    for u, _, lo, hi in random_cases():
        res = trace(len(u))
        always, eventually = robustness(Always(lo, hi, signal(u)), res), robustness(Eventually(lo, hi, signal(u)), res)
        for k in range(len(u)):
            assert always[k] == definition_always(u, lo, hi, k)
            assert eventually[k] == definition_eventually(u, lo, hi, k)


def test_until_matches_the_definition():
    for u, v, lo, hi in random_cases():
        result = robustness(Until(lo, hi, signal(u), signal(v)), trace(len(u)))
        for k in range(len(u)):
            assert result[k] == definition_until(u, v, lo, hi, k)


def test_windows_of_every_width_on_one_signal():
    u = np.random.default_rng(1).normal(size=40)
    for lo in range(0, 42):
        for hi in range(lo, 45):
            result = robustness(Always(lo, hi, signal(u)), trace(40))
            assert result.tolist() == [definition_always(u, lo, hi, k) for k in range(40)]


def test_the_usual_identities():
    for u, v, lo, hi in random_cases(100, seed=2):
        res = trace(len(u))
        phi, psi = signal(u), signal(v)
        always = robustness(Always(lo, hi, phi), res)
        assert np.array_equal(always, robustness(Not(Eventually(lo, hi, Not(phi))), res))
        assert np.array_equal(robustness(Not(Not(phi)), res), u)
        assert np.array_equal(robustness(Not(And(phi, psi)), res), robustness(Or(Not(phi), Not(psi)), res))
        true = Pred(lambda r: np.full(len(r.t), INF))
        assert np.array_equal(robustness(Until(lo, hi, true, psi), res), robustness(Eventually(lo, hi, psi), res))


def test_the_end_of_a_prefix_is_the_value_at_time_zero():
    for u, v, lo, hi in random_cases(60, seed=3):
        for phi in (Always(lo, hi, signal(u)), Until(lo, hi, signal(u), signal(v)),
                    Or(Eventually(0, hi, signal(u)), Always(0, 1, Eventually(lo, hi, signal(v))))):
            assert prefix_robustness(phi, trace(len(u)))[-1] == robustness(phi, trace(len(u)))[0]


def test_prefix_matches_cutting_the_trace_by_hand():
    u, v = np.random.default_rng(4).normal(size=(2, 15))
    phi = lambda a, b: Until(1, 3, Always(0, 2, signal(a)), Eventually(0, 1, signal(b)))
    curve = prefix_robustness(phi(u, v), trace(15))
    for t in range(15):
        assert curve[t] == robustness(phi(u[:t + 1], v[:t + 1]), trace(t + 1))[0]


# ---- cost --------------------------------------------------------------------------------------
def test_the_cost_of_a_window_does_not_depend_on_its_width():
    """Checking each of these windows sample by sample would take about 10^10 comparisons."""
    n, width = 200_000, 100_000
    ramp = np.arange(n, dtype=float)
    start = time.perf_counter()
    result = _window_min(ramp, 0, width - 1)
    assert time.perf_counter() - start < 1.0
    assert np.array_equal(result, ramp)
    assert np.array_equal(_window_min(ramp[::-1], 0, width - 1)[:n - width], ramp[::-1][width - 1:n - 1])


def test_a_formula_built_from_module_level_functions_can_cross_to_a_worker():
    phi = Always(0, INF, And(Pred(math.sqrt), Eventually(0, 2, Pred(abs))))
    copy = pickle.loads(pickle.dumps(phi))
    assert copy.a == 0 and copy.b == INF and copy.child.children[0].fn is math.sqrt
