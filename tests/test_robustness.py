import math
import time

import numpy as np
import pytest

from aerial.comms.graph import make_topology
from aerial.resilience.robustness import (is_rs_robust, is_rs_robust_bruteforce, max_r, max_s, resilience_conditions,
                                         robustness_witness)


def random_graph(rng, n=None, directed=None):
    n = int(rng.integers(2, 11)) if n is None else n
    directed = bool(rng.integers(2)) if directed is None else directed
    A = rng.random((n, n)) < rng.uniform(0.1, 0.95)
    np.fill_diagonal(A, False)
    if not directed:
        upper = np.triu(A, 1)
        A = upper | upper.T
    return A


def circulant(n, hops):
    A = np.zeros((n, n), dtype=bool)
    agents = np.arange(n)
    for hop in hops:
        A[agents, (agents + hop) % n] = True
        A[agents, (agents - hop) % n] = True
    return A


# ---- the card's "done when" ------------------------------------------------------------------
def test_fast_and_brute_force_agree_on_300_random_graphs():
    rng = np.random.default_rng(0)
    verdicts = []
    for _ in range(300):
        A = random_graph(rng)
        r, s = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        verdict = is_rs_robust(A, r, s)
        assert verdict == is_rs_robust_bruteforce(A, r, s), (A.astype(int).tolist(), r, s)
        verdicts.append(verdict)
    assert 50 < sum(verdicts) < 250  # both answers came up often enough for this to mean something


@pytest.mark.slow
def test_fast_and_brute_force_agree_for_every_r_and_s_on_the_largest_graphs():
    rng = np.random.default_rng(1)
    for _ in range(40):
        A = random_graph(rng, n=10)
        for r in range(1, 6):
            for s in range(1, 6):
                assert is_rs_robust(A, r, s) == is_rs_robust_bruteforce(A, r, s)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, pytest.param(20, marks=pytest.mark.slow)])
def test_complete_graph_is_exactly_half_n_robust(n):
    A = make_topology("complete", n)
    assert max_r(A) == math.ceil(n / 2)
    assert is_rs_robust(A, math.ceil(n / 2))
    assert not is_rs_robust(A, math.ceil(n / 2) + 1)


@pytest.mark.parametrize("n", [4, 5, 8, 12])
def test_ring_is_one_robust_and_not_two_robust(n):
    A = make_topology("ring", n)
    assert is_rs_robust(A, 1) and not is_rs_robust(A, 2) and max_r(A) == 1


@pytest.mark.parametrize("n", [2, 3, 6, 10])
def test_line_is_one_robust(n):
    A = make_topology("line", n)
    assert is_rs_robust(A, 1) and max_r(A) == 1


def test_a_twenty_node_graph_is_checked_in_under_thirty_seconds():
    A = random_graph(np.random.default_rng(3), n=20, directed=False)
    start = time.perf_counter()
    assert isinstance(is_rs_robust(A, 3, 2), bool)
    assert isinstance(max_r(A), int)
    assert time.perf_counter() - start < 30.0


# ---- other known facts and the rest of the interface -------------------------------------------
def test_disconnected_graph_is_not_even_one_robust():
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = True
    assert not is_rs_robust(A, 1) and max_r(A) == 0
    assert robustness_witness(A, 1) == ([0, 1], [2, 3]) or robustness_witness(A, 1) == ([2, 3], [0, 1])


def test_star_is_one_robust_and_not_two():
    A = make_topology("star", 6)
    assert max_r(A) == 1


def test_directed_graph_uses_in_neighbours_as_the_convention_says():
    # A[i, j]: i hears j.  Agents 0 and 1 hear nobody and agent 2 hears both, so {0} and {1} can never
    # be reached from outside: not 1-robust.  Reversed, agent 2 is heard by both and leads them: it is.
    A = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=bool)
    assert not is_rs_robust(A, 1) and not is_rs_robust_bruteforce(A, 1, 1)
    assert sorted(robustness_witness(A, 1)) == [[0], [1]]
    assert is_rs_robust(A.T, 1) and is_rs_robust_bruteforce(A.T, 1, 1)


@pytest.mark.parametrize("seed", range(8))
def test_a_witness_really_violates_the_definition(seed):
    rng = np.random.default_rng(seed)
    A = random_graph(rng, n=int(rng.integers(4, 10)))
    r, s = 2, 2
    pair = robustness_witness(A, r, s)
    if pair is None:
        assert is_rs_robust_bruteforce(A, r, s)
        return
    first, second = pair
    assert first and second and not set(first) & set(second)

    def reachable(group):
        return sum(1 for i in group if sum(1 for j in range(len(A)) if A[i, j] and j not in group) >= r)

    assert reachable(first) < len(first) and reachable(second) < len(second)
    assert reachable(first) + reachable(second) < s


@pytest.mark.parametrize("seed", range(6))
def test_robustness_is_monotone_so_bisection_is_valid(seed):
    A = random_graph(np.random.default_rng(10 + seed), n=8)
    by_r = [is_rs_robust(A, r) for r in range(0, 7)]
    assert by_r == sorted(by_r, reverse=True)
    by_s = [is_rs_robust(A, 1, s) for s in range(1, 9)]
    assert by_s == sorted(by_s, reverse=True)
    assert max_r(A) == sum(by_r) - 1
    assert max_s(A, 1) == sum(by_s)


def test_max_s_on_known_graphs():
    assert max_s(make_topology("complete", 6), 1) == 6
    assert max_s(make_topology("ring", 6), 1) >= 1
    assert max_s(make_topology("ring", 6), 2) == 0  # not even 2-robust


def test_zero_robust_holds_for_any_graph_and_bad_arguments_fail():
    assert is_rs_robust(np.zeros((4, 4), dtype=bool), 0)
    with pytest.raises(ValueError):
        is_rs_robust(make_topology("ring", 4), -1)
    with pytest.raises(ValueError):
        is_rs_robust(make_topology("ring", 4), 1, 0)
    with pytest.raises(ValueError):
        is_rs_robust_bruteforce(make_topology("complete", 13), 1, 1)


def test_a_graph_too_small_to_hold_two_sets_is_vacuously_robust():
    assert is_rs_robust(np.zeros((1, 1), dtype=bool), 5, 5)


@pytest.mark.parametrize("graph, F, f_local, f_total", [
    (make_topology("complete", 8), 1, True, True),    # 4-robust, and more than enough at F = 1
    (make_topology("complete", 8), 2, False, True),   # needs 5-robust: K_8 stops at 4
    (make_topology("ring", 8), 1, False, False),
    (make_topology("complete", 10), 2, True, True),
])
def test_resilience_conditions(graph, F, f_local, f_total):
    conditions = resilience_conditions(graph, F)
    assert conditions == {"(2F+1)-robust": f_local, "(F+1,F+1)-robust": f_total}


# ---- the graphs the rest of the project relies on ----------------------------------------------
def test_the_graph_in_the_wmsr_wiring_tests_is_at_least_three_robust():
    assert max_r(make_topology("complete", 8)) >= 3  # tests/test_filtered_primitives.py


def test_a_sparse_circulant_is_three_robust_too():
    # Six neighbours each rather than eleven, and still enough for F = 1.
    assert max_r(circulant(12, (1, 2, 3))) >= 3
