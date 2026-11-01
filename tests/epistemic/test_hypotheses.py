import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from aerial.epistemic.hypotheses import Hypotheses, mask_of
from aerial.epistemic.knowledge import Belief, Knowledge, TrustAll


def test_world_counts():
    assert Hypotheses(10, 1, 0).n_hypotheses == 10
    assert Hypotheses(20, 2, 3).n_hypotheses == 191


def test_unit_proof_exhausts_budget_at_f1():
    h = Hypotheses(5, 1, 0)
    h.add_unit(3)
    assert h.known_faulty().tolist() == [False, False, False, True, False]
    assert h.known_honest()[[1, 2, 4]].all() and h.budget_left() == 0


def test_framing_is_never_a_proof():
    h = Hypotheses(5, 1, 0)
    h.add_clause((4, 2))                       # liar 4 accuses honest 2
    assert not h.known_faulty().any()
    assert h.possibly_faulty()[[2, 4]].all()


def test_two_independent_accusers_prove_at_f1():
    h = Hypotheses(6, 1, 0)
    h.add_clause((1, 5))
    h.add_clause((2, 5))
    assert h.known_faulty()[5] and h.n_hypotheses == 1


def test_f2_needs_three_accusers():
    h = Hypotheses(8, 2, 0)
    h.add_clause((1, 7)); h.add_clause((2, 7))
    assert not h.known_faulty()[7]
    h.add_clause((3, 7))
    assert h.known_faulty()[7]


def test_empty_world_set_falls_back_to_own_observations():
    h = Hypotheses(4, 1, 0)
    h.add_unit(1); h.add_unit(2)                # more liars than F: assumption broken
    assert h.assumption_broken
    k = Knowledge(h)
    own = np.array([True, False])
    rep = np.array([0, mask_of([3])], dtype=np.uint64)
    assert k.holds(own, rep).tolist() == [True, False]


def test_knowledge_needs_an_honest_reporter():
    h = Hypotheses(5, 1, 0)
    k, b = Knowledge(h), Belief(h)
    rep = np.array([mask_of([1]), mask_of([1, 2]), 0], dtype=np.uint64)
    own = np.zeros(3, bool)
    assert k.holds(own, rep).tolist() == [False, True, False]   # one possibly-faulty reporter is not enough
    assert b.holds(own, rep).tolist() == [True, True, False]
    h.add_unit(2)                                                 # budget exhausted: 1 is known honest
    assert k.holds(own, rep).tolist() == [True, True, False]
    assert Belief(h).holds(own[:1], np.array([mask_of([2])], dtype=np.uint64)).tolist() == [False]
    assert TrustAll(h).holds(own[:1], np.array([mask_of([2])], dtype=np.uint64)).tolist() == [True]


def brute_force_known(N, F, me, constraints, reporters):
    worlds = [set(c) for k in range(F + 1) for c in itertools.combinations([j for j in range(N) if j != me], k)]
    ok = [w for w in worlds if all(any(j in w for j in con) for con in constraints)]
    rep = {j for j in range(N) if reporters >> j & 1}
    return bool(ok) and all(rep - w for w in ok)


@settings(max_examples=150, deadline=None)
@given(N=st.integers(3, 7), F=st.integers(0, 2), data=st.data())
def test_knowledge_matches_brute_force(N, F, data):
    me = 0
    h = Hypotheses(N, F, me)
    cons = []
    for _ in range(data.draw(st.integers(0, 3))):
        js = tuple(data.draw(st.lists(st.integers(1, N - 1), min_size=1, max_size=2, unique=True)))
        cons.append(js)
        h.add_clause(js)
    reps = data.draw(st.lists(st.integers(1, 2 ** N - 1), min_size=1, max_size=5))
    reps = [r & ~1 for r in reps]
    got = Knowledge(h).holds(np.zeros(len(reps), bool), np.array(reps, dtype=np.uint64))
    exp = [brute_force_known(N, F, me, cons, r) if r else False for r in reps]
    assert got.tolist() == exp


@settings(max_examples=100, deadline=None)
@given(N=st.integers(3, 8), F=st.integers(1, 2), data=st.data())
def test_true_faulty_set_never_eliminated_by_true_evidence(N, F, data):
    """Replay-style invariant: evidence produced by honest agents never removes S*."""
    me = 0
    liars = set(data.draw(st.lists(st.integers(1, N - 1), max_size=F, unique=True)))
    h = Hypotheses(N, F, me)
    for _ in range(data.draw(st.integers(0, 6))):
        kind = data.draw(st.sampled_from(["proof", "accuse", "framing"]))
        if kind == "proof" and liars:
            h.add_unit(data.draw(st.sampled_from(sorted(liars))))
        elif kind == "accuse" and liars:
            accuser = data.draw(st.integers(1, N - 1))
            if accuser not in liars:
                h.add_clause((accuser, data.draw(st.sampled_from(sorted(liars)))))
        elif kind == "framing" and liars:
            h.add_clause((data.draw(st.sampled_from(sorted(liars))), data.draw(st.integers(1, N - 1))))
    assert np.uint64(mask_of(liars)) in set(h.worlds.tolist())
    assert not (h.known_faulty() & ~np.isin(np.arange(N), list(liars))).any()
