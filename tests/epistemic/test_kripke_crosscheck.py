"""T3.11: the fast knowledge code agrees with an explicit S5 Kripke model on small random instances.

Worlds = (faulty set S, truth value of one claimed fact c). An honest reporter's claim of c is true, so worlds where
c is false must have every reporter in S. Agent i cannot tell apart worlds that agree with its evidence; the ported
engine then evaluates K_i(c) and K_i(j faulty) by set containment.
"""
import itertools

import numpy as np
import pytest

from aerial.epistemic.hypotheses import Hypotheses
from aerial.epistemic.kripke import Atom, K, KripkeModel
from aerial.epistemic.knowledge import Knowledge
from aerial.epistemic.nested import log10_worlds


def explicit(N, F, me, clauses, reporters):
    others = [j for j in range(N) if j != me]
    worlds = []
    for k in range(F + 1):
        for S in itertools.combinations(others, k):
            S = frozenset(S)
            if not all(any(j in S for j in cl) for cl in clauses):
                continue
            for c in (True, False):
                if not c and not reporters <= S:       # an honest reporter cannot claim a false fact
                    continue
                worlds.append((S, c))
    val = {"c": lambda w: w[1]}
    for j in range(N):
        val[f"f{j}"] = (lambda jj: (lambda w: jj in w[0]))(j)
    return KripkeModel(worlds, local_state=lambda a, w: 0 if a == me else w, valuation=val), worlds


@pytest.mark.parametrize("seed", range(300))
def test_fast_knowledge_equals_kripke(seed):
    rng = np.random.default_rng(seed)
    N, F, me = int(rng.integers(3, 6)), int(rng.integers(0, 3)), 0
    clauses = [tuple(rng.choice(np.arange(1, N), size=int(rng.integers(1, 3)), replace=False))
               for _ in range(int(rng.integers(0, 3)))]
    reporters = frozenset(int(j) for j in rng.choice(np.arange(1, N), size=int(rng.integers(1, N)), replace=False))
    model, worlds = explicit(N, F, me, clauses, reporters)
    h = Hypotheses(N, F, me)
    for cl in clauses:
        h.add_clause(cl)
    if not worlds:
        assert h.assumption_broken
        return
    actual = worlds[0]
    mask = np.uint64(sum(1 << j for j in reporters))
    assert Knowledge(h).holds(np.array([False]), np.array([mask]))[0] == model.holds(K(me, Atom("c")), actual)
    for j in range(1, N):
        assert h.known_faulty()[j] == model.holds(K(me, Atom(f"f{j}")), actual)


def test_world_count_exact_small():
    h = Hypotheses(4, 1, 0)
    own = np.array([False, False, True])
    rep = np.array([0b0010, 0b0110, 0], dtype=np.uint64)
    # S={}: u=1 (cell 0? no: reporter 1 outside S) -> cells uncertain: none of the reported ones; cell with rep 0 -> u
    brute = 0
    for S in [0, 0b0010, 0b0100, 0b1000]:
        u = sum(1 for o, r in zip(own, rep) if not o and (int(r) & ~S) == 0)
        brute += 2 ** u
    assert np.isclose(10 ** log10_worlds(h, own, rep), brute)
