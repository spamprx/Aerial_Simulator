"""Nested knowledge K_i K_j Safe_j: what agent i knows about what agent j knows.

Agent i cannot look inside j, so it goes by what j says, and that is worth something only from an agent i knows to be
honest.  This is the approximation through reported knowledge that the gate uses:

    K_i K_j Safe_j  is taken to hold  <=>  j is known honest to i, i holds j's payload, and that payload says k_safe[j]

It is sound.  If i knows j is honest and j says it knows, then in every world i considers possible j is telling the
truth, so j does know.  So a liar's forged k_safe = True proves nothing until i knows the liar is honest, which it never
will.  It is not complete: i could sometimes work out what j knows without being told.  ``compare_on_small_instance``
measures how much that loses, against the exact value from the T4.1 engine.  The payload arrives one step late, so
what i reads is what j knew on the step before.
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np

from .kripke import And, Atom, K, KripkeModel


def nested_safe_row(i: int, known_honest: np.ndarray, obs, k_safe_i: bool) -> np.ndarray:
    """(N,) K_i K_j Safe_j for each j, by reported knowledge.  On the diagonal K_i K_i phi is K_i phi, as S5 has it."""
    says_safe = obs.payload_for(i, "k_safe")
    row = np.zeros(obs.n, dtype=bool) if says_safe is None else known_honest & obs.adj[i] & says_safe
    row[i] = k_safe_i
    return row


def compare_on_small_instance(n_agents: int = 3, n_cells: int = 5, F: int = 1, separation: int = 2,
                              range_bucket: int = 2) -> Dict[str, int]:
    """How often reported knowledge gives the exact K_i K_j Safe_j, over every world of a model small enough to list.

    Agents sit in cells on a line.  A world says who is faulty (at most F), where everyone truly is, what each reports (an
    honest agent reports the truth), and what each says about its own safety (an honest agent says what it knows).  An
    agent sees its own cell, every report, every claim, and each neighbour's distance rounded down to ``range_bucket``.
    Safe_j means nobody is within ``separation`` of j.  Counts cover honest i and every j other than i.
    """
    agents = range(n_agents)
    cells = range(n_cells)

    def sees(a, world):
        _, true, reports = world[:3]
        return (true[a], reports, tuple(abs(true[a] - true[b]) // range_bucket for b in agents))

    def is_safe(a, world):
        true = world[1]
        return all(abs(true[a] - true[b]) >= separation for b in agents if b != a)

    base_worlds = []
    for size in range(F + 1):
        for faulty in itertools.combinations(agents, size):
            for true in itertools.product(cells, repeat=n_agents):
                lies = itertools.product(cells, repeat=size)
                for lie in lies:
                    reports = list(true)
                    for agent, cell in zip(faulty, lie):
                        reports[agent] = cell
                    base_worlds.append((frozenset(faulty), true, tuple(reports)))
    safe_atoms = {f"safe{a}": (lambda world, a=a: is_safe(a, world)) for a in agents}
    base = KripkeModel(base_worlds, sees, safe_atoms)
    knows_own = [base.extension(K(a, Atom(f"safe{a}"))) for a in agents]

    # Now add what each agent says.  An honest agent's claim is fixed by what it knows; a faulty one can say either.
    worlds = []
    for world in base_worlds:
        faulty = world[0]
        options = [(False, True) if a in faulty else (world in knows_own[a],) for a in agents]
        worlds.extend((*world, claims) for claims in itertools.product(*options))
    atoms = {**safe_atoms, **{f"honest{a}": (lambda w, a=a: a not in w[0]) for a in agents},
             **{f"says{a}": (lambda w, a=a: w[3][a]) for a in agents}}
    model = KripkeModel(worlds, lambda a, world: (*sees(a, world), world[3]), atoms)

    counts = {"total": 0, "agree": 0, "both_true": 0, "reported_but_not_exact": 0, "exact_but_not_reported": 0}
    for i, j in itertools.permutations(agents, 2):
        exact = model.extension(K(i, K(j, Atom(f"safe{j}"))))
        reported = model.extension(And(K(i, Atom(f"honest{j}")), Atom(f"says{j}")))
        honest_i = model.extension(Atom(f"honest{i}"))
        counts["total"] += len(honest_i)
        counts["both_true"] += len(exact & reported & honest_i)
        counts["reported_but_not_exact"] += len((reported - exact) & honest_i)
        counts["exact_but_not_reported"] += len((exact - reported) & honest_i)
    counts["agree"] = counts["total"] - counts["reported_but_not_exact"] - counts["exact_but_not_reported"]
    return counts
