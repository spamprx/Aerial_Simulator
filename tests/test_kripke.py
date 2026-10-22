"""The S5 engine: textbook muddy children, the S5 axioms on random models, nested knowledge by hand."""
import itertools

import numpy as np
import pytest

from aerial.epistemic.kripke import And, Atom, C, E, K, KripkeModel, Not, Or, implies


# ---- muddy children ----------------------------------------------------------------------------
CHILDREN = range(3)
MUDDY = [Atom(f"m{child}") for child in CHILDREN]


def muddy_model():
    """A world says which of the three children are muddy.  Each child sees the other two foreheads."""
    worlds = list(itertools.product((False, True), repeat=3))
    return KripkeModel(worlds, lambda child, w: tuple(x for other, x in enumerate(w) if other != child),
                       {f"m{child}": (lambda w, c=child: w[c]) for child in CHILDREN})


def knows_own_state(child):
    return Or(K(child, MUDDY[child]), K(child, Not(MUDDY[child])))


NOBODY_KNOWS = And(*(Not(knows_own_state(child)) for child in CHILDREN))
SOMEONE_MUDDY = Or(*MUDDY)


def rounds_of_questioning(rounds):
    """Father says someone is muddy, then asks ``rounds`` times, and each time nobody steps forward."""
    model = muddy_model().restrict(SOMEONE_MUDDY)
    for _ in range(rounds):
        model = model.restrict(NOBODY_KNOWS)
    return model


@pytest.mark.parametrize("world", [w for w in itertools.product((False, True), repeat=3) if any(w)])
def test_muddy_children_step_forward_on_the_round_the_book_says(world):
    """With k muddy children, the muddy ones know at round k, after k - 1 silences, and not before."""
    k = sum(world)
    for silences in range(k - 1):
        model = rounds_of_questioning(silences)
        assert model.holds(NOBODY_KNOWS, world)
    model = rounds_of_questioning(k - 1)
    for child in CHILDREN:
        if world[child]:
            assert model.holds(K(child, MUDDY[child]), world)
        else:  # a clean child sees k muddy ones, and cannot yet tell whether it is one of them
            assert not model.holds(knows_own_state(child), world)


@pytest.mark.parametrize("world", [w for w in itertools.product((False, True), repeat=3) if any(w)])
def test_clean_children_know_one_round_after_the_muddy_ones(world):
    k = sum(world)
    model = rounds_of_questioning(k - 1).restrict(Not(NOBODY_KNOWS))
    for child in CHILDREN:
        if not world[child]:
            assert model.holds(K(child, Not(MUDDY[child])), world)


def test_without_the_announcement_nobody_can_tell():
    model = muddy_model()
    assert not model.holds(knows_own_state(0), (True, True, True))
    assert not model.holds(knows_own_state(0), (True, False, False))  # it might be all clean


def test_the_announcement_is_what_lets_a_lone_muddy_child_know():
    world = (True, False, False)
    assert not muddy_model().holds(K(0, MUDDY[0]), world)
    assert muddy_model().restrict(SOMEONE_MUDDY).holds(K(0, MUDDY[0]), world)


def test_common_knowledge_of_the_announcement_comes_only_from_announcing_it():
    world = (True, True, True)
    # Before: everyone can see a muddy child, and yet it is not common knowledge that someone is muddy.
    before = muddy_model()
    assert before.holds(E(CHILDREN, SOMEONE_MUDDY), world)
    assert not before.holds(C(CHILDREN, SOMEONE_MUDDY), world)
    assert before.restrict(SOMEONE_MUDDY).holds(C(CHILDREN, SOMEONE_MUDDY), world)


# ---- S5, on random models ----------------------------------------------------------------------
def random_model(seed, n_worlds=24, n_agents=3, n_atoms=3):
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 4, size=(n_agents, n_worlds))
    truth = rng.random((n_atoms, n_worlds)) < 0.5
    return KripkeModel(range(n_worlds), lambda agent, w: int(states[agent, w]),
                       {f"p{a}": (lambda w, a=a: bool(truth[a, w])) for a in range(n_atoms)})


def random_formula(rng, depth=3):
    if depth == 0 or rng.random() < 0.2:
        return Atom(f"p{rng.integers(3)}")
    kind = rng.integers(5)
    child = random_formula(rng, depth - 1)
    if kind == 0:
        return Not(child)
    if kind == 1:
        return And(child, random_formula(rng, depth - 1))
    if kind == 2:
        return Or(child, random_formula(rng, depth - 1))
    if kind == 3:
        return K(int(rng.integers(3)), child)
    return C({int(rng.integers(3)), int(rng.integers(3))}, child)


@pytest.mark.parametrize("seed", range(10))
def test_the_s5_axioms_hold_on_random_models(seed):
    model = random_model(seed)
    rng = np.random.default_rng(seed)
    for _ in range(25):
        phi = random_formula(rng)
        for agent in range(3):
            knows = K(agent, phi)
            assert model.extension(implies(knows, phi)) == model.worlds                      # T
            assert model.extension(implies(knows, K(agent, knows))) == model.worlds          # 4
            assert model.extension(implies(Not(knows), K(agent, Not(knows)))) == model.worlds  # 5


def test_common_knowledge_is_the_limit_of_everyone_knowing():
    for seed in range(10):
        model = random_model(seed)
        group = {0, 1}
        phi = Atom("p0")
        level = E(group, phi)
        for _ in range(len(model.worlds)):
            level = E(group, level)
        assert model.extension(C(group, phi)) == model.extension(level)


def test_a_group_of_one_has_common_knowledge_exactly_when_that_agent_knows():
    model = random_model(0)
    assert model.extension(C({1}, Atom("p0"))) == model.extension(K(1, Atom("p0")))
    assert model.extension(E({1}, Atom("p0"))) == model.extension(K(1, Atom("p0")))


# ---- nested knowledge, by hand -----------------------------------------------------------------
def test_nested_knowledge_on_a_hand_built_example():
    """Worlds 1, 2, 3.  Agent a cannot tell 1 from 2, and agent b cannot tell 2 from 3.  p holds at 1, 2."""
    relation = {"a": {1: "x", 2: "x", 3: "y"}, "b": {1: "u", 2: "v", 3: "v"}}
    model = KripkeModel([1, 2, 3], lambda agent, w: relation[agent][w], {"p": lambda w: w in (1, 2)})
    p = Atom("p")
    assert model.extension(K("a", p)) == {1, 2}       # a's class {1, 2} is all p, and {3} is not
    assert model.extension(K("b", p)) == {1}          # b's class {2, 3} contains a world without p
    assert model.extension(K("a", K("b", p))) == frozenset()  # at 1, a must allow 2, where b does not know p
    assert model.extension(K("b", K("a", p))) == {1}  # at 1, b is sure of world 1, where a knows p
    assert model.extension(C({"a", "b"}, p)) == frozenset()  # 1, 2 and 3 are all connected


def test_knowledge_of_a_conjunction_is_the_conjunction_of_knowledge():
    model = random_model(3)
    a, b = Atom("p0"), Atom("p1")
    assert model.extension(K(0, And(a, b))) == model.extension(And(K(0, a), K(0, b)))


# ---- the engine itself -------------------------------------------------------------------------
def test_possible_is_the_class_of_the_world():
    model = KripkeModel([1, 2, 3], lambda agent, w: w // 2, {})
    assert model.possible("a", 2) == {2, 3}
    assert model.possible("a", 1) == {1}


def test_extensions_are_memoised():
    calls = []
    model = KripkeModel([1, 2, 3], lambda agent, w: w, {"p": lambda w: calls.append(w) or True})
    model.extension(And(Atom("p"), Atom("p"), K("a", Atom("p"))))
    model.holds(Atom("p"), 1)
    assert len(calls) == 3  # once per world, however often the atom appears


def test_formulas_compare_by_structure():
    assert And(Atom("p"), K(1, Atom("q"))) == And(Atom("p"), K(1, Atom("q")))
    assert E([1, 2], Atom("p")) == E({2, 1}, Atom("p"))
    assert hash(C([1], Atom("p"))) == hash(C({1}, Atom("p")))
    assert And(Atom("p")) != Or(Atom("p"))


def test_the_empty_conjunction_holds_and_the_empty_disjunction_fails():
    model = KripkeModel([1, 2], lambda agent, w: w, {})
    assert model.extension(And()) == {1, 2}
    assert model.extension(Or()) == frozenset()


def test_bad_input_is_rejected():
    model = KripkeModel([1], lambda agent, w: w, {"p": lambda w: True})
    with pytest.raises(KeyError, match="unknown atom"):
        model.extension(Atom("q"))
    with pytest.raises(KeyError, match="not a world"):
        model.holds(Atom("p"), 2)
    with pytest.raises(ValueError, match="must not be empty"):
        E([], Atom("p"))
    with pytest.raises(ValueError, match="must not be empty"):
        C([], Atom("p"))
    with pytest.raises(TypeError, match="not a formula"):
        model.extension("p")
