import numpy as np
import pytest

from aerial.core.config import Config
from aerial.experiments.runner import run_one, run_pair
from aerial.spec.mission_specs import grade


def small(**kw):
    c = Config(N=4).override(["world.scenario=open_field", "sim.T=40", "world.n_survivors=4",
                              "sim.record_level=standard"])
    return c.override([f"{k.replace('__', '.')}={v}" for k, v in kw.items()])


def test_honest_open_field_finds_survivors_safely():
    res, scn, sim = run_one(small(fleet="naive"))
    m = res.metrics
    assert m["collisions"] == 0 and m["terrain_hits"] == 0
    assert m["found"] >= 1 and m["coverage"] > 0.1
    g = grade(res, scn)
    assert g["rho_safe"] > 0


def test_gated_with_no_liar_never_accuses():
    res, _, _ = run_one(small(fleet="gated"))
    assert res.metrics["false_accusations"] == 0


def test_run_pair_asserts_pairing():
    out, _ = run_pair(small(sim__T=5), fleets=("naive", "gated"))
    assert set(out) == {"naive", "gated"}


def test_unpaired_controller_trips_assertion(monkeypatch):
    import aerial.agents.mission as mission
    orig = mission.MissionAgent.step

    def greedy(self, obs):
        if self.fleet == "gated":
            self.ctx.cfg  # noqa
            obs_rng = getattr(self, "_leak", None)
        return orig(self, obs)
    # an agent that draws from an environment stream breaks pairing
    import aerial.experiments.runner as runner
    real_build = runner.build_sim

    def build(cfg, scn=None):
        sim, scn = real_build(cfg, scn)
        if cfg.fleet == "gated":
            sim.rng.comm.random(3)
        return sim, scn
    monkeypatch.setattr(runner, "build_sim", build)
    with pytest.raises(AssertionError, match="pairing"):
        run_pair(small(sim__T=1), fleets=("naive", "gated"))


@pytest.mark.slow
@pytest.mark.parametrize("lie", ["range_aware", "fake_sweep_naive", "true_pose", "radio_shadow"])
@pytest.mark.parametrize("seed", range(3))
def test_soundness_under_attack(lie, seed):
    c = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "fleet=gated", "sim.T=150",
                           f"attack.position_lie={lie}", f"sim.seed={seed}", "sim.record_level=minimal"])
    res, scn, sim = run_one(c, stop_when_done=False)
    liar = np.flatnonzero(scn.liars)
    for a in sim.agents:
        if a.is_liar:
            continue
        S = np.uint64(sum(1 << int(j) for j in liar))
        assert S in set(a.hyp.worlds.tolist()), "the true faulty set was eliminated"
        assert not (a.hyp.known_faulty() & ~scn.liars).any(), "an honest drone was proven faulty"
        known = a.cov.cleared(a.std_done)
        surv = scn.survivor_cells
        bad = known[surv] & ~(a.cov.survivor_own[surv] | a.cov.survivor_rep[surv].astype(bool))
        assert not bad.any(), "a survivor cell is 'known' empty"
