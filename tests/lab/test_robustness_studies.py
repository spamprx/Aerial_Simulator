import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aerial.lab import SimConfig, SimResult, Simulator, is_f_local

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import exp_degradation as degradation  # noqa: E402
import exp_er_robustness as er  # noqa: E402
import plot_robustness  # noqa: E402


def result(pos, compromised=(), rejected=None, adj=None):
    pos = np.array(pos, dtype=float)
    K, n = len(pos) - 1, pos.shape[1]
    mask = np.zeros(n, dtype=bool)
    mask[list(compromised)] = True
    adj = np.ones((K, n, n), dtype=bool) if adj is None else adj
    extras = {} if rejected is None else {"rejected_pos": rejected}
    return SimResult(SimConfig(n_agents=n, dim=pos.shape[2], dt=1.0, t_final=float(K)), np.arange(K + 1.0), pos, np.zeros_like(pos),
                     np.zeros((K, n, pos.shape[2])), adj, np.zeros(K), mask, extras)


# ---- T2.6 ------------------------------------------------------------------------------------
def test_overshoot_measures_honest_agents_leaving_their_starting_box():
    # Honest agents start in x = 0 .. 2; agent 0 reaches x = 3, and the liar's wandering does not count.
    res = result([[[0, 0], [2, 0], [50, 0]], [[3, 0], [2, 0], [90, 0]]], compromised=[2])
    assert er.measure_overshoot(res) == 1.0
    assert degradation.measure_overshoot(res) == 1.0
    assert er.measure_overshoot(result([[[0, 0], [2, 0]], [[1, 0], [1, 0]]])) == 0.0


def test_graphs_are_reproducible_and_may_be_disconnected():
    assert np.array_equal(er.sample_graph(10, 0.5, 3), er.sample_graph(10, 0.5, 3))
    assert not np.array_equal(er.sample_graph(10, 0.5, 3), er.sample_graph(10, 0.5, 4))
    assert any(er.graph_facts((10, 0.1, g))["max_r"] == 0 for g in range(5))  # sparse graphs stay in


def test_graph_facts_match_the_condition():
    for g in range(4):
        facts = er.graph_facts((10, 0.8, g))
        assert facts["robust_F1"] == (facts["max_r"] >= 3) and facts["robust_F2"] == (facts["max_r"] >= 5)


def test_compromised_sets_are_f_local_and_as_large_as_sampling_finds():
    rng = np.random.default_rng(0)
    sparse = er.sample_graph(20, 0.1, 0)
    mask = er.largest_f_local(sparse, 1, rng)
    assert mask.sum() > 1 and is_f_local(sparse, mask, 1)
    dense = np.ones((10, 10), dtype=bool) & ~np.eye(10, dtype=bool)
    assert er.largest_f_local(dense, 2, rng).sum() == 2  # on a complete graph F is all there can be


def test_the_full_study_covers_what_the_card_lists():
    assert er.FLEET_SIZES == (10, 15, 20) and er.BOUNDS == (1, 2) and er.GRAPHS_PER_CELL == 30
    assert er.EDGE_PROBABILITIES == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert set(er.ATTACKS) == {"constant_offset", "slow_drift", "random_spoof"}


def test_each_graph_runs_every_attack_with_a_control_only_on_some():
    facts = [er.graph_facts((10, 0.8, g)) for g in (0, 1)]
    cases = list(er.study_cases(facts, (1,)))
    filtered = [labels for labels, _ in cases if labels["filter"] == "wmsr"]
    controls = [labels for labels, _ in cases if labels["filter"] == "mean"]
    assert len(filtered) == 2 * 3 and len(controls) == 3 and {c["graph"] for c in controls} == {0}
    for labels, cfg in cases:
        Simulator(cfg)  # validates, including F-local on this exact graph
        assert cfg.F == 1 and cfg.topology == "custom"
    law = {labels["filter"]: cfg for labels, cfg in cases if labels["graph"] == 0 and labels["attack"] == "slow_drift"}
    assert Simulator(law["wmsr"]).controller.damping == Simulator(law["mean"]).controller.damping == "own"
    assert law["wmsr"].compromised_ids == law["mean"].compromised_ids and law["wmsr"].attack_params == law["mean"].attack_params


def er_row(F=1, r=3, holds=True, success=True, flt="wmsr", attack="slow_drift"):
    return {"F": F, "max_r": r, "condition_holds": holds, "success": success, "filter": flt, "attack": attack, "n": 10, "p": 0.5,
            "graph": 0, "final_spread": 0.0 if success else 5.0, "hull_overshoot": 0.0, "t_converge": 1.0}


def test_summary_separates_the_condition_and_lists_any_counterexample():
    rows = [er_row(), er_row(success=False), er_row(r=1, holds=False), er_row(flt="mean", success=False)]
    summary = er.summarise(rows)
    assert summary["F=1"]["condition_holds"] == {"runs": 2, "success_rate": 0.5}
    assert summary["F=1"]["condition_fails"] == {"runs": 1, "success_rate": 1.0}
    assert summary["unfiltered_control"] == {"runs": 1, "success_rate": 0.0}
    assert len(summary["F=1"]["failures_when_it_holds"]) == 1


def test_success_figure_is_written(tmp_path):
    rows = [er_row(r=r, holds=r >= 3, success=r >= 2) for r in (1, 2, 3, 4)] + [er_row(F=2, r=5), er_row(F=2, r=2, holds=False, success=False)]
    assert plot_robustness.plot_success(rows, tmp_path).stat().st_size > 1000


# ---- T2.7 ------------------------------------------------------------------------------------
def test_honest_discard_counts_only_honest_senders_only_while_disagreeing():
    # Three steps of three agents, with agent 2 the liar.  Spread is large for two steps, then zero.
    far, close = [[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]], [[1.0, 0.0], [1.0, 0.0], [5.0, 0.0]]
    rejected = np.zeros((3, 3, 3), dtype=bool)
    rejected[0, 0, 1] = True    # an honest report thrown away while disagreeing: counts
    rejected[0, 0, 2] = True    # a liar's report thrown away: does not
    rejected[2, 1, 0] = True    # thrown away after agreeing: does not
    adj = np.ones((3, 3, 3), dtype=bool)
    adj[:, np.arange(3), np.arange(3)] = False
    res = result([far, far, close, close], compromised=[2], rejected=rejected, adj=adj)
    # Two disagreeing steps with two honest links each: one of those four was discarded.
    assert degradation.measure_honest_discarded(res) == pytest.approx(0.25)


def test_honest_discard_is_zero_if_they_already_agree():
    same = [[1.0, 1.0], [1.0, 1.0]]
    assert degradation.measure_honest_discarded(result([same, same], rejected=np.zeros((1, 2, 2), dtype=bool))) == 0.0


def test_the_study_covers_what_the_card_lists_with_nested_liars_on_shared_graphs():
    assert degradation.ASSUMED == (0, 1, 2) and degradation.ACTUAL == (0, 1, 2, 3) and degradation.SEEDS >= 30
    cases = list(degradation.study_cases(1, n=10))
    assert len(cases) == 3 * 4 * 3
    same_seed = {(l["actual"], l["assumed_F"]): c for l, c in cases if l["attack"] == "slow_drift"}
    assert same_seed[(1, 0)].compromised_ids == same_seed[(1, 2)].compromised_ids  # paired across F
    assert same_seed[(2, 1)].compromised_ids[:1] == same_seed[(1, 1)].compromised_ids  # and nested across counts
    assert same_seed[(3, 0)].topology_params == same_seed[(0, 2)].topology_params
    for _, cfg in cases:
        assert cfg.allow_beyond_assumption


def test_the_filter_throws_away_honest_messages_even_with_nobody_lying():
    # This is the study's headline, so it gets checked directly: more F means more discarded, from zero.
    rows = {}
    for labels, cfg in degradation.study_cases(1, attacks=("constant_offset",), actual=(0,), n=10):
        rows[labels["assumed_F"]] = degradation.measure_honest_discarded(Simulator(cfg).run(n_steps=300))
    assert rows[0] == 0.0 and 0.0 < rows[1] < rows[2]


def test_degradation_figure_is_written(tmp_path):
    rows = [{"assumed_F": F, "actual": a, "final_spread": 10.0 ** (a - F), "hull_overshoot": float(a > F), "honest_discarded": 0.1 * F,
             "success": a <= F} for F in (0, 1, 2) for a in (0, 1, 2, 3)]
    assert plot_robustness.plot_degradation(rows, tmp_path).stat().st_size > 1000


def test_figures_can_be_redrawn_from_saved_rows_without_re_running_a_study(tmp_path):
    folder = tmp_path / "er_robustness" / "20260101-000000"
    folder.mkdir(parents=True)
    rows = [er_row(r=r, holds=r >= 3, success=r >= 2) for r in (1, 2, 3)]
    with open(folder / "rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    loaded = plot_robustness.latest_rows("er_robustness", tmp_path)
    assert loaded[0]["success"] is False and loaded[0]["max_r"] == 1 and loaded[0]["attack"] == "slow_drift"
    assert plot_robustness.plot_success(loaded, tmp_path).exists()


# ---- both scripts, end to end ---------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("script, name, figure", [("exp_er_robustness.py", "er_robustness", "er_robustness.png"),
                                                  ("exp_degradation.py", "degradation", "degradation.png")])
def test_the_script_writes_rows_a_summary_and_a_figure(tmp_path, script, name, figure):
    subprocess.run([sys.executable, str(SCRIPTS / script), "--quick", "--workers", "4", "--root", str(tmp_path)], check=True,
                   capture_output=True, text=True, env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"})
    folder = next((tmp_path / name).iterdir())
    for part in ("config.json", "grid.json", "env.json", "rows.csv", "summary.json"):
        assert (folder / part).exists()
    assert (tmp_path / "figures" / figure).exists()
    with open(folder / "rows.csv") as f:
        assert len(list(csv.DictReader(f))) > 0
    assert json.loads((folder / "summary.json").read_text())
