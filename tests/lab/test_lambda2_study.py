import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aerial.lab import SimConfig, Simulator, make_topology
from aerial.lab.stability import fleet_rate, predicted_rate

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import exp_lambda2 as study  # noqa: E402
import plot_lambda2  # noqa: E402

K_P, K_V, DT = study.K_P, study.K_V, study.DT


def consensus_run(topology="ring", n=6, t_final=120.0, drop_prob=0.0):
    cfg = SimConfig(n_agents=n, dim=2, dt=DT, topology=topology, t_final=t_final, drop_prob=drop_prob, record_extras=False,
                    controller_params={"k_p": K_P, "k_v": K_V})
    return Simulator(cfg).run()


def test_metrics_on_a_graph_with_a_known_spectrum():
    res = consensus_run()
    assert study.measure_lambda2(res) == pytest.approx(1.0)  # a ring of 6: 2 - 2 cos(2 pi / 6)
    assert study.measure_fleet_rate(res) == pytest.approx(fleet_rate(K_P, K_V, DT, make_topology("ring", 6)))
    assert study.measure_rate(res) == pytest.approx(study.measure_fleet_rate(res), rel=0.05)
    assert 0.0 < study.measure_t_converge(res) < 120.0


def test_fleet_rate_metric_with_packet_loss_uses_the_average_graph():
    res = consensus_run(drop_prob=0.3, t_final=20.0)
    assert res.adj.any(axis=0).sum() == make_topology("ring", 6).sum()  # every link shows up at some point
    assert study.measure_fleet_rate(res) == pytest.approx(fleet_rate(K_P, K_V, DT, make_topology("ring", 6), scale=0.7))


def test_scaling_the_laplacian_scales_every_eigenvalue():
    # Half of every link on a ring of 8 has the spectrum of the ring with lambda halved.
    ring = make_topology("ring", 8)
    eigenvalues = np.linalg.eigvalsh(np.diag(ring.sum(axis=1)) - ring.astype(float))[1:]
    expected = min(predicted_rate(K_P, K_V, DT, 0.5 * lam) for lam in eigenvalues)
    assert fleet_rate(K_P, K_V, DT, ring, scale=0.5) == pytest.approx(expected)


def test_run_length_is_long_enough_for_slow_graphs_and_short_for_fast_ones():
    slow, fast = study.run_length("line", {}, 20, 0.0), study.run_length("complete", {}, 20, 0.0)
    assert slow > 10.0 * fast
    assert slow * fleet_rate(K_P, K_V, DT, make_topology("line", 20)) >= 13.9  # about e^-14 of the start
    assert study.run_length("line", {}, 20, 0.3) > slow  # loss slows it down
    assert study.run_length("erdos_renyi", {"p": 0.2}, 10, 0.3) > study.run_length("erdos_renyi", {"p": 0.2}, 10, 0.0)


def test_the_full_study_covers_what_the_card_lists():
    cases = list(study.study_cases(quick=False))
    assert len(cases) == 2 * 3 * 8 * 10
    assert {labels["n"] for labels, _ in cases} == {10, 20}
    assert {labels["drop_prob"] for labels, _ in cases} == {0.0, 0.1, 0.3}
    assert {labels["graph"] for labels, _ in cases} == {"ring", "line", "grid", "star", "complete", "ER p=0.2",
                                                        "ER p=0.4", "ER p=0.6"}
    assert {labels["seed"] for labels, _ in cases} == set(range(10))
    for _, cfg in cases:
        assert cfg.max_speed is None and cfg.max_accel is None and cfg.controller == "consensus"


def row(n=10, graph="ring", drop_prob=0.0, lam=0.5, rate=0.3, predicted=0.3):
    return {"n": n, "graph": graph, "drop_prob": drop_prob, "seed": 0, "lambda2": lam, "rate": rate, "fleet_rate": predicted,
            "t_converge": 10.0}


def test_summary_reports_the_agreement_per_loss_rate():
    rows = [row(rate=0.33, predicted=0.3), row(rate=0.27, predicted=0.3), row(rate=float("nan")),
            row(drop_prob=0.3, rate=0.3, predicted=0.3)]
    summary = study.summarise(rows)
    assert summary["overdamped_limit"] == pytest.approx(K_P / K_V)
    assert summary["drop_0.0"]["whole_spectrum"]["mean_error"] == pytest.approx(0.1)
    assert summary["drop_0.0"]["whole_spectrum"]["runs"] == 2  # the run that never decayed is left out
    assert summary["drop_0.3"]["whole_spectrum"]["mean_error"] == pytest.approx(0.0)
    assert "N=10 ring" in summary["drop_0.0"]["by_graph"]


@pytest.mark.slow
def test_figures_are_written_even_with_only_part_of_the_study(tmp_path):
    rows = [row(lam=lam, rate=lam * 0.7, predicted=lam * 0.7) for lam in (0.1, 0.2, 0.4)]
    rows += [row(n=20, lam=lam, rate=lam * 0.7, predicted=lam * 0.7, graph="line") for lam in (0.1, 0.2)]
    for r in rows:
        r["lambda2_rate"] = r["fleet_rate"]
    figures = plot_lambda2.plot_all(rows, tmp_path, K_P, K_V, DT)
    assert [f.name for f in figures] == ["lambda2_convergence.png", "lambda2_agreement.png"]
    assert all(f.stat().st_size > 1000 for f in figures)
    lossy = [{**r, "drop_prob": 0.1} for r in rows]
    assert plot_lambda2.plot_packet_loss(rows + lossy, tmp_path, K_P, K_V, DT).exists()


@pytest.mark.slow
def test_the_script_runs_end_to_end_and_its_results_say_what_the_card_asks(tmp_path):
    subprocess.run([sys.executable, str(SCRIPTS / "exp_lambda2.py"), "--quick", "--workers", "4", "--root", str(tmp_path)],
                   check=True, capture_output=True, text=True, env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"})
    folder = next((tmp_path / "lambda2").iterdir())
    for name in ("config.json", "grid.json", "env.json", "rows.csv", "summary.json"):
        assert (folder / name).exists()
    for name in ("lambda2_convergence.png", "lambda2_agreement.png", "lambda2_packet_loss.png"):
        assert (tmp_path / "figures" / name).exists()
    with open(folder / "rows.csv") as f:
        rows = [r for r in csv.DictReader(f) if float(r["drop_prob"]) == 0.0]
    sparse = sorted((float(r["lambda2"]), float(r["rate"])) for r in rows if float(r["lambda2"]) < 0.9)
    rates = [rate for _, rate in sparse]
    assert len(rates) >= 4 and rates[-1] > 2.0 * rates[0]  # speeding up as lambda_2 grows
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["drop_0.0"]["whole_spectrum"]["mean_error"] < 0.05
