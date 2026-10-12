"""Do the robustness conditions predict whether W-MSR works?  Erdos-Renyi graphs (T2.6, proposal 2b).

    python scripts/exp_er_robustness.py            # the full study
    python scripts/exp_er_robustness.py --quick    # a small version, to check that the script works

For each sampled graph it records max_r and whether the graph is (2F + 1)-robust, then runs W-MSR consensus
against each attack with an F-local compromised set, as large as sampling can find.  Success means the
honest agents agree and never leave the box around where they started.  One graph in five also runs
with a plain mean, as a control showing these attacks do break an unfiltered fleet.  Writes the rows to
results/er_robustness/<time>/ and the figure to results/figures/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from aerial.lab import (SimConfig, is_rs_robust, make_topology, max_r, pick_compromised_f_local, record_experiment,
                    run_configs)
from aerial.lab.analysis.metrics import position_spread_series, time_to_converge

FLEET_SIZES = (10, 15, 20)
EDGE_PROBABILITIES = tuple(round(0.1 * k, 1) for k in range(1, 10))
BOUNDS = (1, 2)
GRAPHS_PER_CELL = 30
ATTACKS = ("constant_offset", "slow_drift", "random_spoof")
T_FINAL = 60.0
CONTROL_EVERY = 5  # one graph in this many also runs unfiltered
SPREAD_TOL = 0.1  # metres
HULL_TOL = 1e-6  # metres: round-off, not a real excursion


def attack_params(attack: str, rng: np.random.Generator) -> dict:
    heading = rng.normal(size=2)
    heading /= np.linalg.norm(heading)
    if attack == "constant_offset":
        return {"pos_offset": (15.0 * heading).tolist()}
    if attack == "slow_drift":
        return {"pos_rate": (1.0 * heading).tolist()}
    return {"pos_std": 15.0, "vel_std": 1.0}


def measure_spread(res) -> float:
    return float(position_spread_series(res)[-1])


def measure_t_converge(res) -> float:
    return time_to_converge(position_spread_series(res), res.t, SPREAD_TOL)


def measure_overshoot(res) -> float:
    """How far any honest agent got outside the box around where the honest agents started."""
    honest = res.pos[:, res.honest]
    low, high = honest[0].min(axis=0), honest[0].max(axis=0)
    return float(max((low - honest).max(), (honest - high).max(), 0.0))


METRICS = {"final_spread": measure_spread, "t_converge": measure_t_converge, "hull_overshoot": measure_overshoot}


def sample_graph(n: int, p: float, index: int) -> np.ndarray:
    """Disconnected graphs stay in: they are what r = 0 means."""
    return make_topology("erdos_renyi", n, np.random.default_rng([n, int(round(p * 10)), index]), p=p, ensure_connected=False)


def graph_facts(task) -> dict:
    n, p, index = task
    A = sample_graph(n, p, index)
    return {"n": n, "p": p, "graph": index, "max_r": max_r(A), "edges": int(A.sum()) // 2,
            **{f"robust_F{F}": is_rs_robust(A, 2 * F + 1) for F in BOUNDS}}


def largest_f_local(A: np.ndarray, F: int, rng: np.random.Generator) -> np.ndarray:
    """An F-local set as large as sampling finds, up to a quarter of the fleet: F-local bounds each
    neighbourhood, so sparse graphs can hold many more than F.  F itself always works."""
    for size in range(max(len(A) // 4, F), F, -1):
        try:
            return pick_compromised_f_local(A, size, F, rng, tries=200)
        except ValueError:
            continue
    return pick_compromised_f_local(A, F, F, rng)


def study_cases(facts, bounds):
    for fact in facts:
        A = sample_graph(fact["n"], fact["p"], fact["graph"])
        for F in bounds:
            for a, attack in enumerate(ATTACKS):
                rng = np.random.default_rng([fact["n"], int(round(fact["p"] * 10)), fact["graph"], F, a])
                compromised = np.flatnonzero(largest_f_local(A, F, rng)).tolist()
                labels = {**{k: fact[k] for k in ("n", "p", "graph", "max_r")}, "F": F, "attack": attack,
                          "n_compromised": len(compromised), "condition_holds": fact[f"robust_F{F}"]}
                settings = dict(n_agents=fact["n"], dim=2, t_final=T_FINAL, topology="custom", seed=fact["graph"], F=F,
                                topology_params={"adjacency": A.astype(int).tolist()}, compromised_ids=compromised,
                                attack=attack, attack_params=attack_params(attack, rng), record_extras=False,
                                record_lambda2=False)
                yield {**labels, "filter": "wmsr"}, SimConfig(controller="wmsr_consensus", **settings)
                if fact["graph"] % CONTROL_EVERY == 0:
                    # The same law and damping with the filter taken away, so only the filter differs.
                    yield {**labels, "filter": "mean"}, SimConfig(
                        controller="consensus", controller_params={"filter": "mean", "damping": "own"}, **settings)


def summarise(rows) -> dict:
    def rate(chosen):
        return {"runs": len(chosen), "success_rate": float(np.mean([r["success"] for r in chosen])) if chosen else None}

    summary = {"unfiltered_control": rate([r for r in rows if r["filter"] == "mean"])}
    for F in sorted({r["F"] for r in rows}):
        of_f = [r for r in rows if r["F"] == F and r["filter"] == "wmsr"]
        summary[f"F={F}"] = {
            "condition_holds": rate([r for r in of_f if r["condition_holds"]]),
            "condition_fails": rate([r for r in of_f if not r["condition_holds"]]),
            "unfiltered_when_it_holds": rate([r for r in rows if r["F"] == F and r["filter"] == "mean" and r["condition_holds"]]),
            "by_r": {str(r): rate([x for x in of_f if x["max_r"] == r]) for r in sorted({x["max_r"] for x in of_f})},
            "by_attack_when_it_fails": {a: rate([x for x in of_f if x["attack"] == a and not x["condition_holds"]]) for a in ATTACKS},
            "failures_when_it_holds": [{k: r[k] for k in ("n", "p", "graph", "attack", "final_spread", "hull_overshoot", "t_converge")}
                                       for r in of_f if r["condition_holds"] and not r["success"]][:20],
            "worst_overshoot_when_it_holds": max((r["hull_overshoot"] for r in of_f if r["condition_holds"]), default=None),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a small version, to check that the script works")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--root", default="results", help="where to write the results and figures")
    args = parser.parse_args(argv)

    sizes, probabilities, bounds, graphs = ((10,), (0.3, 0.6, 0.9), (1,), 4) if args.quick else \
        (FLEET_SIZES, EDGE_PROBABILITIES, BOUNDS, GRAPHS_PER_CELL)
    tasks = [(n, p, g) for n in sizes for p in probabilities for g in range(graphs)]
    print(f"checking {len(tasks)} graphs on {args.workers} workers")
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        facts = pool.map(graph_facts, tasks, chunksize=4)
    cases = list(study_cases(facts, bounds))
    print(f"{len(cases)} runs")
    results = run_configs([cfg for _, cfg in cases], METRICS, args.workers)
    rows = [{**labels, **result} for (labels, _), result in zip(cases, results)]
    for row in rows:
        row["success"] = bool(row["final_spread"] < SPREAD_TOL and row["hull_overshoot"] < HULL_TOL)
    summary = summarise(rows)

    grid = {"n": list(sizes), "p": list(probabilities), "F": list(bounds), "attack": list(ATTACKS), "graphs": graphs}
    path = record_experiment("er_robustness", SimConfig(controller="wmsr_consensus", t_final=T_FINAL), grid, range(graphs), rows, args.root)
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"rows and summary in {path}")
    print(json.dumps(summary, indent=1)[:3000])

    from plot_robustness import plot_success
    print(f"figure in {plot_success(rows, Path(args.root) / 'figures')}")


if __name__ == "__main__":
    main()
