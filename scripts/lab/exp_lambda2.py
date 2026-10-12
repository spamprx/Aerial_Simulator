"""How convergence speed varies with the algebraic connectivity lambda_2(L) (T1.9, proposal 2a).

    python scripts/exp_lambda2.py            # the full study: a few minutes on a multi-core machine
    python scripts/exp_lambda2.py --quick    # a small version, to check that the script works

Summed second-order consensus, with no saturation, over standard topologies, with and without packet
loss.  Writes the rows and a record of the run to results/lambda2/<time>/, and the figures to
results/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from aerial.lab import SimConfig, algebraic_connectivity, make_topology, record_experiment, run_configs
from aerial.lab.analysis.metrics import decay_rate, position_spread_series, time_to_converge
from aerial.lab.stability import fleet_rate, predicted_rate

K_P, K_V, DT = 1.0, 1.5, 0.05
FIXED = ("ring", "line", "grid", "star", "complete")
ER_PROBABILITIES = (0.2, 0.4, 0.6)
DROP_PROBABILITIES = (0.0, 0.1, 0.3)
FLEET_SIZES = (10, 20)
SEEDS = 10
CONVERGED_TOL = 0.1  # metres of position spread, as in the default t_converge_pos metric


def measure_rate(res) -> float:
    return decay_rate(position_spread_series(res), res.t)


def measure_t_converge(res) -> float:
    return time_to_converge(position_spread_series(res), res.t, CONVERGED_TOL)


def measure_lambda2(res) -> float:
    """The mean of lambda_2 over the run: with no loss, simply the graph's own value."""
    return float(np.nanmean(res.lambda2))


def measure_fleet_rate(res) -> float:
    """The rate the graph's whole spectrum predicts.  Packets drop independently on every link, so
    on average each neighbour counts for (1 - p) of itself: L becomes (1 - p) * L of the full graph."""
    full_graph = res.adj.any(axis=0)
    return fleet_rate(K_P, K_V, DT, full_graph, scale=1.0 - res.cfg.drop_prob)


METRICS = {"rate": measure_rate, "t_converge": measure_t_converge, "lambda2": measure_lambda2,
           "fleet_rate": measure_fleet_rate}


def run_length(topology: str, params: dict, n: int, drop_prob: float) -> float:
    """Long enough for the slowest mode to fall well below the transient, and no longer.  Random
    graphs have no fixed spectrum to ask, but they are well connected, so they get a fixed length."""
    if topology == "erdos_renyi":
        return 150.0 / (1.0 - drop_prob)
    rate = fleet_rate(K_P, K_V, DT, make_topology(topology, n), scale=1.0 - drop_prob)
    return float(np.clip(14.0 / rate, 40.0, 1500.0))


def study_cases(quick: bool):
    sizes, drops, seeds = ((10,), (0.0, 0.3), 2) if quick else (FLEET_SIZES, DROP_PROBABILITIES, SEEDS)
    probabilities = (0.4,) if quick else ER_PROBABILITIES
    for n in sizes:
        for drop_prob in drops:
            shapes = [(name, name, {}) for name in FIXED]
            shapes += [(f"ER p={p}", "erdos_renyi", {"p": p}) for p in probabilities]
            for label, topology, params in shapes:
                for seed in range(seeds):
                    yield {"n": n, "drop_prob": drop_prob, "graph": label, "seed": seed}, SimConfig(
                        n_agents=n, dim=3, dt=DT, topology=topology, topology_params=params, drop_prob=drop_prob,
                        seed=seed, t_final=run_length(topology, params, n, drop_prob), record_extras=False,
                        controller="consensus", controller_params={"k_p": K_P, "k_v": K_V})


def agreement(rows, predicted_key: str):
    """Mean and worst relative difference between measured and predicted rates."""
    errors = [abs(r["rate"] - r[predicted_key]) / r[predicted_key] for r in rows if np.isfinite(r["rate"])]
    return {"runs": len(errors), "mean_error": float(np.mean(errors)), "worst_error": float(np.max(errors))}


def summarise(rows) -> dict:
    """The numbers the report quotes."""
    for row in rows:
        row["lambda2_rate"] = predicted_rate(K_P, K_V, DT, row["lambda2"])
    summary = {"gains": {"k_p": K_P, "k_v": K_V, "dt": DT}, "overdamped_limit": K_P / K_V}
    for drop_prob in sorted({r["drop_prob"] for r in rows}):
        chosen = [r for r in rows if r["drop_prob"] == drop_prob]
        summary[f"drop_{drop_prob}"] = {
            "whole_spectrum": agreement(chosen, "fleet_rate"),
            "lambda2_alone": agreement(chosen, "lambda2_rate"),
            "by_graph": _by_graph(chosen),
        }
    return summary


def _by_graph(rows) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["n"], row["graph"])].append(row)
    return {f"N={n} {graph}": {
        "lambda2": float(np.mean([r["lambda2"] for r in group])),
        "measured_rate": float(np.nanmean([r["rate"] for r in group])),
        "whole_spectrum": float(np.mean([r["fleet_rate"] for r in group])),
        "lambda2_alone": float(np.mean([r["lambda2_rate"] for r in group])),
        "t_converge": float(np.mean([r["t_converge"] for r in group])),
    } for (n, graph), group in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a small version, to check that the script works")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--root", default="results", help="where to write the results and figures")
    args = parser.parse_args(argv)

    cases = list(study_cases(args.quick))
    print(f"{len(cases)} runs on {args.workers} workers")
    results = run_configs([cfg for _, cfg in cases], METRICS, args.workers)
    rows = [{**labels, **result} for (labels, _), result in zip(cases, results)]
    summary = summarise(rows)

    base = SimConfig(dt=DT, controller="consensus", controller_params={"k_p": K_P, "k_v": K_V})
    grid = {"n": sorted({r["n"] for r in rows}), "drop_prob": sorted({r["drop_prob"] for r in rows}),
            "graph": sorted({r["graph"] for r in rows})}
    path = record_experiment("lambda2", base, grid, sorted({r["seed"] for r in rows}), rows, args.root)
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"rows and summary in {path}")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "by_graph"} for key, value in summary.items()
                      if key.startswith("drop_")}, indent=1))

    # Imported here so the experiment itself runs without a plotting stack.
    from plot_lambda2 import plot_all
    for figure in plot_all(rows, Path(args.root) / "figures", K_P, K_V, DT):
        print(f"figure in {figure}")


if __name__ == "__main__":
    main()
