"""What breaks as the attack outgrows the assumed bound, and what the filter costs (T2.7, proposal 2b).

    python scripts/exp_degradation.py            # the full study
    python scripts/exp_degradation.py --quick    # a small version, to check that the script works

W-MSR consensus at each assumed F in {0, 1, 2} against each actual number compromised in {0, 1, 2, 3},
on the same graphs and with the same liars.  Measures the final honest spread, the convergence time,
how far honest agents leave their starting box, and the fraction of honest messages W-MSR throws away:
the "filtering makes it ignorant" number.  Writes to results/degradation/<time>/ and results/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from aerial.lab import SimConfig, make_topology, record_experiment, run_configs
from aerial.lab.analysis.metrics import position_spread_series, time_to_converge

N, EDGE_PROBABILITY = 20, 0.6
ASSUMED = (0, 1, 2)
ACTUAL = (0, 1, 2, 3)
ATTACKS = ("constant_offset", "slow_drift", "random_spoof")
SEEDS = 30
T_FINAL = 60.0
SPREAD_TOL = 0.1  # metres
HULL_TOL = 1e-6  # metres


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
    honest = res.pos[:, res.honest]
    low, high = honest[0].min(axis=0), honest[0].max(axis=0)
    return float(max((low - honest).max(), (honest - high).max(), 0.0))


def measure_honest_discarded(res) -> float:
    """Of the honest reports honest agents received while they still disagreed, the fraction W-MSR threw
    away.  Afterwards there is nothing to learn, so those steps would only dilute the number."""
    disagreeing = position_spread_series(res)[:-1] > SPREAD_TOL
    if not disagreeing.any():
        return 0.0
    honest = res.honest
    pairs = np.ix_(np.flatnonzero(disagreeing), np.flatnonzero(honest), np.flatnonzero(honest))
    received = res.adj[pairs].sum()
    return float(res.extras["rejected_pos"][pairs].sum() / received) if received else 0.0


METRICS = {"final_spread": measure_spread, "t_converge": measure_t_converge, "hull_overshoot": measure_overshoot,
           "honest_discarded": measure_honest_discarded}


def study_cases(seeds: int, assumed=ASSUMED, actual=ACTUAL, attacks=ATTACKS, n: int = N):
    for seed in range(seeds):
        A = make_topology("erdos_renyi", n, np.random.default_rng([seed, 1]), p=EDGE_PROBABILITY)
        for a, attack in enumerate(attacks):
            rng = np.random.default_rng([seed, 2, a])
            order = rng.permutation(n)  # nested, so 2 liars are the first liar plus one more
            params = attack_params(attack, rng)
            for count in actual:
                for F in assumed:
                    labels = {"seed": seed, "attack": attack, "actual": count, "assumed_F": F}
                    yield labels, SimConfig(
                        n_agents=n, dim=2, t_final=T_FINAL, topology="custom", seed=seed, F=F, allow_beyond_assumption=True,
                        topology_params={"adjacency": A.astype(int).tolist()}, compromised_ids=order[:count].tolist(),
                        attack=attack, attack_params=params, controller="wmsr_consensus", record_lambda2=False)


def summarise(rows) -> dict:
    cells = {}
    for F in sorted({r["assumed_F"] for r in rows}):
        for count in sorted({r["actual"] for r in rows}):
            chosen = [r for r in rows if r["assumed_F"] == F and r["actual"] == count]
            cells[f"F={F} actual={count}"] = {
                "runs": len(chosen),
                "success_rate": float(np.mean([r["success"] for r in chosen])),
                "median_spread": float(np.median([r["final_spread"] for r in chosen])),
                "median_overshoot": float(np.median([r["hull_overshoot"] for r in chosen])),
                "median_t_converge": float(np.median([r["t_converge"] for r in chosen])),
                "mean_honest_discarded": float(np.mean([r["honest_discarded"] for r in chosen])),
            }
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a small version, to check that the script works")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--root", default="results", help="where to write the results and figures")
    args = parser.parse_args(argv)

    seeds = 2 if args.quick else SEEDS
    cases = list(study_cases(seeds, attacks=ATTACKS[:1] if args.quick else ATTACKS, n=10 if args.quick else N))
    print(f"{len(cases)} runs on {args.workers} workers")
    results = run_configs([cfg for _, cfg in cases], METRICS, args.workers)
    rows = [{**labels, **result} for (labels, _), result in zip(cases, results)]
    for row in rows:
        row["success"] = bool(row["final_spread"] < SPREAD_TOL and row["hull_overshoot"] < HULL_TOL)
    summary = summarise(rows)

    grid = {"assumed_F": list(ASSUMED), "actual": list(ACTUAL), "attack": list(ATTACKS)}
    path = record_experiment("degradation", SimConfig(controller="wmsr_consensus", t_final=T_FINAL), grid, range(seeds), rows, args.root)
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"rows and summary in {path}")
    for name, cell in summary.items():
        print(f"{name:16} success {cell['success_rate']:5.2f}  spread {cell['median_spread']:9.2e}  overshoot {cell['median_overshoot']:8.2f}  "
              f"t {cell['median_t_converge']:6.1f}  discarded {cell['mean_honest_discarded']:.3f}")

    from plot_robustness import plot_degradation
    print(f"figure in {plot_degradation(rows, Path(args.root) / 'figures')}")


if __name__ == "__main__":
    main()
