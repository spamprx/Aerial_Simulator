"""Command line:

    python -m aerial.cli run   --config configs/lab_default.json --seeds 5 [--plot out.png]
    python -m aerial.cli live  --config configs/lab_default.json --speed 2 [--save run.gif]
    python -m aerial.cli sweep --config configs/lab_default.json \
           --grid drop_prob=0,0.1,0.3 --grid n_agents=5,10 --seeds 10 --out results/sweep.csv \
           --workers 8 --exp-name my_study
    python -m aerial.cli pair  --config configs/scenarios/demo_corridor.json --seeds 5 \
           --fleets naive,gated --out results/pair.csv --workers 8 --exp-name my_pair
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import SimConfig
from .experiments.runner import FLEETS, pair_sweep, run_batch, run_one, summarize, sweep
from .scenarios import fleet_overrides


def _parse_grid(items):
    grid = {}
    for item in items or []:
        key, vals = item.split("=", 1)
        parsed = []
        for v in vals.split(","):
            try:
                parsed.append(json.loads(v))
            except json.JSONDecodeError:
                parsed.append(v)  # bare string such as ring
        grid[key] = parsed
    return grid


def _pair_table(rows, fleets) -> str:
    lines = [f"{'fleet':11} {'runs':>5} {'violate':>8} {'finish':>7} {'mean s':>8} {'worst rho':>10} {'collisions':>11} {'held s':>7} {'requeries':>10}"]
    for fleet in fleets:
        chosen = [row for row in rows if row["variant"] == fleet]
        finished = [row["completion_time"] for row in chosen if np.isfinite(row["completion_time"])]
        lines.append(f"{fleet:11} {len(chosen):5d} {np.mean([r['safety_robustness'] < 0.0 for r in chosen]):8.0%} {len(finished):7d} "
                     f"{np.mean(finished) if finished else float('inf'):8.2f} {min(r['safety_robustness'] for r in chosen):10.2f} "
                     f"{np.mean([r['n_collisions'] for r in chosen]):11.1f} {np.mean([r['hesitation_time'] for r in chosen]):7.2f} "
                     f"{np.mean([r['n_requeries'] for r in chosen]):10.0f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aerial")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one config over several seeds")
    r.add_argument("--config", default=None)
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--plot", default=None, help="save a time-series figure of seed 0 here")

    v = sub.add_parser("live", help="animate the simulation while it runs")
    v.add_argument("--config", default=None)
    v.add_argument("--speed", type=float, default=1.0, help="simulated s per wall-clock s (1 = real time)")
    v.add_argument("--trail", type=int, default=40, help="trail length in steps (0 = off)")
    v.add_argument("--seed", type=int, default=None)
    v.add_argument("--save", default=None, help="write .gif / .mp4 instead of opening a window")

    s = sub.add_parser("sweep", help="cartesian parameter sweep -> CSV")
    s.add_argument("--config", default=None)
    s.add_argument("--grid", action="append", help="key=v1,v2,...  (repeatable)")
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--out", default="results/sweep.csv")
    s.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    s.add_argument("--exp-name", default=None, help="also record config, grid, rows and env under results/<name>/<time>/")

    p = sub.add_parser("pair", help="the fleets of D1 under identical randomness -> CSV")
    p.add_argument("--config", required=True, help="a scenario, such as configs/scenarios/demo_corridor.json")
    p.add_argument("--fleets", default=",".join(FLEETS), help="which fleets, comma separated")
    p.add_argument("--seeds", type=int, default=1, help="the seeds change the randomness, and the scenario's layout stays as it is")
    p.add_argument("--out", default="results/pair.csv")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    p.add_argument("--exp-name", default=None, help="also record config, variants, rows and env under results/<name>/<time>/")

    a = ap.parse_args(argv)
    cfg = SimConfig.load(a.config) if a.config else SimConfig()

    if a.cmd == "live":
        from .viz.live import animate
        if a.seed is not None:
            cfg = cfg.with_overrides(seed=a.seed)
        animate(cfg, speed=a.speed, trail=a.trail, save=a.save)
        return

    if a.cmd == "pair":
        fleets = a.fleets.split(",")
        rows = pair_sweep(cfg, {fleet: fleet_overrides(cfg, fleet) for fleet in fleets}, range(a.seeds), out_csv=a.out,
                          workers=a.workers, exp_name=a.exp_name)
        print(_pair_table(rows, fleets))
        print(f"paired: same start, liars, packet drops and attack noise. {len(rows)} runs written to {a.out}")
        return

    if a.cmd == "run":
        rows = run_batch(cfg, range(a.seeds))
        for k in rows[0]:
            if k == "seed":
                continue
            v = np.array([r[k] for r in rows], dtype=float)
            fin = v[np.isfinite(v)]
            note = "" if len(fin) == len(v) else f"   ({len(v) - len(fin)} non-finite, e.g. never converged)"
            mean, std = (fin.mean(), fin.std()) if len(fin) else (float("nan"), float("nan"))
            print(f"{k:>24s}: mean {mean:10.4f}   std {std:9.4f}{note}")
        if a.plot:
            import matplotlib
            matplotlib.use("Agg")
            from .viz.plots import plot_timeseries
            _, res = run_one(cfg, return_result=True)
            plot_timeseries(res).savefig(a.plot, dpi=150)
            print(f"saved {a.plot}")
    else:
        grid = _parse_grid(a.grid)
        rows = sweep(cfg, grid, range(a.seeds), out_csv=a.out, verbose=True, workers=a.workers, exp_name=a.exp_name)
        print(f"{len(rows)} runs written to {a.out}")
        if grid:
            print(json.dumps({str(k): v for k, v in
                              summarize(rows, list(grid), "final_pos_spread").items()}, indent=1))


if __name__ == "__main__":
    main()
