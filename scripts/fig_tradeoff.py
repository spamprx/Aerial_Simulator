"""Trade-off figure from the newest results/tradeoff rows: completion time vs survivors missed, per fleet."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    runs = sorted(Path(a.root, "tradeoff").glob("*/rows.csv"))
    if not runs:
        print("no tradeoff rows yet; run scripts/exp_tradeoff.py")
        raise SystemExit(0)
    rows = list(csv.DictReader(open(runs[-1])))
    fig, ax = plt.subplots(figsize=(6, 4))
    for fleet, colour in [("naive", "#c53030"), ("gated", "#2b6cb0"), ("oracle", "#2f855a"), ("unfiltered", "#718096")]:
        rr = [r for r in rows if r["fleet"] == fleet]
        if rr:
            ax.scatter([float(r["completion_time"]) for r in rr],
                       [int(r["survivors"]) - int(r["found"]) for r in rr], s=14, alpha=0.6, label=fleet, c=colour)
    ax.set_xlabel("mission completion time (s)")
    ax.set_ylabel("survivors missed")
    ax.legend()
    out = Path("docs/report/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=160)
    print(out / "tradeoff.png")
