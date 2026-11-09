"""Analyse the one-parameter-at-a-time sweep (scripts/exp_sweep.py).

Outputs (figures/ and results/sweep/<stamp>/):
  sweep_found.png       survivors found (%) against each parameter, naive vs gated, mean with 95% interval
  sweep_collisions.png  collisions per run (solid) and collisions with the liar (dashed)
  sweep_time.png        completion time: mean with 95% interval, and p95 (dotted)
  sweep_tradeoff.png    every setting as one point: completion time vs survivors found
  stats.csv / stats.json  per setting and fleet: mean, 95% CI, median, p95, worst, success rates;
                          naive-vs-gated paired Wilcoxon tests; the breaking-point table
The raw data stays untouched in rows.jsonl and runs/ (one .json + .npz per run).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

COL = {"naive": "#c53030", "gated": "#2b6cb0"}
BASE = {"N": 10, "n_liars": 1, "zone": 100}
SWEEPS = [("N", "number of drones"), ("n_liars", "number of liars"), ("zone", "city size (m)")]


def load(folder: Path):
    rows = [json.loads(l) for l in open(folder / "rows.jsonl")]
    rows = [r for r in rows if not r.get("error")]
    for r in rows:
        rec = folder / "runs" / f"run_{int(r['run']):04d}.json"
        if rec.exists():
            d = json.loads(rec.read_text())
            r["coll_liar"] = sum(c["with_liar"] for c in d["collisions"])
        else:
            r["coll_liar"] = np.nan
        r["found_pct"] = 100.0 * r["found"] / max(1, r["survivors"])
    return rows


def ci(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def group(rows, key):
    """Rows of the sweep over ``key``: the other two parameters at their baseline values."""
    others = [k for k in BASE if k != key]
    return [r for r in rows if all(r[k] == BASE[k] for k in others)]


def summarise(rr):
    found, coll, t = (np.array([r[k] for r in rr], float) for k in ("found_pct", "collisions", "completion_time"))
    return {"runs": len(rr),
            "found_pct": ci(found), "found_median": float(np.median(found)), "found_worst": float(found.min()),
            "all_found_rate": float(np.mean(found >= 100)),
            "collisions": ci(coll), "coll_with_liar": ci([r["coll_liar"] for r in rr]),
            "no_collision_rate": float(np.mean(coll == 0)),
            "time": ci(t), "time_median": float(np.median(t)), "time_p95": float(np.percentile(t, 95)),
            "time_worst": float(t.max()), "finished_rate": float(np.mean([bool(r["all_landed"]) for r in rr])),
            "liar_caught_rate": float(np.mean([bool(r["liar_caught"]) for r in rr])) if rr[0]["n_liars"] else None,
            "false_accusations": int(sum(r["false_accusations"] for r in rr))}


def paired(rr, metric):
    by = {f: {r["seed"]: r[metric] for r in rr if r["fleet"] == f} for f in ("naive", "gated")}
    common = sorted(set(by["naive"]) & set(by["gated"]))
    d = np.array([by["naive"][s] - by["gated"][s] for s in common], float)
    p = float(stats.wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else 1.0
    return {"mean_diff_naive_minus_gated": float(d.mean()) if len(d) else np.nan, "p": p, "pairs": len(common)}


def panel_plot(table, metric, ylabel, fname, extra=None, cap=None):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (key, label) in zip(axs, SWEEPS):
        if cap:
            ax.axhline(cap, color="#718096", ls="--", lw=1)
            ax.text(0.02, cap, " time limit", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7,
                    color="#4a5568")
        xs = sorted(table[key])
        for f in ("naive", "gated"):
            m = np.array([table[key][x][f][metric][0] for x in xs])
            h = np.array([table[key][x][f][metric][1] for x in xs])
            ax.plot(xs, m, "-o", color=COL[f], label=f, lw=2)
            ax.fill_between(xs, m - h, m + h, color=COL[f], alpha=0.15)
            if extra:
                ax.plot(xs, [extra[1](table[key][x][f]) for x in xs], extra[0], color=COL[f], lw=1.2)
        ax.set_xticks(xs)
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path("figures") / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    folder = sorted(Path(a.root, "sweep").glob("*/rows.jsonl"))[-1].parent
    rows = load(folder)
    Path("figures").mkdir(exist_ok=True)
    table, tests = {}, {}
    for key, _ in SWEEPS:
        g = group(rows, key)
        table[key] = {}
        for x in sorted({r[key] for r in g}):
            rr = [r for r in g if r[key] == x]
            table[key][x] = {f: summarise([r for r in rr if r["fleet"] == f]) for f in ("naive", "gated")}
            tests[f"{key}={x}"] = {m: paired(rr, m) for m in ("found_pct", "collisions", "completion_time")}
    panel_plot(table, "found_pct", "survivors found (%)", "sweep_found.png")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
    for ax, (key, label) in zip(axs, SWEEPS):
        xs = sorted(table[key]); w = 0.38
        for k, f in enumerate(("naive", "gated")):
            tot = [round(table[key][x][f]["collisions"][0] * table[key][x][f]["runs"]) for x in xs]
            lia = [round(table[key][x][f]["coll_with_liar"][0] * table[key][x][f]["runs"]) for x in xs]
            pos = np.arange(len(xs)) + (k - 0.5) * w
            ax.bar(pos, tot, w, color=COL[f], label=f)
            ax.bar(pos, lia, w, color="none", edgecolor="black", hatch="///", lw=0)
            for p_, t_ in zip(pos, tot):
                ax.text(p_, t_ + 0.05, str(t_), ha="center", fontsize=8)
        ax.set_xticks(range(len(xs)), [str(x) for x in xs]); ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(f"collisions (total over {table['N'][10]['naive']['runs']} runs)")
    axs[0].set_ylim(0, 3); axs[0].legend(fontsize=8, title="hatched: with the liar", title_fontsize=7)
    fig.tight_layout(); fig.savefig(Path("figures") / "sweep_collisions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    T = max(r["T"] for r in rows)
    panel_plot(table, "time", "completion time (s)", "sweep_time.png", cap=T)
    # every setting as an arrow from naive to gated
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    marks = {"N": "o", "n_liars": "s", "zone": "^"}
    names = {"N": "drones", "n_liars": "liars", "zone": "city (m)"}
    for key, _ in SWEEPS:
        for x, v in table[key].items():
            n, g = v["naive"], v["gated"]
            ax.annotate("", xy=(g["time"][0], g["found_pct"][0]), xytext=(n["time"][0], n["found_pct"][0]),
                        arrowprops=dict(arrowstyle="->", color="#a0aec0", lw=1))
            ax.scatter(n["time"][0], n["found_pct"][0], color=COL["naive"], marker=marks[key], s=46, zorder=3)
            ax.scatter(g["time"][0], g["found_pct"][0], color=COL["gated"], marker=marks[key], s=46, zorder=3)
            is_base = x == BASE[key]
            if is_base and key != "N":
                continue                                    # the baseline is the same setup in all three sweeps
            ax.annotate("baseline" if is_base else f"{names[key]}={x}", (n["time"][0], n["found_pct"][0]),
                        fontsize=6.5, xytext=(4, -9), textcoords="offset points", color="#4a5568")
    for key, _ in SWEEPS:
        ax.scatter([], [], color="#718096", marker=marks[key], label=f"varying {names[key]}")
    ax.scatter([], [], color=COL["naive"], label="naive"); ax.scatter([], [], color=COL["gated"], label="gated")
    ax.set_xlabel("completion time (s)"); ax.set_ylabel("survivors found (%)")
    ax.legend(fontsize=7, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("every arrow: same setup, naive → gated", fontsize=9)
    fig.tight_layout(); fig.savefig(Path("figures") / "sweep_tradeoff.png", dpi=180, bbox_inches="tight")
    # breaking points: values where the mean falls below 95% of survivors found
    breaking = {key: {f: [x for x in sorted(table[key]) if table[key][x][f]["found_pct"][0] < 95] for f in COL}
                for key, _ in SWEEPS}
    out = {"folder": str(folder), "table": {k: {str(x): v for x, v in t.items()} for k, t in table.items()},
           "paired_tests": tests, "below_95pct_found": breaking,
           "false_accusations_total": int(sum(r["false_accusations"] for r in rows)), "runs": len(rows)}
    (folder / "stats.json").write_text(json.dumps(out, indent=2, default=float))
    with open(folder / "stats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "value", "fleet", "runs", "found_pct_mean", "found_pct_ci", "found_worst",
                    "all_found_rate", "collisions_mean", "collisions_ci", "coll_with_liar_mean", "no_collision_rate",
                    "time_mean", "time_ci", "time_median", "time_p95", "time_worst", "finished_rate",
                    "liar_caught_rate", "false_accusations"])
        for key, t in table.items():
            for x, v in t.items():
                for f, s in v.items():
                    w.writerow([key, x, f, s["runs"], *np.round(s["found_pct"], 2), s["found_worst"],
                                s["all_found_rate"], *np.round(s["collisions"], 2), round(s["coll_with_liar"][0], 2),
                                s["no_collision_rate"], *np.round(s["time"], 1), s["time_median"], s["time_p95"],
                                s["time_worst"], s["finished_rate"], s["liar_caught_rate"], s["false_accusations"]])
    print(json.dumps({"runs": len(rows), "false_accusations_total": out["false_accusations_total"],
                      "below_95pct_found": breaking}, indent=1))
    print("stats:", folder / "stats.csv")


if __name__ == "__main__":
    main()
