"""Statistics and figures for E1-E3 from the newest results/<exp>/<stamp>/rows.csv.

Writes figures/e1_headline.png, figures/e2_detection.png, figures/e3_caution.png and results/summary.json.
Means come with 95% Student-t intervals, rates with 95% Wilson intervals, and naive vs gated is compared per seed
with a Wilcoxon signed-rank test (the runs are paired by seed).
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

FLEET_COL = {"unfiltered": "#718096", "naive": "#c53030", "gated": "#2b6cb0", "oracle": "#2f855a"}
FIG = Path("figures")


def newest(root, exp):
    runs = sorted(Path(root, exp).glob("*/rows.csv"))
    if not runs:
        return None
    rows = [r for r in csv.DictReader(open(runs[-1])) if not r.get("error")]
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = {"True": True, "False": False}.get(v, v)
    return rows


def mean_ci(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def missed(r):
    return r["survivors"] - r["found"]


def e1(rows, out):
    fleets = [f for f in ["naive", "gated"] if any(r["fleet"] == f for r in rows)]   # oracle/unfiltered kept in rows, not shown
    metrics = [("survivors missed", missed), ("collisions", lambda r: r["collisions"]),
               ("completion time (s)", lambda r: r["completion_time"])]
    res = {}
    fig, axs = plt.subplots(1, len(metrics), figsize=(10, 3.2))
    for ax, (name, fn) in zip(axs, metrics):
        ms, hs = [], []
        for f in fleets:
            m, h = mean_ci([fn(r) for r in rows if r["fleet"] == f])
            ms.append(m); hs.append(h)
            res.setdefault(f, {})[name] = [m, h]
        ax.bar(fleets, ms, yerr=hs, color=[FLEET_COL[f] for f in fleets], capsize=4)
        top = max(m + h for m, h in zip(ms, hs)) or 1.0
        for x, (m, h) in enumerate(zip(ms, hs)):
            ax.text(x, m + h + 0.03 * top, f"{m:.0f}" if m >= 10 else f"{m:.2f}".rstrip("0").rstrip("."),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, top * 1.18)
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e1_headline.png", dpi=180, bbox_inches="tight")
    for f in fleets:
        rr = [r for r in rows if r["fleet"] == f]
        res[f]["seeds"] = len(rr)
        res[f]["seeds_missing_any"] = int(sum(missed(r) > 0 for r in rr))
        res[f]["seeds_with_collision"] = int(sum(r["collisions"] > 0 for r in rr))
        res[f]["finished"] = int(sum(bool(r["all_landed"]) for r in rr))
        res[f]["caught"] = int(sum(bool(r["liar_caught"]) for r in rr))
    if {"naive", "gated"} <= set(fleets):
        by = {f: {r["seed"]: r for r in rows if r["fleet"] == f} for f in ("naive", "gated")}
        common = sorted(set(by["naive"]) & set(by["gated"]))
        tests = {}
        for name, fn in metrics:
            d = np.array([fn(by["naive"][s]) - fn(by["gated"][s]) for s in common])
            p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            tests[name] = {"mean_diff_naive_minus_gated": float(d.mean()), "wilcoxon_p": p, "pairs": len(common)}
        res["paired_naive_vs_gated"] = tests
    out["e1"] = res


def e2(rows, out):
    modes = ["fake_sweep_naive", "range_aware", "radio_shadow", "true_pose"]
    modes = [m for m in modes if any(r["liar_mode"] == m for r in rows)]
    claims = [True, False]
    table = {}
    for m in modes:
        for c in claims:
            g = [r for r in rows if r["liar_mode"] == m and r["claim_all"] == c and r["fleet"] == "gated"]
            n = [r for r in rows if r["liar_mode"] == m and r["claim_all"] == c and r["fleet"] == "naive"]
            if not g:
                continue
            k = sum(bool(r["liar_caught"]) for r in g)
            p, lo, hi = wilson(k, len(g))
            t = [r["t_all_know"] for r in g if r["t_all_know"] == r["t_all_know"]]
            rules = {}
            for r in g:
                rules[r["first_rule"]] = rules.get(r["first_rule"], 0) + 1
            table[f"{m}|{'claims whole area' if c else 'claims as it sweeps'}"] = {
                "runs": len(g), "catch_rate": p, "catch_ci": [lo, hi],
                "median_time_all_know": float(np.median(t)) if t else None,
                "first_rule": rules,
                "missed_naive": mean_ci([missed(r) for r in n]), "missed_gated": mean_ci([missed(r) for r in g]),
                "collisions_naive": mean_ci([r["collisions"] for r in n]),
                "collisions_gated": mean_ci([r["collisions"] for r in g])}
    out["e2"] = table
    labels = list(table)
    if not labels:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.55 * len(labels) + 1.4))
    y = np.arange(len(labels))
    axs[0].barh(y, [table[l]["catch_rate"] for l in labels], color="#2b6cb0",
                xerr=np.array([[table[l]["catch_rate"] - table[l]["catch_ci"][0],
                                table[l]["catch_ci"][1] - table[l]["catch_rate"]] for l in labels]).T, capsize=3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([l.replace("|", "\n") for l in labels], fontsize=7)
    axs[0].set_xlim(0, 1.05)
    axs[0].set_title("gated: fraction of runs where the liar is caught", fontsize=9)
    for i, l in enumerate(labels):
        rule = max(table[l]["first_rule"], key=table[l]["first_rule"].get)
        tk = table[l]["median_time_all_know"]
        axs[0].text(0.02, i, f"{rule}" + (f", all know by {tk:.0f}s" if tk else ""), va="center", fontsize=7,
                    color="white" if table[l]["catch_rate"] > 0.35 else "black")
    w = 0.38
    axs[1].barh(y - w / 2, [table[l]["missed_naive"][0] for l in labels], w, color="#c53030", label="naive")
    axs[1].barh(y + w / 2, [table[l]["missed_gated"][0] for l in labels], w, color="#2b6cb0", label="gated")
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([])
    axs[1].set_title("survivors missed (mean)", fontsize=9)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e2_detection.png", dpi=180, bbox_inches="tight")


def e3(rows, out):
    qs = sorted({r["audit_q"] for r in rows})
    res = []
    for q in qs:
        g = [r for r in rows if r["audit_q"] == q]
        k = sum(bool(r["liar_caught"]) for r in g)
        p, lo, hi = wilson(k, len(g))
        theory = float(np.mean([1 - (1 - q) ** r["survivors_in_claim"] for r in g]))
        res.append({"q": q, "runs": len(g), "catch_rate": p, "catch_ci": [lo, hi], "theory_lower_bound": theory,
                    "missed": mean_ci([missed(r) for r in g]), "completion": mean_ci([r["completion_time"] for r in g])})
    out["e3"] = res
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = [r["q"] for r in res]
    ax.errorbar(x, [r["catch_rate"] for r in res],
                yerr=np.array([[r["catch_rate"] - r["catch_ci"][0], r["catch_ci"][1] - r["catch_rate"]]
                               for r in res]).T, marker="o", color="#2b6cb0", capsize=3, lw=2,
                label="liar caught (simulation)")
    ax.plot(x, [r["theory_lower_bound"] for r in res], "k--", lw=1.2, label=r"guarantee from audits alone $1-(1-q)^k$")
    ax.text(0.02, 0.97, f"survivors missed: {res[0]['missed'][0]:.1f} at q={res[0]['q']:g}  →  "
            f"{res[-1]['missed'][0]:.1f} at q={res[-1]['q']:g}", transform=ax.transAxes, fontsize=8.5,
            color="#c53030", va="top")
    ax.set_xlabel("audit fraction q"); ax.set_ylabel("fraction of runs, liar caught"); ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "e3_caution.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    out, total_runs, false_acc = {}, 0, 0
    for exp, fn in [("e1_headline", e1), ("e2_detection", e2), ("e3_caution", e3)]:
        rows = newest(a.root, exp)
        if rows:
            fn(rows, out)
            total_runs += len(rows)
            false_acc += int(sum(r["false_accusations"] for r in rows))
    out["soundness"] = {"runs": total_runs, "false_accusations": false_acc}
    Path(a.root, "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
