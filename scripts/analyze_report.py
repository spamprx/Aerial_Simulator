"""Statistics and figures for the final report, from the newest rows of the reruns with the current code.

Reads results/{e1_fair, e2_current, e3_current}/<newest>/rows.csv and results/localisation/<newest>/rows.csv.
Writes figures/report_*.png and results/report_summary.json.

Means come with 95% Student-t intervals, rates with 95% Wilson intervals; naive vs gated is compared per seed with a
Wilcoxon signed-rank test (runs are paired by seed).

    python scripts/analyze_report.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
COL = {"unfiltered": "#718096", "naive": "#c53030", "gated": "#2b6cb0", "oracle": "#2f855a"}
FLEETS = ["unfiltered", "naive", "gated", "oracle"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "savefig.dpi": 200,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-"})


def newest(exp):
    runs = sorted((RES / exp).glob("*/rows.csv"))
    if not runs:
        return None, None
    rows = []
    for r in csv.DictReader(open(runs[-1])):
        if r.get("error"):
            continue
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = {"True": True, "False": False}.get(v, v)
        rows.append(r)
    return rows, str(runs[-1].parent.relative_to(ROOT))


def mean_ci(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def found_pct(r):
    return 100.0 * r["found"] / r["survivors"]


def paired(rows, a, b, metric):
    A = {r["seed"]: metric(r) for r in rows if r["fleet"] == a}
    Bd = {r["seed"]: metric(r) for r in rows if r["fleet"] == b}
    seeds = sorted(set(A) & set(Bd))
    x = np.array([A[s] for s in seeds]); y = np.array([Bd[s] for s in seeds])
    d = x - y
    p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(x, y).pvalue)
    return {"pairs": len(seeds), "mean_diff": float(d.mean()), "wilcoxon_p": p}


def errbar(ax, xs, means, cis, colours):
    ax.bar(xs, means, yerr=cis, color=colours, capsize=4, width=0.62, edgecolor="none", zorder=3,
           error_kw={"elinewidth": 1.2, "ecolor": "#2d3748"})


# --------------------------------------------------------------------------------------------- E1: four fleets
def e1(out):
    rows, folder = newest("e1_fair")
    if rows is None:
        return
    T = rows[0]["T"]
    res = {"folder": folder, "T": T}
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    metrics = [("survivors found (%)", found_pct),
               ("survivors missed in the liar's area", lambda r: r["missed_in_claim"]),
               ("completion time (s)", lambda r: r["completion_time"])]
    for ax, (name, fn) in zip(axs, metrics):
        ms, cs, fl = [], [], []
        for f in FLEETS:
            rr = [r for r in rows if r["fleet"] == f]
            m, c = mean_ci([fn(r) for r in rr])
            res.setdefault(f, {})[name] = [m, c]
            ms.append(m); cs.append(c); fl.append(f)
        ok = [i for i, m in enumerate(ms) if m == m]
        errbar(ax, [fl[i] for i in ok], [ms[i] for i in ok], [cs[i] for i in ok], [COL[fl[i]] for i in ok])
        for i in ok:
            ax.text(fl[i], ms[i] + (cs[i] if cs[i] == cs[i] else 0), f"{ms[i]:.0f}" if ms[i] >= 10 else f"{ms[i]:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#2d3748")
        if "time" in name and "completion" in name:
            ax.axhline(T, ls="--", lw=1, color="#a0aec0")
            ax.text(-0.45, T, "time limit", fontsize=7.5, color="#718096", va="bottom")
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="x", visible=False)
    if res["naive"]["survivors found (%)"][0] > 80:
        axs[0].set_ylim(60, 104)
    for f in FLEETS:
        rr = [r for r in rows if r["fleet"] == f]
        res[f]["collisions per run"] = list(mean_ci([r["collisions"] for r in rr]))
        res[f]["time until all honest know the liar (s)"] = list(mean_ci([r["t_all_know"] for r in rr]))
        res[f].update(runs=len(rr), finished=int(sum(r["completion_time"] < T - 1e-6 for r in rr)),
                      caught=int(sum(bool(r["liar_caught"]) for r in rr)),
                      seeds_missing_any=int(sum(r["found"] < r["survivors"] for r in rr)),
                      seeds_with_collision=int(sum(r["collisions"] > 0 for r in rr)),
                      false_accusations=int(sum(r["false_accusations"] for r in rr)),
                      missed_in_claim=mean_ci([r["missed_in_claim"] for r in rr]),
                      found_worst=float(min(found_pct(r) for r in rr)),
                      completion_median=float(np.median([r["completion_time"] for r in rr])))
    res["paired"] = {f"naive_vs_{b}": {"found_pct": paired(rows, "naive", b, found_pct),
                                       "completion_time": paired(rows, "naive", b, lambda r: r["completion_time"]),
                                       "collisions": paired(rows, "naive", b, lambda r: r["collisions"])}
                     for b in ("gated", "oracle")}
    res["paired"]["gated_vs_oracle"] = {"completion_time": paired(rows, "gated", "oracle",
                                                                  lambda r: r["completion_time"])}
    fig.tight_layout()
    fig.savefig(FIG / "report_e1_fleets.png", bbox_inches="tight")
    plt.close(fig)
    out["e1_fair"] = res


# --------------------------------------------------------------------------------------------- E2: detection
LIE_NAME = {"fake_sweep_naive": "fake sweep", "range_aware": "range-aware", "radio_shadow": "radio shadow",
            "true_pose": "true pose\n(coverage only)"}


def e2(out):
    rows, folder = newest("e2_current")
    if rows is None:
        return
    res = {"folder": folder}
    modes = ["fake_sweep_naive", "range_aware", "radio_shadow", "true_pose"]
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    width = 0.38
    rule_cols = {"R1": "#2b6cb0", "R2": "#63b3ed", "R3": "#805ad5", "R4": "#d69e2e", "R5": "#38a169",
                 "R6": "#e53e3e", "none": "#cbd5e0"}
    rule_tot = {}
    for ci, claim in enumerate([True, False]):
        rates, lo, hi, tk, tkc, nmiss, gmiss = [], [], [], [], [], [], []
        for m in modes:
            g = [r for r in rows if r["liar_mode"] == m and r["claim_all"] == claim and r["fleet"] == "gated"]
            n = [r for r in rows if r["liar_mode"] == m and r["claim_all"] == claim and r["fleet"] == "naive"]
            k = sum(bool(r["liar_caught"]) for r in g)
            p, a, b = wilson(k, len(g))
            rates.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (b - p))
            tall = [r["t_all_know"] for r in g if r["t_all_know"] == r["t_all_know"]]
            tfirst = [r["t_first_proof"] for r in g if r["t_first_proof"] == r["t_first_proof"]]
            rules = {}
            for r in g:
                rules[r["first_rule"]] = rules.get(r["first_rule"], 0) + 1
                rule_tot.setdefault(m, {}).setdefault(r["first_rule"], 0)
                rule_tot[m][r["first_rule"]] += 1
            key = f"{m}|{'claims whole area' if claim else 'claims gradually'}"
            res[key] = {"runs": len(g), "catch_rate": p, "catch_ci": [a, b], "caught": k,
                        "median_t_first_proof": float(np.median(tfirst)) if tfirst else None,
                        "median_t_all_know": float(np.median(tall)) if tall else None,
                        "first_rule": rules,
                        "naive_missed": mean_ci([r["survivors"] - r["found"] for r in n]),
                        "gated_missed": mean_ci([r["survivors"] - r["found"] for r in g]),
                        "naive_found_pct": mean_ci([found_pct(r) for r in n]),
                        "gated_found_pct": mean_ci([found_pct(r) for r in g]),
                        "naive_time": mean_ci([r["completion_time"] for r in n]),
                        "gated_time": mean_ci([r["completion_time"] for r in g]),
                        "false_accusations": int(sum(r["false_accusations"] for r in g + n))}
            nmiss.append(res[key]["naive_missed"][0]); gmiss.append(res[key]["gated_missed"][0])
        x = np.arange(len(modes)) + (ci - 0.5) * width
        lab = "liar claims its whole corner at once" if claim else "liar claims cells gradually"
        axs[0].bar(x, rates, width, yerr=[lo, hi], capsize=3, color="#2b6cb0" if claim else "#90cdf4",
                   label=lab, zorder=3, error_kw={"elinewidth": 1})
        w2 = 0.2
        base = np.arange(len(modes)) + (ci * 2 - 1.5) * w2
        tag = "whole corner" if claim else "gradual"
        axs[1].bar(base, nmiss, w2, color="#c53030" if claim else "#feb2b2", zorder=3, label=f"naive, {tag}")
        axs[1].bar(base + w2, gmiss, w2, color="#2b6cb0" if claim else "#90cdf4", zorder=3, label=f"gated, {tag}")
    for ax in axs[:2]:
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels([LIE_NAME[m] for m in modes], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("liar caught (%), gated fleet")
    axs[0].set_ylim(0, 115)
    axs[0].set_title("Which liars are caught?")
    axs[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    axs[1].set_ylabel("survivors missed per run")
    axs[1].set_title("What the lie costs")
    axs[1].legend(fontsize=7.5, frameon=False)
    # first rule that caught the liar, pooled over claim modes
    bottom = np.zeros(len(modes))
    for rule in ["R1", "R2", "R3", "R4", "R5", "R6", "none"]:
        v = np.array([rule_tot.get(m, {}).get(rule, 0) for m in modes], float)
        if v.sum() == 0:
            continue
        axs[2].bar(np.arange(len(modes)), v, 0.6, bottom=bottom, color=rule_cols[rule],
                   label=rule if rule != "none" else "not caught", zorder=3)
        bottom += v
    axs[2].set_xticks(np.arange(len(modes)))
    axs[2].set_xticklabels([LIE_NAME[m] for m in modes], fontsize=8.5)
    axs[2].set_ylabel("gated runs")
    axs[2].set_title("First proof that fired")
    axs[2].legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
    axs[2].set_ylim(0, bottom.max() * 1.35)
    axs[2].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "report_e2_detection.png", bbox_inches="tight")
    plt.close(fig)
    res["first_rule_by_mode"] = rule_tot
    out["e2_current"] = res


# --------------------------------------------------------------------------------------------- E3: audit fraction
def e3(out):
    rows, folder = newest("e3_current")
    if rows is None:
        return
    res = {"folder": folder}
    qs = sorted({r["audit_q"] for r in rows})
    rate, lo, hi, miss, mci, tm, tci = [], [], [], [], [], [], []
    for q in qs:
        g = [r for r in rows if r["audit_q"] == q]
        k = sum(bool(r["liar_caught"]) for r in g)
        p, a, b = wilson(k, len(g))
        m, c = mean_ci([r["survivors"] - r["found"] for r in g])
        t, tc = mean_ci([r["completion_time"] for r in g])
        tall = [r["t_first_proof"] for r in g if r["t_first_proof"] == r["t_first_proof"]]
        res[str(q)] = {"runs": len(g), "catch_rate": p, "catch_ci": [a, b], "missed": [m, c], "time": [t, tc],
                       "median_t_first_proof": float(np.median(tall)) if tall else None,
                       "finished": int(sum(r["completion_time"] < r["T"] - 1e-6 for r in g)),
                       "false_accusations": int(sum(r["false_accusations"] for r in g))}
        rate.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (b - p))
        miss.append(m); mci.append(c); tm.append(t); tci.append(c if False else tc)
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.2))
    axs[0].errorbar(qs, rate, yerr=[lo, hi], marker="o", color="#2b6cb0", capsize=4, lw=2)
    qq = np.linspace(0, max(qs), 50)
    axs[0].plot(qq, 100 * qq, ls="--", color="#a0aec0", lw=1)
    axs[0].text(max(qs) * 0.55, 100 * max(qs) * 0.55 - 9, "guaranteed lower bound $q$\n(audits alone, one survivor)",
                fontsize=7.5, color="#718096")
    axs[0].set_ylim(0, 105)
    axs[0].set_ylabel("liar caught (%)")
    axs[0].set_title("Catch rate vs audit fraction")
    axs[1].errorbar(qs, miss, yerr=mci, marker="o", color="#c53030", capsize=4, lw=2)
    axs[1].set_ylabel("survivors missed per run")
    axs[1].set_title("Survivors missed")
    axs[1].set_ylim(bottom=0)
    axs[2].errorbar(qs, tm, yerr=tci, marker="o", color="#4a5568", capsize=4, lw=2)
    axs[2].set_ylabel("completion time (s)")
    axs[2].set_title("Cost of caution")
    for ax in axs:
        ax.set_xlabel("audit fraction $q$")
    fig.tight_layout()
    fig.savefig(FIG / "report_e3_audit.png", bbox_inches="tight")
    plt.close(fig)
    out["e3_current"] = res


# --------------------------------------------------------------------------------------------- localisation
def localisation(out):
    rows, folder = newest("localisation")
    if rows is None:
        return
    res = {"folder": folder}
    obs = sorted({int(r["observers"]) for r in rows})
    offs = sorted({r["offset"] for r in rows})
    fig, axs = plt.subplots(1, len(obs), figsize=(12.6, 2.9), sharey=True)
    for ax, n in zip(axs, obs):
        for agg, col, lab in [("mean", "#718096", "plain mean"), ("wmsr", "#2b6cb0", "W-MSR")]:
            rr = sorted([r for r in rows if int(r["observers"]) == n and r["aggregator"] == agg], key=lambda r: r["offset"])
            ax.plot([r["offset"] for r in rr], [r["err_mean"] for r in rr], marker="o", color=col, lw=2, label=lab)
            res[f"{n}|{agg}"] = {str(r["offset"]): [r["err_mean"], r["err_max"]] for r in rr}
        ax.plot(offs, offs, ls=":", color="#c53030", lw=1, label="size of the lie")
        ax.set_title(f"{n} observers (one lies)")
        ax.set_xlabel("liar's offset (m)")
    axs[0].set_ylabel("honest estimate error (m)")
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "report_localisation.png", bbox_inches="tight")
    plt.close(fig)
    out["localisation"] = res


if __name__ == "__main__":
    out = {}
    for fn in (e1, e2, e3, localisation):
        fn(out)
    (RES / "report_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v.get("folder") for k, v in out.items()}, indent=1))
