"""Figures for the presentation (figures/*.png): top-down demo snapshots and knowledge curves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from aerial.core.config import Config
from aerial.experiments.runner import run_pair
from aerial.viz.dashboard_data import cleared_curves, fraction_knowing

OUT = Path("figures")


def topdown(ax, res, scn, title):
    H = scn.terrain.H
    ax.imshow(H.T, origin="lower", cmap="Greys", vmin=0, vmax=35, extent=[0, 120, 0, 120])
    g = scn.grid
    reg = g.target[scn.claim_region, :2]
    lo, hi = reg.min(0) - g.size / 2, reg.max(0) + g.size / 2
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ls="--", lw=1.5, ec="#c53030"))
    ax.text(lo[0] + 1, hi[1] - 5, "liar's claimed area", color="#c53030", fontsize=8)
    for i in range(res.pos.shape[1]):
        c = "#c53030" if res.liars[i] else "#2b6cb0"
        ax.plot(res.pos[:, i, 0], res.pos[:, i, 1], color=c, lw=1.6 if res.liars[i] else 0.7,
                alpha=1 if res.liars[i] else 0.55)
    found = {int(k) for k in res.metrics.get("found_times", {})}
    for s in scn.survivors:
        if s.cell in found:
            ax.plot(*s.pos[:2], marker="D", ms=6, color="#2f855a" if s.alive else "#4a5568", mec="white")
        else:
            ax.plot(*s.pos[:2], marker="X", ms=9, color="#e53e3e", mec="white")
    d = np.linalg.norm(res.pos[:, :, None] - res.pos[:, None], axis=-1)
    n = d.shape[1]
    d[:, np.arange(n), np.arange(n)] = np.inf
    k, i, j = np.nonzero(d < res.config.gate.d_safe)
    for kk, ii in zip(k[::5], i[::5]):
        ax.plot(*res.pos[kk, ii, :2], marker="*", ms=12, color="#ed8936", mec="black", mew=0.4)
    m = res.metrics
    ax.set_title(f"{title}: found {m['found']}/{m['survivors']}, collisions {m['collisions']}", fontsize=10)
    ax.set_xlim(0, 120); ax.set_ylim(0, 120); ax.set_xticks([]); ax.set_yticks([])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    c = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "sim.T=240",
                           "attack.position_lie=range_aware", "attack.claim_all=true", "attack.seek_quiet=false",
                           "attack.motion=ambush",
                           "sim.record_level=standard"])
    from aerial.viz.dashboard_data import run_pair_cached
    out, scn = run_pair_cached(c, ("naive", "gated"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    topdown(axs[0], out["naive"], scn, "Naive")
    topdown(axs[1], out["gated"], scn, "Gated")
    fig.tight_layout(); fig.savefig(OUT / "demo_topdown.png", dpi=180, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    cn, cg = cleared_curves(out["naive"]), cleared_curves(out["gated"])
    ax.plot(out["naive"].t[1:], cn["believed_cleared_frac"], color="#c53030",
            label="naive: area it thinks is searched (includes the lie)")
    ax.plot(out["gated"].t[1:], cg["known_cleared_frac"], color="#2b6cb0", label="gated: area it knows is searched")
    fr = fraction_knowing(out["gated"])
    ax.plot(out["gated"].t[1:], fr, color="#2f855a", lw=2, label="gated: fraction knowing the liar")
    ax.set_xlabel("time (s)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "knowledge_curves.png", dpi=180, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(scn.terrain.H.T, origin="lower", cmap="terrain", extent=[0, 120, 0, 120])
    ax.set_title("Generated disaster area (height, m)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "terrain.png", dpi=180)

    summary = {f: {k: out[f].metrics[k] for k in ["found", "survivors", "missed_in_claim", "collisions",
                                                    "completion_time", "t_all_know", "false_accusations"]}
               for f in out}
    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
