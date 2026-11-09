"""Render a replay as an MP4 video: naive (left) vs gated (right), top-down, with live stats.

Reads the data written by scripts/export_viewer.py, so no simulation is needed:
    python scripts/make_video.py                       # collision demo  -> docs/media/video_collisions.mp4
    python scripts/make_video.py --preset demo         # main demo       -> docs/media/video_demo.mp4
    python scripts/make_video.py --speed 6 --fps 25    # 6x real time, 25 frames per second
Needs ffmpeg on the PATH. A good offline backup for a presentation (the 3D pages need internet once).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap

DATA = {"collisions": "docs/media/viewer_collisions_data.json", "demo": "docs/media/viewer_data.json"}
COL = {"honest": "#2563eb", "liar": "#dc2626", "caught": "#f59e0b", "found_alive": "#16a34a", "found_dead": "#475569",
       "unfound": "#e11d48"}
TITLE = {"naive": "Naive fleet  (believes every claim)", "gated": "Gated fleet  (acts only on what it can prove)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(DATA), default="collisions")
    ap.add_argument("--speed", type=float, default=8.0, help="simulated seconds per video second")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=None, help="only render the first N simulated seconds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = json.loads(Path(DATA[a.preset]).read_text())
    out = Path(a.out or f"docs/media/video_{a.preset}.mp4")
    runs = {r["fleet"]: r for r in D["runs"]}
    fleets = [f for f in ("naive", "gated") if f in runs]
    N, dt, A = D["N"], D["dt"], D["arena"]
    liar = D["liars"][0] if D["liars"] else -1
    nf = len(runs[fleets[0]]["frames"]["pos"])
    T_end = (nf - 1) * dt if a.seconds is None else min(a.seconds, (nf - 1) * dt)
    times = np.arange(0.0, T_end + 1e-9, a.speed / a.fps)

    # static layers
    n = D["terrain"]["n"]
    H = np.array(D["terrain"]["H"]).reshape(n, n)
    g = D["grid"]
    gn, gs, go = g["n"], g["size"], g["origin"]
    claim = np.array(D["claim"])
    ca, cb = claim // gn, claim % gn
    cx0, cx1 = go[0] + ca.min() * gs, go[0] + (ca.max() + 1) * gs
    cy0, cy1 = go[1] + cb.min() * gs, go[1] + (cb.max() + 1) * gs
    surv = D["survivors"]
    sp = np.array([s["pos"][:2] for s in surv])

    fig, axs = plt.subplots(1, len(fleets), figsize=(6.4 * len(fleets), 7.0), facecolor="white")
    axs = np.atleast_1d(axs)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03, wspace=0.05)
    sup = fig.suptitle("", fontsize=13, fontweight="bold", color="#0d47a1")
    art = {}
    for ax, f in zip(axs, fleets):
        r = runs[f]
        ax.set_xlim(0, A); ax.set_ylim(0, A); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#cbd5e1")
        ax.imshow(np.clip(H.T, 0, 35), origin="lower", extent=[0, A, 0, A], cmap="Greys", vmin=-4, vmax=38,
                  interpolation="nearest")
        cover = np.full((gn, gn), np.nan)
        cov_img = ax.imshow(cover.T, origin="lower", extent=[go[0], go[0] + gn * gs, go[1], go[1] + gn * gs],
                            cmap=ListedColormap(["#86efac"]), alpha=0.45, vmin=0, vmax=1, interpolation="nearest")
        ax.plot([cx0, cx1, cx1, cx0, cx0], [cy0, cy0, cy1, cy1, cy0], "--", color=COL["liar"], lw=1.4)
        claim_txt = ax.text(cx0 + 1, cy1 - 2, "liar claims: 'searched, nobody here'", color=COL["liar"], fontsize=7.5,
                            va="top", visible=False)
        trails = [ax.plot([], [], "-", lw=0.8, alpha=0.5, color=COL["liar"] if i == liar else COL["honest"])[0]
                  for i in range(N)]
        drones = ax.scatter(np.zeros(N), np.zeros(N), s=46, zorder=5, edgecolors="white", linewidths=0.8)
        holds = ax.scatter([], [], s=150, facecolors="none", edgecolors="#facc15", linewidths=1.6, zorder=4)
        ghost = ax.scatter([], [], s=60, color=COL["liar"], alpha=0.35, zorder=5, marker="o")
        ghost_line, = ax.plot([], [], ":", color=COL["liar"], lw=1.2)
        ghost_txt = ax.text(0, 0, "claims to be here", color=COL["liar"], fontsize=7, visible=False)
        surv_sc = ax.scatter(sp[:, 0], sp[:, 1], s=40, marker="X", zorder=4, edgecolors="white", linewidths=0.5)
        bursts = [(c["t"], ax.scatter([c["pos"][0]], [c["pos"][1]], s=0, color="#ef4444", alpha=0.6, zorder=6),
                   ax.text(c["pos"][0], c["pos"][1] + 3,
                           (f"Drone {[d for d in c['pair'] if d != liar][0]} hit the liar" if c.get("liar")
                            else f"Collision {c['pair'][0]} & {c['pair'][1]}"),
                           fontsize=8, color="white", fontweight="bold", ha="center", visible=False,
                           bbox=dict(boxstyle="round,pad=0.25", fc="#dc2626", ec="none")))
                  for c in r["collisions"]]
        marks = ax.scatter([], [], s=50, marker="*", color="#ef4444", edgecolors="black", linewidths=0.4, zorder=6)
        title = ax.set_title(TITLE[f], fontsize=10.5, loc="left", color="#0f172a")
        art[f] = dict(r=r, cov_img=cov_img, claim_txt=claim_txt, trails=trails, drones=drones, holds=holds,
                      ghost=ghost, ghost_line=ghost_line, ghost_txt=ghost_txt, surv_sc=surv_sc, bursts=bursts,
                      marks=marks, title=title)

    def frame(t):
        k = min(int(t / dt), nf - 2)
        w = t / dt - k
        sup.set_text(f"AERIAL v2 replay   ·   t = {t:5.1f} s   ·   {a.speed:g}x speed")
        for f in fleets:
            o, r = art[f], art[f]["r"]
            F = r["frames"]
            P = (1 - w) * np.array(F["pos"][k]).reshape(N, 3) + w * np.array(F["pos"][k + 1]).reshape(N, 3)
            caught = r["caught_first"] is not None and t >= r["caught_first"]
            lying = r["lie_start"] is not None and t >= r["lie_start"]
            cols = [COL["caught"] if (i == liar and caught) else COL["liar"] if i == liar else COL["honest"]
                    for i in range(N)]
            o["drones"].set_offsets(P[:, :2]); o["drones"].set_color(cols); o["drones"].set_edgecolors("white")
            k0 = max(0, k - 150)
            hist = np.array(F["pos"][k0:k + 1]).reshape(-1, N, 3)
            for i, tr in enumerate(o["trails"]):
                tr.set_data(hist[:, i, 0], hist[:, i, 1])
            hold = [i for i in range(N) if F["hold"][k][i] == "1"]
            o["holds"].set_offsets(P[hold, :2] if hold else np.zeros((0, 2)))
            gpos = F["ghost"][k]
            if gpos is not None and liar >= 0:
                o["ghost"].set_offsets([gpos[:2]])
                o["ghost_line"].set_data([P[liar, 0], gpos[0]], [P[liar, 1], gpos[1]])
                o["ghost_txt"].set_position((gpos[0] + 1.5, gpos[1] + 1.5)); o["ghost_txt"].set_visible(True)
            else:
                o["ghost"].set_offsets(np.zeros((0, 2))); o["ghost_line"].set_data([], []); o["ghost_txt"].set_visible(False)
            o["claim_txt"].set_visible(lying)
            seen = np.array(r["seen"])
            cov = np.where((seen >= 0) & (seen <= t), 1.0, np.nan).reshape(gn, gn)
            o["cov_img"].set_data(cov.T)
            found = [r["found"].get(str(s["cell"]), np.inf) <= t for s in surv]
            o["surv_sc"].set_color([(COL["found_alive"] if s["alive"] else COL["found_dead"]) if fd else COL["unfound"]
                                    for s, fd in zip(surv, found)])
            o["surv_sc"].set_edgecolors("white")
            past = []
            for tc, sc, tx in o["bursts"]:
                age = t - tc
                fresh = 0 <= age < 3.0
                sc.set_sizes([2500 * (0.2 + age / 3.0)] if fresh else [0]); sc.set_alpha(0.6 * (1 - age / 3.0) if fresh else 0)
                tx.set_visible(fresh)
                if age >= 0:
                    past.append(sc.get_offsets()[0])
            o["marks"].set_offsets(np.array(past) if past else np.zeros((0, 2)))
            n_found = sum(found)
            n_coll = sum(1 for c in r["collisions"] if c["t"] <= t)
            n_cl = sum(1 for c in r["collisions"] if c["t"] <= t and c.get("liar"))
            liar_s = (f"liar PROVEN at {r['caught_first']:.1f} s" if caught else
                      "liar lying, undetected" if lying else "liar not lying yet")
            o["title"].set_text(f"{TITLE[f]}\nsurvivors {n_found}/{len(surv)}   ·   collisions {n_coll}"
                                f"{f' ({n_cl} with liar)' if n_coll else ''}   ·   {liar_s}")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=times, interval=1000 / a.fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2400), dpi=110)
    print(out.resolve(), f"{out.stat().st_size / 1e6:.1f} MB, {len(times) / a.fps:.0f} s of video")


if __name__ == "__main__":
    main()
