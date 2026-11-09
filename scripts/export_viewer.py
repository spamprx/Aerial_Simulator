"""Export the demo mission (naive vs gated on one scenario) into the 3D viewer: docs/media/viewer.html.

The page is a single self-contained file: viewer/template.html with the run data inlined as JSON.
Usage:  python scripts/export_viewer.py [--T 330] [--out docs/media/viewer.html]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aerial.core.config import Config
from aerial.experiments.runner import run_one
from aerial.world.scenario import generate_scenario

EVERY = 2          # export every 2nd tick (5 Hz); the viewer interpolates between frames
MODE_NAMES = ["Transit", "Explore", "Confirm", "Audit", "Return", "Landed"]


def r2(a, nd=2):
    return np.round(np.asarray(a, float), nd).ravel().tolist()


def record(cfg, scn):
    N = cfg.N
    honest = np.flatnonzero(~scn.liars)
    liar = int(np.flatnonzero(scn.liars)[0]) if scn.liars.any() else -1
    seen_t = np.full(scn.grid.n_cells, -1.0)
    frames = {"pos": [], "yaw": [], "mode": [], "hold": [], "ghost": []}
    lie_start = [None]

    def on_step(sim, k):
        t = (k + 1) * cfg.sim.dt
        union = np.zeros(scn.grid.n_cells, bool)
        for i in honest:
            a = sim.agents[i]
            union |= a.cov.own | a.cov.survivor_own
        new = union & (seen_t < 0)
        seen_t[new] = t
        if (k + 1) % EVERY:
            return
        frames["pos"].append(r2(sim.pos))
        frames["yaw"].append(r2(sim.yaw, 3))
        frames["mode"].append("".join(str(["TRANSIT", "EXPLORE", "CONFIRM", "AUDIT", "RETURN", "LANDED"].index(a.mode))
                                      for a in sim.agents))
        frames["hold"].append("".join("0" if a.gate_open else "1" for a in sim.agents))
        g = None
        if liar >= 0:
            la = sim.agents[liar]
            if getattr(la, "lying", False) and la.fake is not None and la.a.position_lie != "true_pose":
                g = r2(la.fake)
                if lie_start[0] is None:
                    lie_start[0] = t
            elif getattr(la, "lying", False) and lie_start[0] is None:
                lie_start[0] = t
        frames["ghost"].append(g)

    res, _, sim = run_one(cfg, scn, stop_when_done=False, on_step=on_step)
    m = res.metrics
    # collision episodes: first tick a pair drops below d_safe
    d = np.linalg.norm(res.pos[:, :, None] - res.pos[:, None], axis=-1)
    iu = np.triu_indices(N, 1)
    close = d[:, iu[0], iu[1]] < cfg.gate.d_safe
    start = close[1:] & ~close[:-1]
    collisions = [{"t": round(float(res.t[k + 1]), 2), "pair": [int(iu[0][p]), int(iu[1][p])],
                   "liar": bool(liar in (int(iu[0][p]), int(iu[1][p]))),
                   "pos": r2(0.5 * (res.pos[k + 1, iu[0][p]] + res.pos[k + 1, iu[1][p]]))}
                  for k, p in zip(*np.nonzero(start))]
    # when the first honest drone proved the liar lied, and when all knew
    kf = res.extras.get("known_faulty")
    t_first = t_all = None
    if kf is not None and liar >= 0:
        knows = kf[:, honest, liar]
        if knows.any(axis=1).any():
            t_first = float(res.t[1:][np.argmax(knows.any(axis=1))])
        if knows.all(axis=1).any():
            t_all = float(res.t[1:][np.argmax(knows.all(axis=1))])
    landed = [a.done_t for a in sim.agents]
    return {
        "fleet": cfg.fleet,
        "frames": frames,
        "found": {str(c): round(float(v), 2) for c, v in m["found_times"].items()},
        "seen": r2(seen_t, 1),
        "collisions": collisions,
        "lie_start": lie_start[0],
        "caught_first": t_first,
        "caught_all": t_all,
        "landed": [None if x is None else round(float(x), 1) for x in landed],
        "metrics": {k: m[k] for k in ["found", "found_alive", "alive", "survivors", "missed_in_claim", "collisions",
                                      "completion_time", "all_landed", "false_accusations"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=240.0)
    ap.add_argument("--out", default="docs/media/viewer.html")
    ap.add_argument("--rebuild", action="store_true", help="reuse docs/media/viewer_data.json, skip simulating")
    a = ap.parse_args()
    data_file = Path("docs/media/viewer_data.json")
    if a.rebuild:
        build(json.loads(data_file.read_text()), a.out)
        return
    base = Config().override(["world.scenario=demo_district", "attack.n_compromised=1",
                              "attack.position_lie=range_aware", "attack.claim_all=true", "attack.seek_quiet=false",
                              "attack.motion=ambush",
                              f"sim.T={a.T}", "sim.record_level=standard"])
    scn = generate_scenario(base)
    runs = []
    for fleet in ("naive", "gated"):
        print("running", fleet, flush=True)
        runs.append(record(base.override([f"fleet={fleet}"]), scn))
    g = scn.grid
    data = {
        "dt": base.sim.dt * EVERY, "T": a.T, "N": base.N,
        "arena": base.world.arena, "zone": base.world.zone, "zoneOrigin": r2(scn.terrain.zone_origin),
        "terrain": {"n": int(scn.terrain.H.shape[0]), "cell": scn.terrain.cell, "H": r2(scn.terrain.H, 1)},
        "grid": {"n": g.n, "size": g.size, "origin": r2(g.origin), "h": r2(g.target[:, 2] - 0.3, 1)},
        "claim": [int(c) for c in scn.claim_region],
        "liars": np.flatnonzero(scn.liars).tolist(),
        "survivors": [{"cell": s.cell, "pos": r2(s.pos), "alive": bool(s.alive)} for s in scn.survivors],
        "base": r2(scn.base),
        "sensor": {"fov_h": base.sensor.fov_h, "fov_v": base.sensor.fov_v, "tilt": base.sensor.tilt},
        "modes": MODE_NAMES,
        "runs": runs,
    }
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(json.dumps(data, separators=(",", ":"), default=float))
    for r in runs:
        print(r["fleet"], r["metrics"], "caught_all", r["caught_all"])
    build(data, a.out)


def build(data, out):
    tpl = (Path(__file__).resolve().parents[1] / "viewer" / "template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=float))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(out.resolve(), f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
