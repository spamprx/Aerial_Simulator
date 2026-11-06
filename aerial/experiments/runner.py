"""Building and running missions (T4.2): ``build_sim``, ``run_one``, ``run_pair`` (with the pairing assertion) and
``sweep`` (capped worker pool, rows to ``results/<exp>/<stamp>/``)."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..agents.mission import MissionAgent
from ..analysis.mission import mission_metrics
from ..core.config import Config
from ..core.simulator import Simulator
from ..resilience.liar import LiarAgent
from ..sensing.camera import sense_all
from ..world.scenario import Scenario, generate_scenario


def static_extra(cfg: Config, scn: Scenario) -> List[dict]:
    N, w = cfg.N, cfg.world
    xs = scn.terrain.zone_origin[0] + (np.arange(N) + 0.5) * w.zone / N
    out = []
    for i in range(N):
        layer = w.layers[i % len(w.layers)]
        out.append(dict(zone_origin=scn.terrain.zone_origin.copy(), coverage_cell=w.coverage_cell,
                        terrain_cell=w.terrain_cell, layer=layer,
                        entry=np.array([xs[i], scn.entry_y + 2.0, layer]),
                        slot=np.array([scn.init_pos[i, 0], scn.init_pos[i, 1], w.z_min])))
    return out


def build_sim(cfg: Config, scn: Optional[Scenario] = None) -> Tuple[Simulator, Scenario]:
    scn = scn or generate_scenario(cfg)
    liars = scn.liars
    agents = []
    truth = np.flatnonzero(liars).tolist()
    liar_rank = {int(j): k for k, j in enumerate(np.flatnonzero(liars))}
    for i in range(cfg.N):
        if liars[i]:
            regions = scn.claim_regions or [scn.claim_region]
            agents.append(LiarAgent(regions[liar_rank[i] % len(regions)], fleet="naive"))
        else:
            agents.append(MissionAgent(fleet=cfg.fleet, oracle_truth=truth))

    def hook(sim, k):
        return sense_all(scn, sim.pos, sim.yaw, k * cfg.sim.dt, cfg.sensor, sim.rng.sense)

    sim = Simulator(cfg, agents, scn.init_pos, init_yaw=scn.init_yaw, liars=liars, base=scn.base,
                    sense_hook=hook, world=scn, static_extra=static_extra(cfg, scn))
    return sim, scn


def run_one(cfg: Config, scn: Optional[Scenario] = None, steps: Optional[int] = None, stop_when_done: bool = True,
            on_step=None):
    """Run one mission. ``on_step(sim, k)`` (optional) is called after every tick, e.g. to export a replay."""
    sim, scn = build_sim(cfg, scn)
    K = int(round(cfg.sim.T / cfg.sim.dt)) if steps is None else steps
    # run with early stop when every honest drone has landed
    from ..core.result import Recorder, SimResult
    rec = Recorder(cfg.sim.record_level)
    pos, vel, yaw, us, lam, adjs = [sim.pos.copy()], [sim.vel.copy()], [sim.yaw.copy()], [], [], []
    found_t = {}
    honest = np.flatnonzero(~scn.liars)
    liar_ids = np.flatnonzero(scn.liars)
    online = {"t_first_proof": np.nan, "t_all_know": np.nan, "gate_closed_ticks": 0, "formation_err_max": 0.0}
    for _ in range(K):
        k = sim.k
        A, u = sim.step()
        pos.append(sim.pos.copy()); vel.append(sim.vel.copy()); yaw.append(sim.yaw.copy()); us.append(u)
        lam.append(sim.net.lambda2(A) if rec.level != "minimal" else np.nan)
        if rec.level != "minimal":
            adjs.append(A)
        rec.add_diagnostics([a.diagnostics() for a in sim.agents])
        if on_step is not None:
            on_step(sim, k)
        for i in honest:
            for c in sim.agents[i].tracks:
                if sim.agents[i].tracks[c]["est"] and c not in found_t:
                    found_t[c] = (k + 1) * cfg.sim.dt
        t_now = (k + 1) * cfg.sim.dt
        if len(liar_ids) and np.isnan(online["t_all_know"]):
            knows = [sim.agents[i].hyp.known_faulty()[liar_ids].all() for i in honest]
            if any(knows) and np.isnan(online["t_first_proof"]):
                online["t_first_proof"] = t_now
            if all(knows):
                online["t_all_know"] = t_now
        online["gate_closed_ticks"] += int(sum(not sim.agents[i].gate_open for i in honest))
        online["formation_err_max"] = max(online["formation_err_max"],
                                          max(sim.agents[i].formation_err for i in honest))
        if stop_when_done and all(sim.agents[i].mode == "LANDED" for i in honest):
            break
    res = SimResult(t=np.arange(len(pos)) * cfg.sim.dt, pos=np.array(pos), vel=np.array(vel), yaw=np.array(yaw),
                    u=np.array(us), adj=np.array(adjs) if adjs else np.zeros((0, cfg.N, cfg.N), bool),
                    lambda2=np.array(lam), liars=scn.liars, config=cfg, extras=rec.stacked())
    res.metrics = mission_metrics(res, scn, sim, found_t)
    res.metrics.update(online)
    return res, scn, sim


def env_signature(cfg: Config, steps: int = 200) -> Dict[str, int]:
    sim, _ = build_sim(cfg)
    for _ in range(steps):
        sim.step()
    return sim.rng.env_signature()


def run_pair(cfg: Config, fleets=("naive", "gated"), check_pairing: bool = True, **kw):
    """Run the same seed and scenario with two fleets; assert both saw identical environment draws."""
    scn = generate_scenario(cfg)
    out = {}
    sigs = {}
    for f in fleets:
        c = cfg.override([f"fleet={f}"])
        res, _, sim = run_one(c, scn, stop_when_done=False, **kw)
        out[f] = res
        sigs[f] = sim.rng.env_signature()
    if check_pairing:
        first = sigs[fleets[0]]
        for f in fleets[1:]:
            assert sigs[f] == first, f"environment draws differ between {fleets[0]} and {f}: pairing broken"
    return out, scn


# ---------------------------------------------------------------------- sweeps
def _row(args):
    cfg_dict, keep_dir, idx = args
    cfg = Config.from_dict(cfg_dict)
    t0 = time.time()
    try:
        res, scn, sim = run_one(cfg)
    except Exception as e:                     # one bad run must not sink the whole sweep
        import traceback
        return {"run": idx, "seed": cfg.sim.seed, "fleet": cfg.fleet, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-400:]}
    m = dict(res.metrics)
    m.update(run=idx, seed=cfg.sim.seed, fleet=cfg.fleet, wall_s=round(time.time() - t0, 2))
    m.update(liar_mode=cfg.attack.position_lie, claim_all=cfg.attack.claim_all, audit_q=cfg.mission.audit_q,
             scenario=cfg.world.scenario, N=cfg.N, F=cfg.F, R=cfg.comms.R, p_drop=cfg.comms.p_drop,
             n_liars=cfg.attack.n_compromised, zone=cfg.world.zone, T=cfg.sim.T,
             trust_gate=cfg.mission.trust_gate, liar_motion=cfg.attack.motion)
    if keep_dir:
        _keep_run(Path(keep_dir), idx, cfg, res, scn, sim, m)
    return {k: v for k, v in m.items() if not isinstance(v, (list, dict, np.ndarray))}


def _keep_run(folder: Path, idx: int, cfg: Config, res, scn, sim, metrics: dict) -> None:
    """Everything needed to analyse a run later without re-simulating: a 1 Hz trajectory (.npz) and a JSON record
    with the metrics, survivor find times, every collision (who, when, where), every proof of lying, and the config."""
    folder.mkdir(parents=True, exist_ok=True)
    every = max(1, int(round(1.0 / cfg.sim.dt)))
    pos = res.pos
    d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
    iu = np.triu_indices(cfg.N, 1)
    close = d[:, iu[0], iu[1]] < cfg.gate.d_safe
    start = close[1:] & ~close[:-1]
    liars = np.flatnonzero(scn.liars).tolist()
    collisions = [{"t": round(float(res.t[k + 1]), 2), "pair": [int(iu[0][p]), int(iu[1][p])],
                   "with_liar": bool(int(iu[0][p]) in liars or int(iu[1][p]) in liars),
                   "pos": np.round(0.5 * (pos[k + 1, iu[0][p]] + pos[k + 1, iu[1][p]]), 2).tolist()}
                  for k, p in zip(*np.nonzero(start))]
    proofs = [{"t": round(k * cfg.sim.dt, 2), "by": int(i), "accused": int(j), "rule": rule}
              for a in sim.agents for (k, i, j, rule) in getattr(a, "proofs", [])]
    np.savez_compressed(folder / f"run_{idx:04d}.npz", t=res.t[::every], pos=pos[::every].astype(np.float32),
                        yaw=res.yaw[::every].astype(np.float32), liars=scn.liars,
                        survivor_cells=scn.survivor_cells, survivor_alive=np.array([s.alive for s in scn.survivors]),
                        claim_region=scn.claim_region)
    record = {"run": idx, "metrics": {k: v for k, v in metrics.items() if k != "found_times"},
              "found_times": metrics.get("found_times", {}), "collisions": collisions, "proofs": proofs,
              "landed": [a.done_t for a in sim.agents], "config": cfg.to_dict()}
    (folder / f"run_{idx:04d}.json").write_text(json.dumps(record, default=float))


def sweep(name: str, configs: Iterable[Config], workers: int = 8, root: str = "results",
          keep: bool = False) -> Path:
    """Run every config in parallel; rows go to rows.jsonl as they finish. ``keep`` also saves each run's detail."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(root) / name / stamp
    out.mkdir(parents=True, exist_ok=True)
    configs = list(configs)
    (out / "grid.json").write_text(json.dumps([c.to_dict() for c in configs]))
    env = {"python": platform.python_version(), "numpy": np.__version__, "workers": workers}
    try:
        env["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        env["git"] = None
    (out / "env.json").write_text(json.dumps(env))
    rows = []
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as ex, \
            open(out / "rows.jsonl", "w") as log:
        keep_dir = str(out / "runs") if keep else None
        futs = [ex.submit(_row, (c.to_dict(), keep_dir, k)) for k, c in enumerate(configs)]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            log.write(json.dumps(r, default=float) + "\n")
            log.flush()
            print(f"[{name}] {n}/{len(configs)} done", flush=True)
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out
