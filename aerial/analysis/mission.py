"""Mission metrics (T2.13). Mission metrics count honest drones only; safety counts all drones (D9)."""
from __future__ import annotations

import numpy as np


def mission_metrics(res, scn, sim, found_t: dict) -> dict:
    cfg = res.config
    honest = np.flatnonzero(~scn.liars)
    pos = res.pos
    # safety: drone-drone separation (all pairs) and terrain collisions
    d = np.linalg.norm(pos[:, :, None] - pos[:, None, :], axis=-1)
    iu = np.triu_indices(cfg.N, 1)
    pair = d[:, iu[0], iu[1]]
    min_sep = float(pair.min()) if pair.size else np.inf
    collisions = int(((pair[1:] < cfg.gate.d_safe) & (pair[:-1] >= cfg.gate.d_safe)).sum())
    h = scn.terrain.height_at(pos[..., :2])
    clearance = pos[..., 2] - h
    terrain_hits = int((clearance < 0).any(axis=0).sum())
    # survivors
    surv = scn.survivors
    cells = scn.survivor_cells
    found = np.array([c in found_t for c in cells])
    alive = np.array([s.alive for s in surv])
    claim = np.isin(cells, scn.claim_region) if scn.liars.any() else np.zeros(len(cells), bool)
    # classification by honest drones (any honest drone's decision)
    correct = []
    for s in surv:
        decs = [sim.agents[i].tracks[s.cell]["belief"].decision for i in honest if s.cell in sim.agents[i].tracks]
        decs = [x for x in decs if x is not None]
        if decs:
            correct.append(decs[0] == ("alive" if s.alive else "dead"))
    # coverage: union of honest drones' own views over ground-truth viewable cells
    union = np.zeros(scn.grid.n_cells, bool)
    for i in honest:
        union |= sim.agents[i].cov.own | sim.agents[i].cov.survivor_own
    view = scn.viewable
    # knowledge
    liars = np.flatnonzero(scn.liars)
    kf = res.extras.get("known_faulty")
    t_first = t_all = np.nan
    if kf is not None and len(liars):
        knows = kf[:, honest][:, :, liars].all(axis=2)          # (K, honest)
        any_k, all_k = knows.any(axis=1), knows.all(axis=1)
        t_first = float(np.argmax(any_k) * cfg.sim.dt) if any_k.any() else np.nan
        t_all = float(np.argmax(all_k) * cfg.sim.dt) if all_k.any() else np.nan
    false_acc = 0
    true_proofs = []
    for i in honest:
        for (k, _, j, rule) in sim.agents[i].proofs:
            false_acc += int(not scn.liars[j])
            if scn.liars[j]:
                true_proofs.append((k, rule))
    first_rule = min(true_proofs)[1] if true_proofs else "none"
    done_ts = [sim.agents[i].done_t for i in honest]
    completion = float(max(done_ts)) if all(t is not None for t in done_ts) else float(res.t[-1])
    kc = res.extras.get("known_cleared_frac")
    bc = res.extras.get("believed_cleared_frac")
    return {
        "completion_time": completion,
        "all_landed": all(t is not None for t in done_ts),
        "survivors": len(surv),
        "found": int(found.sum()),
        "found_alive": int((found & alive).sum()),
        "alive": int(alive.sum()),
        "missed_in_claim": int((~found & claim).sum()),
        "missed_elsewhere": int((~found & ~claim).sum()),
        "classification_acc": float(np.mean(correct)) if correct else np.nan,
        "coverage": float((union & view).sum() / max(1, view.sum())),
        "min_separation": min_sep,
        "collisions": collisions,
        "terrain_hits": terrain_hits,
        "t_first_proof": t_first,
        "t_all_know": t_all,
        "false_accusations": false_acc,
        "first_rule": first_rule,
        "n_proofs": len(true_proofs),
        "liar_caught": bool(true_proofs),
        "survivors_in_claim": int(claim.sum()),
        "holds": int(sum(sim.agents[i].holds for i in honest)),
        "known_cleared_final": float(kc[-1, honest].mean()) if kc is not None and len(kc) else np.nan,
        "believed_cleared_final": float(bc[-1, honest].mean()) if bc is not None and len(bc) else np.nan,
        "msgs_sent": int(sim.gossip.sent.sum()),
        "found_times": {int(k): v for k, v in found_t.items()},
    }
