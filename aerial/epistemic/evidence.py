"""Proofs of lying (T3.4–T3.7). Every rule is something an honest drone can never trigger.

R1  a fresh beacon disagrees with my UWB range taken at the same tick (A17): | |p_j - p_i| - r_ij | > eps'.
    Honest j claims its exact pose at the transmit tick, so only range noise (|n| <= 4 sigma <= eps') remains.
R2  j's own claims imply an impossible speed: claimed |v_h| > v_h or |v_z| > v_z, or two claims further apart than
    the speed limits allow. Uses only j's own signed beacons, so anyone can check it.
R3  a fresh beacon claims a distance > R + eps' (it was heard, so it was within R at the transmit tick).
R4  j's fresh claim lies inside my seen-empty volume (by a margin) and nothing was sighted near it. The frame and the
    beacon are from the same tick, and sightings use the same depth image as ``contains`` (T1.6).
R5  audit: I detected a survivor in a cell j claimed CLEARED. Detection is deterministic and exact-cell (A7), and
    survivors are static, so an honest j that viewed the cell would have reported the survivor instead.
R6  reporter r's shared range to j (for beacon seq s) contradicts j's claim for that seq: clause {j, r}.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Proof = Tuple[str, Tuple[int, ...]]   # (rule, members); one member = Unit, several = Clause


def check_beacons(obs, eps: float, R: float, v_h: float, v_z: float, history: Dict[int, dict]) -> List[Proof]:
    out: List[Proof] = []
    me = obs.own.pos
    dt = obs.static.dt
    for j, b in obs.beacons.items():
        c = b.claim
        hist = history.setdefault(j, {})
        if b.age == 0 and b.uwb_range is not None:
            d = float(np.linalg.norm(c.pos - me))
            if abs(d - b.uwb_range) > eps + 1e-6:      # noise is clipped at exactly 4 sigma = eps'
                out.append(("R1", (j,)))
            if d > R + eps + 1e-6:
                out.append(("R3", (j,)))
        if np.linalg.norm(c.vel[:2]) > v_h + 1e-6 or abs(c.vel[2]) > v_z + 1e-6:
            out.append(("R2", (j,)))
        if b.seq not in hist:
            for s0 in sorted(hist)[-1:]:
                p0 = hist[s0]
                gap = (b.seq - s0) * dt
                if gap > 0:
                    dp = c.pos - p0
                    if np.linalg.norm(dp[:2]) > v_h * gap + 1e-6 or abs(dp[2]) > v_z * gap + 1e-6:
                        out.append(("R2", (j,)))
            hist[b.seq] = c.pos.copy()
            if len(hist) > 50:
                del hist[min(hist)]
    return out


def check_camera(obs, margin: float = 1.0, delta: float = 1.5) -> List[Proof]:
    f = obs.frame
    if f is None or f.visible is None:
        return []
    out: List[Proof] = []
    for j, b in obs.beacons.items():
        if b.age != 0:
            continue
        p = b.claim.pos
        if f.visible.contains(p[None], margin=margin)[0]:
            if len(f.sightings) == 0 or np.min(np.linalg.norm(f.sightings - p, axis=1)) > delta:
                out.append(("R4", (j,)))
    return out


def check_audit(survivor_cells: np.ndarray, cleared_rep: np.ndarray, me: int) -> List[Proof]:
    """R5: for each cell where *I* saw a survivor, every reporter that claimed it CLEARED lied."""
    out: List[Proof] = []
    for c in survivor_cells:
        rep = int(cleared_rep[c])
        j = 0
        while rep:
            if rep & 1 and j != me:
                out.append(("R5", (j,)))
            rep >>= 1
            j += 1
    return out


def check_shared_range(reporter: int, j: int, seq: int, rng: float, p_r: np.ndarray,
                       claims_by_seq: Dict[int, dict], eps: float) -> List[Proof]:
    c = claims_by_seq.get(j, {}).get(seq)
    if c is None:
        return []
    if abs(float(np.linalg.norm(c - p_r)) - rng) > 2 * eps:
        return [("R6", (j, reporter))]
    return []
