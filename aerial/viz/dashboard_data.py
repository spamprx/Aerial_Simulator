"""Data preparation for the dashboard: plain functions, tested headless (T4.7).

Runs are cached on disk by config hash, so the dashboard reloads a paired demo in seconds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..core.config import Config
from ..core.result import SimResult
from ..world.scenario import generate_scenario

CACHE = Path("results/cache")


def config_key(cfg: Config, fleets) -> str:
    blob = json.dumps({"cfg": cfg.to_dict(), "fleets": list(fleets)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def run_pair_cached(cfg: Config, fleets=("naive", "gated"), cache: Path = CACHE) -> Tuple[Dict[str, SimResult], object]:
    from ..experiments.runner import run_pair
    key = config_key(cfg, fleets)
    folder = Path(cache) / key
    scn = generate_scenario(cfg)
    if all((folder / f"{f}.npz").exists() for f in fleets):
        return {f: SimResult.load(folder / f) for f in fleets}, scn
    out, scn = run_pair(cfg, fleets=fleets)
    folder.mkdir(parents=True, exist_ok=True)
    for f, res in out.items():
        res.save(folder / f)
    return out, scn


def knowledge_matrix(res: SimResult, k: int) -> np.ndarray:
    """(N, N) at step k: +1 = i knows j is a liar, 0 = unknown. Rows are observers."""
    kf = res.extras.get("known_faulty")
    if kf is None or len(kf) == 0:
        return np.zeros((len(res.liars),) * 2)
    return kf[min(k, len(kf) - 1)].astype(float)


def fraction_knowing(res: SimResult) -> np.ndarray:
    """Fraction of honest drones that know every liar, over time."""
    kf = res.extras.get("known_faulty")
    liars, honest = np.flatnonzero(res.liars), np.flatnonzero(~res.liars)
    if kf is None or len(liars) == 0:
        return np.zeros(max(0, len(res.t) - 1))
    return kf[:, honest][:, :, liars].all(axis=2).mean(axis=1)


def si_fit(t: np.ndarray, frac: np.ndarray):
    """Least-squares logistic (SI model) fit: I(t) = 1 / (1 + (1/I0 - 1) exp(-beta (t - t0))). Returns (beta, curve)."""
    from scipy.optimize import curve_fit
    if frac.max() <= 0:
        return np.nan, np.zeros_like(frac)
    t0 = t[np.argmax(frac > 0)]
    tt = np.clip(t - t0, 0, None)
    i0 = max(frac[frac > 0].min(), 1e-3)

    def f(x, beta):
        return np.where(t >= t0, 1 / (1 + (1 / i0 - 1) * np.exp(-beta * x)), 0.0)
    try:
        (beta,), _ = curve_fit(f, tt, frac, p0=[1.0], bounds=(1e-3, 100))
    except Exception:
        return np.nan, np.zeros_like(frac)
    return float(beta), f(tt, beta)


def cleared_curves(res: SimResult) -> Dict[str, np.ndarray]:
    honest = np.flatnonzero(~res.liars)
    out = {}
    for key in ("known_cleared_frac", "believed_cleared_frac"):
        v = res.extras.get(key)
        if v is not None and len(v):
            out[key] = v[:, honest].mean(axis=1)
    return out


def tradeoff_rows(root: Path = Path("results/tradeoff")):
    import csv
    runs = sorted(Path(root).glob("*/rows.csv"))
    if not runs:
        return []
    with open(runs[-1]) as f:
        return list(csv.DictReader(f))
