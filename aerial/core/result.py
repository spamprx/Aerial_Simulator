"""SimResult and the Recorder (T0.5). Extras follow the schema in PLAN.md §6."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from .config import Config

# extras every agent stack may report through diagnostics(); key -> per-agent shape suffix
EXTRAS_SCHEMA = {
    "known_faulty": "(N,)", "known_honest": "(N,)", "n_hyp": "()", "gate_open": "()", "holds": "()",
    "known_cleared_frac": "()", "believed_cleared_frac": "()", "mode": "()",
}


@dataclass
class SimResult:
    t: np.ndarray                     # (K+1,)
    pos: np.ndarray                   # (K+1, N, 3)
    vel: np.ndarray
    yaw: np.ndarray                   # (K+1, N)
    u: np.ndarray                     # (K, N, 3)
    adj: np.ndarray                   # (K, N, N) bool (empty at minimal level)
    lambda2: np.ndarray               # (K,)
    liars: np.ndarray                 # (N,) bool ground truth
    config: Config
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return self.config.sim.dt

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        arrays = {k: getattr(self, k) for k in ("t", "pos", "vel", "yaw", "u", "adj", "lambda2", "liars")}
        arrays.update({f"extras__{k}": v for k, v in self.extras.items()})
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        meta = {"config": self.config.to_dict(), "metrics": _jsonable(self.metrics)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimResult":
        path = Path(path)
        data = np.load(path.with_suffix(".npz"), allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())
        extras = {k[len("extras__"):]: data[k] for k in data.files if k.startswith("extras__")}
        base = {k: data[k] for k in ("t", "pos", "vel", "yaw", "u", "adj", "lambda2", "liars")}
        return cls(config=Config.from_dict(meta["config"]), extras=extras, metrics=meta["metrics"], **base)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


class Recorder:
    """Collects per-step data according to the record level (A13)."""

    def __init__(self, level: str):
        self.level = level
        self.rows: Dict[str, List[np.ndarray]] = {}

    def add(self, key: str, value) -> None:
        if self.level == "minimal":
            return
        self.rows.setdefault(key, []).append(np.asarray(value))

    def add_diagnostics(self, diags: List[Dict[str, Any]]) -> None:
        if self.level == "minimal" or not diags:
            return
        keys = sorted(set().union(*[d.keys() for d in diags]))
        for k in keys:
            vals = [d.get(k) for d in diags]
            if any(v is None for v in vals):
                continue
            self.add(k, np.stack([np.asarray(v) for v in vals]))

    def stacked(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.rows.items():
            try:
                out[k] = np.stack(v)
            except ValueError:
                pass
        return out
