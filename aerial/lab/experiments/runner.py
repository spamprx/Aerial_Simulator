"""Experiment runner: single runs, seed batches, paired fleets and parameter sweeps -> CSV."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import multiprocessing
import pickle
import platform
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SimConfig
from ..analysis.metrics import DEFAULT_METRICS, Metric
from ..core.simulator import SimResult, Simulator
from ..scenarios import fleet_overrides

FLEETS = ("naive", "gated", "unfiltered", "oracle")  # D1


def run_one(
    cfg: SimConfig,
    metrics: Optional[Dict[str, Metric]] = None,
    controller=None,
    attack=None,
    return_result: bool = False,
):
    """Run one simulation and evaluate the metrics.  Returns ``row`` (or ``(row, result)``)."""
    metrics = DEFAULT_METRICS if metrics is None else metrics
    res: SimResult = Simulator(cfg, controller=controller, attack=attack).run()
    row = {name: fn(res) for name, fn in metrics.items()}
    return (row, res) if return_result else row


def run_batch(cfg: SimConfig, seeds: Iterable[int], metrics=None) -> List[Dict[str, Any]]:
    """Same configuration over several seeds (the 'repeated randomised runs' of the plan)."""
    rows = []
    for s in seeds:
        row = run_one(cfg.with_overrides(seed=int(s)), metrics)
        rows.append({"seed": int(s), **row})
    return rows


def _run_task(task) -> Dict[str, Any]:
    cfg_dict, metrics = task
    return run_one(SimConfig.from_dict(cfg_dict), metrics)


def _run_signed_task(task) -> Dict[str, Any]:
    cfg_dict, metrics = task
    row, res = run_one(SimConfig.from_dict(cfg_dict), metrics, return_result=True)
    return {**row, "pairing": pairing_signature(res)}


def run_configs(configs: Sequence[SimConfig], metrics=None, workers: int = 1) -> List[Dict[str, Any]]:
    """Evaluate the metrics on each config, returning the rows in the order of ``configs``.

    With ``workers > 1`` each run happens in its own process, so ``metrics`` must be picklable (no
    lambdas) and anything custom a config names must be imported from ``aerial/__init__.py``.  A
    script that calls this needs the usual ``if __name__ == "__main__":`` guard.

    Workers inherit this process's environment on purpose, BLAS thread settings included: a different
    thread count can change eigenvalues in the last digits, and the rows must match a serial run exactly.
    """
    return _run_tasks(_run_task, configs, metrics, workers)


def _run_tasks(task, configs: Sequence[SimConfig], metrics, workers: int) -> List[Dict[str, Any]]:
    configs = list(configs)
    workers = min(workers, len(configs))
    if workers <= 1:
        return [task((cfg.to_dict(), metrics)) for cfg in configs]
    try:
        pickle.dumps(metrics)
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        raise ValueError(f"parallel runs need picklable metrics, so use module-level functions: {error}") from error
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return pool.map(task, [(cfg.to_dict(), metrics) for cfg in configs], chunksize=1)


def paired_streams(res: SimResult) -> Dict[str, np.ndarray]:
    """What has to be identical between paired runs: all of it comes from random streams no controller may touch."""
    return {"pos[0]": res.pos[0], "compromised": res.compromised, "drop_keep": res.extras.get("drop_keep", np.empty(0)),
            "attack_noise": res.extras.get("attack_noise", np.empty(0))}


def _where_they_differ(first: np.ndarray, second: np.ndarray) -> str:
    if first.shape != second.shape:
        return f"shapes {first.shape} and {second.shape}"
    same = (first == second) | ((first != first) & (second != second))  # NaN pads rows, and it is the same in both
    return f"first at index {np.argwhere(~same)[0].tolist()}"


def check_paired(results: Dict[str, SimResult]) -> None:
    """Raise AssertionError unless every one of these runs drew the same randomness.  Runs that did not are not a
    paired comparison, and a difference between them could be down to luck."""
    (first_name, first), *others = results.items()
    expected = paired_streams(first)
    for name, res in others:
        for stream, value in paired_streams(res).items():
            if not np.array_equal(expected[stream], value, equal_nan=True):
                raise AssertionError(f"'{name}' is not paired with '{first_name}': {stream} differs, "
                                     f"{_where_they_differ(expected[stream], value)}")


def pairing_signature(res: SimResult) -> str:
    """A digest of ``paired_streams``, for sweeps too large to keep every run's arrays: same digest, same randomness."""
    digest = hashlib.sha256()
    for stream, value in paired_streams(res).items():
        value = np.asarray(value)
        digest.update(f"{stream} {value.shape} {value.dtype}".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()[:16]


def check_paired_rows(rows: Sequence[Dict[str, Any]], paired_by: Sequence[str]) -> None:
    """``check_paired`` for rows holding a ``pairing`` signature: rows that agree on ``paired_by`` must have the same one."""
    signatures = defaultdict(set)
    for row in rows:
        signatures[tuple(row[key] for key in paired_by)].add(row["pairing"])
    unpaired = [key for key, found in signatures.items() if len(found) > 1]
    if unpaired:
        raise AssertionError(f"{len(unpaired)} groups of runs are not paired, first {dict(zip(paired_by, unpaired[0]))}")


def run_pair(
    cfg: SimConfig,
    variants: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Metric]] = None,
) -> Dict[str, Tuple[Dict[str, Any], SimResult]]:
    """Run the same scenario as several fleets under identical randomness.  Returns ``{name: (row, result)}``.

    ``variants`` maps a name to the overrides that make that fleet, and defaults to the four of D1, from
    ``fleet_overrides``.  This raises AssertionError unless the runs have the same ``pos[0]``, the same ``compromised``, the
    same ``extras["drop_keep"]`` and the same ``extras["attack_noise"]``, which is what makes a difference between them
    the fleet's doing.  The check needs those logs, so ``record_extras`` is always on here.

    G(t) itself differs between variants, because positions differ.  Pairing means the same *randomness*, not the same
    graph: a link is kept or dropped alike in every run, and whether it exists depends on where that fleet has flown.
    """
    if variants is None:
        variants = {fleet: fleet_overrides(cfg, fleet) for fleet in FLEETS}
    if not variants:
        raise ValueError("run_pair needs at least one variant")
    runs = {name: run_one(cfg.with_overrides({**overrides, "record_extras": True}), metrics, return_result=True)
            for name, overrides in variants.items()}
    check_paired({name: res for name, (_, res) in runs.items()})
    return runs


def pair_sweep(
    cfg: SimConfig,
    variants: Optional[Dict[str, Dict[str, Any]]] = None,
    seeds: Iterable[int] = (0,),
    metrics=None,
    out_csv: Optional[str] = None,
    workers: int = 1,
    exp_name: Optional[str] = None,
    results_root="results",
) -> List[Dict[str, Any]]:
    """``run_pair`` for each seed, keeping the rows and not the runs, so it can run in parallel.  Each row holds the
    variant, the seed, the metrics and a ``pairing`` signature, in the order of the seed and then the variant, and this
    raises AssertionError if the variants of a seed are not paired.  ``metrics`` and ``exp_name`` are as in ``sweep``."""
    if variants is None:
        variants = {fleet: fleet_overrides(cfg, fleet) for fleet in FLEETS}
    seeds = [int(s) for s in seeds]
    cases = [(name, seed) for seed in seeds for name in variants]
    configs = [cfg.with_overrides({**variants[name], "seed": seed, "record_extras": True}) for name, seed in cases]
    results = _run_tasks(_run_signed_task, configs, metrics, workers)
    rows = [{"variant": name, "seed": seed, **row} for (name, seed), row in zip(cases, results)]
    if out_csv:
        save_csv(rows, out_csv)
    if exp_name:
        record_experiment(exp_name, cfg, {"variant": variants}, seeds, rows, results_root)
    check_paired_rows(rows, ["seed"])
    return rows


def sweep(
    base: SimConfig,
    grid: Dict[str, Sequence[Any]],
    seeds: Iterable[int] = (0,),
    metrics=None,
    out_csv: Optional[str] = None,
    verbose: bool = False,
    workers: int = 1,
    exp_name: Optional[str] = None,
    results_root="results",
) -> List[Dict[str, Any]]:
    """Cartesian-product sweep.  ``grid`` maps config keys (dotted keys allowed) to value
    lists; every combination is run for every seed.  Each output row holds the grid values,
    the seed and the metrics, in the order of the grid and then the seed, whatever ``workers`` is.

    ``exp_name`` also records the run under ``results_root`` (see ``record_experiment``)."""
    seeds = list(seeds)
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    cases = [(dict(zip(keys, combo)), int(s)) for combo in combos for s in seeds]
    configs = [base.with_overrides({**overrides, "seed": s}) for overrides, s in cases]
    if verbose:
        print(f"{len(combos)} combinations x {len(seeds)} seeds = {len(cases)} runs on {max(workers, 1)} workers")
    results = run_configs(configs, metrics, workers)
    rows = [{**overrides, "seed": s, **row} for (overrides, s), row in zip(cases, results)]
    if out_csv:
        save_csv(rows, out_csv)
    if exp_name:
        path = record_experiment(exp_name, base, grid, seeds, rows, results_root)
        if verbose:
            print(f"recorded in {path}")
    return rows


def environment_info(seeds: Sequence[int] = ()) -> Dict[str, Any]:
    """What it takes to reproduce a result: the code version, package versions and seeds."""
    import matplotlib
    import networkx

    def git(*args) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": networkx.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": [int(s) for s in seeds],
    }


def record_experiment(exp_name: str, base: SimConfig, grid: Dict[str, Sequence[Any]], seeds: Sequence[int],
                      rows: List[Dict[str, Any]], root="results") -> Path:
    """Write ``<root>/<exp_name>/<YYYYmmdd-HHMMSS>/`` with config.json, grid.json, rows.csv and env.json."""
    parent = Path(root) / exp_name
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path, attempt = parent / stamp, 1
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:  # two runs in the same second must not overwrite each other
            path, attempt = parent / f"{stamp}-{attempt}", attempt + 1
    base.save(path / "config.json")
    (path / "grid.json").write_text(json.dumps(grid, indent=2, default=str))
    (path / "env.json").write_text(json.dumps(environment_info(seeds), indent=2))
    save_csv(rows, path / "rows.csv")
    return path


def save_csv(rows: List[Dict[str, Any]], path) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, Any]], group_by: Sequence[str], metric: str):
    """Mean / std / count of ``metric`` per group (finite values only)."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[g] for g in group_by)].append(r[metric])
    out = {}
    for key, vals in groups.items():
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        out[key] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std()) if len(v) else float("nan"),
            "n": int(len(v)),
        }
    return out
