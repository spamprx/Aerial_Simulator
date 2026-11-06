"""Shared helpers for experiment scripts: argument parsing and config grids."""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerial.core.config import Config  # noqa: E402


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--quick", action="store_true", help="tiny grid, for checking the script works")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--root", default="results")
    ap.add_argument("--set", nargs="*", default=[], help="extra dotted config overrides, e.g. mission.trust_gate=true")
    ap.add_argument("--name", default=None, help="results folder name (default: the experiment's own name)")
    return ap


def grid(base: Config, seeds, **axes):
    """Every combination of dotted overrides in ``axes`` (key with '.' replaced by '__') times seeds."""
    keys = list(axes)
    for combo in itertools.product(*axes.values()):
        for s in seeds:
            ov = [f"{k.replace('__', '.')}={v}" for k, v in zip(keys, combo)] + [f"sim.seed={s}"]
            yield base.override(ov)
