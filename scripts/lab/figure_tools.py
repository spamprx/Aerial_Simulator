"""What the figure scripts (T6.4) share: the command line, where a figure's rows come from, and how the demo's runs are kept.

    make figures            every figure from the newest recorded study, into results/figures/
    make figures QUICK=1    every study run small first, into results/quick/, so that nothing there can be taken for a result

A study is recorded in results/<study>/<time>/.  With none recorded, a figure falls back on the rows the report's results
came from, in docs/report/data/<study>.csv.  With neither, it runs the study, unless that would take hours: the trade-off study
says how to run it instead, and its figure is skipped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from aerial.lab import SimResult
from plot_robustness import latest_rows, load_rows, newest_recording

SHIPPED = Path(__file__).resolve().parents[2] / "docs" / "report" / "lab_data"


def figure_args(description: str, argv=None) -> argparse.Namespace:
    """``--quick`` runs a small study first, and ``root`` is then results/quick/, away from the real results."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="run a small study first, to check that the scripts work (for CI)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--root", default="results", help="where the results and figures are")
    args = parser.parse_args(argv)
    args.root = Path(args.root) / "quick" if args.quick else Path(args.root)
    args.figures = args.root / "figures"
    return args


def study_arguments(args: argparse.Namespace) -> List[str]:
    """What to pass an experiment's ``main`` to run it for this figure."""
    return ["--workers", str(args.workers), "--root", str(args.root), *(["--quick"] if args.quick else [])]


def has_recording(study: str, root) -> bool:
    folder = Path(root) / study
    return folder.is_dir() and any((recording / "rows.csv").exists() for recording in folder.iterdir())


def study_rows(study: str, run_study: Callable[[Sequence[str]], object], args: argparse.Namespace, shipped: Path = SHIPPED,
               too_long: str = "") -> Optional[List[dict]]:
    """The rows behind a figure: a small study run now with ``--quick``, or else the newest recorded, or else the ones shipped
    with the report, or else a study run now.  A study that takes too long to start unasked says so in ``too_long``, and then
    with no rows this prints that and returns None."""
    if not args.quick:
        if has_recording(study, args.root):
            return latest_rows(study, args.root)
        if (Path(shipped) / f"{study}.csv").exists():
            return load_rows(Path(shipped) / f"{study}.csv")
        if too_long:
            print(f"no results for {study} yet, so its figure is skipped: {too_long}")
            return None
        print(f"no results for {study}, so running it now: see scripts/ for how long that takes")
    run_study(study_arguments(args))
    return latest_rows(study, args.root)


def demo_runs(args: argparse.Namespace) -> Dict[str, SimResult]:
    """The demo's whole flights, from the newest recording, or from a run now if there is none yet."""
    import exp_demo
    if not has_recording("demo", args.root):
        exp_demo.main(study_arguments(args))
    return exp_demo.load_runs(newest_recording("demo", args.root))
