"""E6: do survivor coordinates sent to rescuers survive a lying observer?

Each of m honest drones holds a noisy estimate of one survivor (sigma = 0.2 + 0.05 * range, as the detector);
one liar reports the survivor offset by d metres. Drones run the consensus update used by the tracks
(aerial/agents/mission.py: track_estimate) with mean or W-MSR (F = 1) on a complete graph for 20 rounds.
"""
import csv
import time
from pathlib import Path

import numpy as np

from _common import parser
from aerial.resilience.aggregators import MeanAggregator, WMSRAggregator


def trial(rng, m, offset, agg, rounds=20):
    truth = np.zeros(2)
    est = truth + rng.normal(0, 0.2 + 0.05 * rng.uniform(5, 15, (m, 1)), (m, 2))
    liar = truth + np.array([offset, 0.0])
    ids = np.arange(m + 1)
    for _ in range(rounds):
        vals = np.vstack([est, liar])
        est = np.array([agg.aggregate(est[i], np.delete(vals, i, axis=0), np.delete(ids, i)).value
                        for i in range(m)])
    return float(np.linalg.norm(est.mean(axis=0) - truth))


if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for m in ([3] if a.quick else [2, 3, 5, 8]):
        for d in ([8.0] if a.quick else [0.0, 2.0, 8.0, 20.0]):
            for name, agg in [("mean", MeanAggregator()), ("wmsr", WMSRAggregator(f=1))]:
                errs = [trial(rng, m, d, agg) for _ in range(5 if a.quick else a.seeds)]
                rows.append(dict(observers=m, offset=d, aggregator=name, err_mean=np.mean(errs),
                                 err_max=np.max(errs)))
                print(rows[-1])
    out = Path(a.root) / "localisation" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
