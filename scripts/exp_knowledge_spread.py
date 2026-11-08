"""E3: how fast knowledge of the liar spreads (fraction of honest drones that know), vs R and p_drop, with an SI fit.

Writes rows with the fitted SI rate beta and the times of first proof / everyone knowing.
"""
import csv
import time
from pathlib import Path

import numpy as np

from _common import Config, parser
from aerial.experiments.runner import run_one
from aerial.viz.dashboard_data import fraction_knowing, si_fit

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    rows = []
    Rs, ps = ([35], [0.1]) if a.quick else ([30, 35, 50], [0.0, 0.1, 0.3])
    for R in Rs:
        for p in ps:
            for s in range(1 if a.quick else a.seeds):
                c = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "fleet=gated",
                                       "sim.T=240", f"comms.R={R}", f"comms.p_drop={p}", f"sim.seed={s}"])
                res, scn, sim = run_one(c)
                fr = fraction_knowing(res)
                beta, _ = si_fit(res.t[1:], fr)
                rows.append(dict(R=R, p_drop=p, seed=s, beta=beta, t_first=res.metrics["t_first_proof"],
                                 t_all=res.metrics["t_all_know"]))
                print(rows[-1])
    out = Path(a.root) / "knowledge_spread" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out)
