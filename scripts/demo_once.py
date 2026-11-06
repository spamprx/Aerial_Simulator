"""Run demo_district once with naive, gated and oracle fleets and print the headline metrics."""
import argparse
import time

from aerial.core.config import Config
from aerial.experiments.runner import run_pair

KEYS = ["completion_time", "found", "missed_in_claim", "missed_elsewhere", "collisions", "terrain_hits",
        "t_first_proof", "t_all_know", "false_accusations", "holds"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lie", default="range_aware")
    ap.add_argument("--T", type=float, default=360)
    a = ap.parse_args()
    c = Config()
    c.world.scenario = "demo_district"
    c.attack.n_compromised = 1
    c.attack.position_lie = a.lie
    c.sim.T = a.T
    c.sim.record_level = "minimal"
    t = time.time()
    out, scn = run_pair(c, fleets=("naive", "gated", "oracle"))
    print("wall", round(time.time() - t), "liar", scn.liars.nonzero()[0])
    for f, res in out.items():
        print(f, {k: res.metrics[k] for k in KEYS})
