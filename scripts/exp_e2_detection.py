"""E2 detection matrix: which liars are caught, how fast, and by which rule.

4 position-lie modes x claim_all on/off, naive vs gated, 20 seeds on district.
"""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["world.scenario=district", "attack.n_compromised=1", "attack.seek_quiet=false",
                              "sim.T=360", "sim.record_level=minimal"]).override(a.set)
    seeds = range(1 if a.quick else (a.seeds if a.seeds != 30 else 20))
    modes = ["fake_sweep_naive", "true_pose"] if a.quick else ["fake_sweep_naive", "range_aware", "radio_shadow",
                                                               "true_pose"]
    g = grid(base, seeds, attack__position_lie=modes, attack__claim_all=["true", "false"],
             fleet=["naive", "gated"])
    print(sweep(a.name or "e2_detection", g, workers=a.workers, root=a.root))
