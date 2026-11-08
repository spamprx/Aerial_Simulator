"""E4: what caution costs when nobody lies (n_compromised = 0)."""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["attack.n_compromised=0", "sim.T=360", "sim.record_level=minimal"])
    seeds = range(1 if a.quick else a.seeds)
    scen = ["district"] if a.quick else ["district", "dense_rubble", "open_field"]
    print(sweep("no_liar", grid(base, seeds, world__scenario=scen, fleet=["naive", "gated"],
                                mission__audit_q=[0.1] if a.quick else [0.0, 0.1, 0.3]), workers=a.workers,
                root=a.root))
