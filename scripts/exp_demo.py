"""E1: demo_district, four fleets x seeds. Writes results/demo/<stamp>/rows.csv."""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "sim.T=360",
                              "sim.record_level=minimal"])
    seeds = range(2 if a.quick else a.seeds)
    fleets = ["naive", "gated"] if a.quick else ["unfiltered", "naive", "gated", "oracle"]
    print(sweep("demo", grid(base, seeds, fleet=fleets), workers=a.workers, root=a.root))
