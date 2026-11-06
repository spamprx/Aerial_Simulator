"""E1 headline: does gating help across many terrains, not just the demo seed?

district scenario, one liar (range_aware fake position + claims its whole corner), 30 seeds (terrain changes per
seed), four fleets. Paired by seed.
"""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["world.scenario=district", "attack.n_compromised=1", "attack.position_lie=range_aware",
                              "attack.claim_all=true", "attack.seek_quiet=false", "sim.T=360",
                              "sim.record_level=minimal"]).override(a.set)
    seeds = range(2 if a.quick else a.seeds)
    fleets = ["naive", "gated"] if a.quick else ["unfiltered", "naive", "gated", "oracle"]
    print(sweep(a.name or "e1_headline", grid(base, seeds, fleet=fleets), workers=a.workers, root=a.root))
