"""E3 cost of caution: audit fraction q against the hardest liar (true pose, fake coverage), 30 seeds.

Only audits (R5) can catch this liar, so the catch rate should follow P(catch) >= 1 - (1 - q)^k.
"""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["world.scenario=district", "attack.n_compromised=1", "attack.position_lie=true_pose",
                              "attack.claim_all=true", "attack.seek_quiet=false", "sim.T=360",
                              "sim.record_level=minimal", "fleet=gated"]).override(a.set)
    qs = [0.0, 0.3] if a.quick else [0.0, 0.05, 0.1, 0.2, 0.3]
    print(sweep(a.name or "e3_caution", grid(base, range(1 if a.quick else a.seeds), mission__audit_q=qs), workers=a.workers,
                root=a.root))
