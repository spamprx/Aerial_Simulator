"""Fleet comparison across setups: change ONE parameter at a time, everything else at the baseline.

Sweeps (baseline: 10 drones, 1 liar, 100 m city; the baseline setting is shared and run once):
  drones   N            5, 10, 15
  liars    n_liars      0, 1, 2      (the drones assume F = max(1, n_liars))
  city     zone side    70, 100, 130 m   (arena = zone + 20 m)
Naive vs gated, 20 seeds each. Naive also runs the motion gate, believing everyone (mission.trust_gate), so the
fleets differ only in trust. Every run's detail is kept under runs/ for later analysis.
"""
from _common import Config, parser
from aerial.experiments.runner import sweep

BASE = ["world.scenario=district", "attack.position_lie=range_aware", "attack.claim_all=true",
        "attack.seek_quiet=false", "sim.T=480", "sim.record_level=minimal", "mission.trust_gate=true"]


def setting(N=10, liars=1, zone=100):
    return {"N": N, "liars": liars, "zone": zone}


def settings():
    out = {}
    for N in (5, 10, 15):
        out[(N, 1, 100)] = setting(N=N)
    for L in (0, 1, 2):
        out[(10, L, 100)] = setting(liars=L)
    for Z in (70, 100, 130):
        out[(10, 1, Z)] = setting(zone=Z)
    return list(out.values())


def config(s, fleet, seed):
    F = max(1, s["liars"])
    return Config(N=s["N"], F=F).override(BASE + [
        f"attack.n_compromised={s['liars']}", f"world.zone={s['zone']}", f"world.arena={s['zone'] + 20}",
        f"fleet={fleet}", f"sim.seed={seed}"])


if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    seeds = range(1 if a.quick else (a.seeds if a.seeds != 30 else 20))
    cfgs = [config(s, f, sd) for s in settings() for sd in seeds for f in ("naive", "gated")]
    if a.quick:
        cfgs = [c.override(["sim.T=20"]) for c in cfgs]
    print(len(cfgs), "runs", flush=True)
    print(sweep("sweep", cfgs, workers=a.workers, root=a.root, keep=True))
