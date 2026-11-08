"""E2: caution (audit fraction q) vs completion time vs survivors missed, across liar modes, loss and range."""
from _common import Config, grid, parser
from aerial.experiments.runner import sweep

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    base = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "sim.T=360",
                              "sim.record_level=minimal"])
    if a.quick:
        g = grid(base, range(1), fleet=["naive", "gated"], mission__audit_q=[0.0, 0.2])
    else:
        g = grid(base, range(a.seeds), fleet=["naive", "gated"], mission__audit_q=[0.0, 0.05, 0.1, 0.2, 0.3],
                 attack__position_lie=["range_aware", "radio_shadow", "true_pose"], comms__p_drop=[0.0, 0.1, 0.3],
                 comms__R=[35])
    print(sweep("tradeoff", g, workers=a.workers, root=a.root))
