"""Render naive vs gated on demo_district as one interactive HTML page (docs/media/demo.html)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from aerial.core.config import Config
from aerial.experiments.runner import run_pair
from aerial.viz.dashboard_data import fraction_knowing
from aerial.viz.scene3d import scene_figure

if __name__ == "__main__":
    c = Config().override(["world.scenario=demo_district", "attack.n_compromised=1", "sim.T=240",
                           "attack.position_lie=range_aware", "attack.claim_all=true", "attack.seek_quiet=false", "sim.record_level=standard"])
    out, scn = run_pair(c, fleets=("naive", "gated", "oracle"))
    html = ["<html><head><meta charset='utf-8'><title>AERIAL v2 demo</title></head><body style='font-family:sans-serif'>",
            "<h2>AERIAL v2 demo: naive vs gated, same seed, one liar</h2>",
            "<p>Blue = honest drones, red = liar (orange once caught). Diamonds = survivors: red until found, "
            "green = found alive, grey = found dead. Thin blue lines = camera frustums. Press play or drag the slider.</p>"]
    first = True
    for name, res in out.items():
        fr = fraction_knowing(res)
        caught = float(res.t[1:][np.argmax(fr > 0)]) if fr.size and fr.max() > 0 else None
        m = res.metrics
        fig = scene_figure(res, scn, title=f"{name}: found {m['found']}/{m['survivors']}, "
                           f"missed in liar's area {m['missed_in_claim']}, collisions {m['collisions']}, "
                           f"done at {m['completion_time']:.0f} s, liar known by all at {m['t_all_know']} s",
                           n_frames=80, caught_at=caught)
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        first = False
    html.append("</body></html>")
    p = Path("docs/media/demo.html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(html))
    for name, res in out.items():
        print(name, {k: res.metrics[k] for k in ["found", "missed_in_claim", "collisions", "completion_time", "t_all_know"]})
    print(p.resolve())
