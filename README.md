# AERIAL v2: knowledge-gated UAV search and rescue

**Resilient 3D Aerial Traffic Management and Knowledge-Gated Coordination for UAV Fleets in Disaster Response.**
AI3403 Multi-Agent Systems, IIT Hyderabad, group *Aerial*.

A decentralised fleet of drones searches an **unmapped**, randomly collapsed city for survivors. Each drone sees only
through a narrow camera and talks only to nearby drones over a lossy radio. **One drone is a liar**: it fakes its
position, claims areas as "searched, nobody here" and hides the survivors it sees.

| | Naive fleet | Gated fleet |
|---|---|---|
| Trusts | every claim (W-MSR filters numbers only) | only what it can **prove** |
| Survivors found (1 liar) | 82–87% | **100%** |
| Survivors found (2 liars) | 56% | **100%** |
| Liar caught | never | 100% of runs, 0 false accusations |
| Cost | | about +90 s mission time |

> The filtering that makes a fleet resilient also makes it ignorant. We quantify that trade-off.

## See it

| | |
|---|---|
| `docs/media/viewer_collisions.html` | 3D replay: naive drones fly into a liar hiding in their path; gated drones hold back. **Next collision ⏭** button for presenting |
| `docs/media/viewer.html` | 3D replay: the liar claims a corner; the naive fleet misses its survivors, the gated fleet re-sweeps it |
| `docs/media/video_collisions.mp4`, `video_demo.mp4` | The same replays as offline videos (top-down, side by side, live stats) |
| `docs/story.md` | One mission told step by step, from take-off to landing |

The 3D pages are single HTML files: open them in a browser. They need internet once, to load three.js.

## Quick start

```bash
pip install -e ".[dev,dashboard]"
make test                                         # ~640 tests, about 30 s
python scripts/demo_once.py --T 240               # one mission, naive vs gated vs oracle, metrics printed
python scripts/export_viewer.py --preset collisions   # rebuild the 3D replay (about 6 min)
python scripts/make_video.py --preset collisions      # MP4 from the replay data (needs ffmpeg)
```

## Documentation

| File | Read it for |
|---|---|
| **`CONTEXT.md`** | Everything: repository map, how to do each task, config knobs, experiments, the GPU server, which code produced which result, open items |
| `EXPERIMENTS.md` | Each experiment's question, design, results and data format |
| `docs/story.md` | The mission walk-through (how drones map, split the work, find survivors, catch the liar, come home) |
| `PLAN.md` | The full design and task cards |
| `docs/decisions.md` | The locked design decisions |
| `presentation.tex` | The slides (compile on Overleaf with `figures/`) |
| `docs/history/CONTEXT_first_build.md` | The first build's rebuild plan |

## How it works

| Layer | Method | Course topic |
|---|---|---|
| Transit | Consensus formation with W-MSR | consensus, resilient consensus |
| Exploration | Height-interval terrain mapping from the camera; geodesic Voronoi partition with work stealing | coverage, task allocation |
| Communication | Disk graph with packet loss, relaying and re-query | networks, information spread |
| Liar detection | S5 possible worlds over sets of liars; proofs from physics (radio range, speed, camera) and audits | epistemic logic, Kripke models |
| Safety | Knowledge gate: move only if K_i(Safe) | behavioural planning with epistemic logic |
| Survivors | Alive/dead belief (SPRT); W-MSR-filtered positions | belief vs knowledge, distributed estimation |
| Grading | Offline STL robustness | signal temporal logic |
| Theory studies | λ₂ convergence, (r,s)-robustness, Erdős–Rényi, degradation (`aerial/lab/`) | graph theory |
