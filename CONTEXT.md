# AERIAL v2: repository context

Read this file first. It describes what the project is, how the repository is laid out, how to do every common task,
what was run on the lab's GPU server, which result came from which version of the code, and what is still open.

> Course: AI3403 Multi-Agent Systems, IIT Hyderabad. Group: **Aerial**.
> Title: *Resilient 3D Aerial Traffic Management and Knowledge-Gated Coordination for UAV Fleets in Disaster Response.*
> Suggested short project name (not yet adopted): **AEGIS**, Aerial Epistemic Gating In Search-and-rescue.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Quick start on a new machine](#2-quick-start-on-a-new-machine)
3. [Repository map](#3-repository-map)
4. [How the simulation works](#4-how-the-simulation-works)
5. [How to do common tasks](#5-how-to-do-common-tasks)
6. [Configuration knobs](#6-configuration-knobs)
7. [Experiments and results](#7-experiments-and-results)
8. [The GPU server: what was run and how](#8-the-gpu-server-what-was-run-and-how)
9. [Which code produced which result](#9-which-code-produced-which-result)
10. [Presentation and deliverables](#10-presentation-and-deliverables)
11. [Open items and next steps](#11-open-items-and-next-steps)
12. [Lessons and gotchas](#12-lessons-and-gotchas)

---

## 1. The project in one page

**The problem.**
- After a disaster, a decentralised fleet of UAVs must find survivors in **unmapped** 3D terrain where buildings have
  collapsed at random.
- Each drone sees only through a limited field-of-view camera and talks only to neighbours in radio range. Packets
  drop, and messages are relayed drone to drone.
- **One drone may be a liar.** It fakes its position, claims areas as "searched, nobody here" without searching them,
  and hides the survivors it sees.

**Thesis.** *The filtering that makes a fleet resilient (W-MSR) also makes it ignorant.* A drone should act only on
what it **knows**, not on what it is told.

**Two fleets, same drones, different trust:**

| Fleet | What it trusts | What it does |
|---|---|---|
| **naive** | believes every claim | filters numeric values with W-MSR |
| **gated** | acts only on what it can prove | catches liars, audits teammates' claims, and gates every move on K_i(Safe) |

There are two reference fleets as well: `unfiltered` (plain mean, trusts everything) and `oracle` (told the liar at
t = 0). We decided not to show the oracle in the presentation.

**How a liar is caught.**
- Each drone keeps every possible set of at most F liars (10 sets for N=10, F=1; 191 for N=20, F=2). Evidence rules
  sets out; "I *know* j lies" means j is in every remaining set (S5 possible worlds).
- **Proofs** (rules an honest drone can never trigger):

| Rule | Evidence |
|---|---|
| R1 | the UWB range contradicts the reported position |
| R2 | an impossible speed |
| R3 | claims to be out of radio range, yet was heard |
| R4 | the camera sees the claimed spot empty |
| R5 | an audit finds a survivor in a cell the drone claimed was empty |

- **Weaker evidence:** R6 (a teammate's shared range contradicts the drone) and an accusation each only prove
  "one of these two lies". So a liar framing someone proves nothing.
- **Exclusion:** once caught, the liar's claims are discarded, the cells only it vouched for reopen, and the dynamic
  partition re-sweeps them.

**Headline results.**
- **Survivors:** gated finds **100%** in every setting tested. Naive finds 82–87% with one liar and 56% with two.
- **Liar caught:** in 100% of gated runs where it fakes its position. **0 false accusations** in 870 runs on the
  server.
- **Cost of caution:** the gated fleet is slower, by about +90 s at the baseline.

Details are in §7.

**Further reading:**
- `docs/story.md`: one mission told step by step.
- `PLAN.md`: the full design and the task cards.
- `docs/decisions.md`: the locked decisions.
- `EXPERIMENTS.md`: the experiment guide.

---

## 2. Quick start on a new machine

```bash
unzip AERIAL_v2_repo_*.zip && cd new_project      # or wherever you unpacked it
python3 -m venv .venv && source .venv/bin/activate  # optional
pip install -e ".[dev,dashboard]"                   # numpy, scipy, networkx, matplotlib, pytest, hypothesis, streamlit, plotly
make test                                           # ~640 tests, about 30 s; all should pass
python scripts/demo_once.py --T 240                 # one mission: naive vs gated vs oracle, prints the metrics
xdg-open docs/media/viewer_collisions.html          # the 3D replay (internet needed once for three.js)
```

- **Python:** 3.10 or newer (developed on 3.13 locally; the server uses 3.11).
- **ffmpeg** is needed only to make MP4 videos.
- **torch** is needed only for the optional GPU ray casting (§8).
- **The `slow` and `bench` pytest markers** are skipped by default. Run them with `make slow` / `make bench`.

---

## 3. Repository map

```
aerial/                     the package
  core/        config.py (all knobs, validation, JSON, dotted overrides), rng.py (paired random streams),
               dynamics.py (3D double integrator + yaw), simulator.py (the tick loop), result.py (SimResult, recorder),
               types.py (observations, actions, beacons)
  world/       terrain.py (city + random collapse, 2.5D height map), scenario.py (survivors, liars, claim regions,
               scenarios), geometry.py (ray casting, line of sight, `viewed`), gpu.py (optional CUDA ray casting)
  sensing/     camera.py (depth rays -> terrain intervals, viewed cells, survivor looks, drone sightings)
  comms/       network.py (disk graph + drops + re-query + relaying/gossip), messages.py (event kinds), graph.py
  mapping/     maps.py (per-drone terrain intervals [lo,hi] and coverage map, every claim tagged with its reporters)
  planning/    nav.py (navigation grid, Dijkstra paths, geodesic Voronoi partition)
  agents/      mission.py (THE drone: all four fleets, mission state machine, audits, gate, escape), base.py
  epistemic/   hypotheses.py (possible faulty sets, S5 knowledge), knowledge.py (belief/knowledge standards),
               evidence.py (proof rules R1-R6), gate.py (the motion gate K_i(Safe)), belief.py (alive/dead SPRT),
               nested.py (K_i K_j, world counting), kripke.py (ported S5 engine)
  resilience/  liar.py (the liar agent: motions, position lies, fake coverage), aggregators.py (mean, W-MSR),
               robustness.py ((r,s)-robustness)
  spec/        stl.py (ported STL robustness engine), mission_specs.py (safety / find / know / cover specs)
  analysis/    mission.py (per-run metrics)
  experiments/ runner.py (build_sim, run_one with on_step hook, run_pair with pairing check, sweep with kept data)
  viz/         scene3d.py (Plotly 3D, used by the Streamlit dashboard), dashboard_data.py
  lab/         the FIRST build, ported unchanged, for the lambda2 / Erdos-Renyi / degradation theory studies
scripts/       experiments (exp_*.py), analysis (analyze_*.py), replay/video/figures (export_viewer.py,
               make_video.py, make_slide_figures.py, make_demo_html.py), demo_once.py, lab/ (theory studies)
viewer/        template.html: the Three.js 3D replay page (export_viewer.py fills it with run data)
dashboard/     app.py: Streamlit dashboard (optional; slower than the replay pages)
tests/         mirrors the package; contracts/ (decentralisation, pairing, import guard), epistemic/ (soundness,
               Kripke cross-check), lab/ (ported first-build tests), mission/, world/, sensing/, viz/
configs/       default.json, lab_default.json
docs/          story.md (mission walk-through), decisions.md, media/ (replay pages, videos, replay data),
               report/lab_data/ (first-build study rows), history/CONTEXT_first_build.md (the original plan)
figures/       every figure used in the slides (regenerable, see section 5)
results/       raw data and statistics of every experiment (section 7); results/cache is a re-creatable run cache
presentation.tex          the slides (beamer, metropolis); compile on Overleaf together with figures/
PLAN.md                   the design and task cards (sections 1-4 hold the decisions; section 8 the cards)
EXPERIMENTS.md            the experiment guide (what each experiment is, data format, how to regenerate)
```

**Rules the code follows.**
- A drone sees only its own observation, and `agents/`, `epistemic/`, `mapping/` and `planning/` never import
  `aerial.world`. Tests enforce both.
- The environment's random streams draw full blocks every tick, so naive and gated runs of one seed are *paired*.
  `run_pair` asserts it.
- New optional features default to off, so older results stay reproducible.

---

## 4. How the simulation works

**One tick (0.1 s):**
1. every drone states its beacon (a liar may lie here);
2. the camera produces frames (5 Hz) and UWB ranges are measured;
3. the disk graph is built, packets drop, and beacons and relayed events are delivered;
4. each drone decides from its own observation;
5. the physics step runs;
6. the step is recorded.

**Mission state machine.**
- **TRANSIT:** formation flight with W-MSR consensus to the zone edge.
- **EXPLORE:** each drone sweeps its own region of the geodesic Voronoi partition, then steals work.
- **CONFIRM:** hover to decide alive/dead (SPRT belief).
- **AUDIT:** gated drones only.
- **RETURN**, then **LANDED**.
- Back-edges reopen work whenever cells revert (for example, when a liar's claims are discarded).

**Mapping unknown terrain.** For every 1 m cell, each drone keeps an interval [lo, hi] for the ground height:
- a ray passing over a cell lowers hi;
- a ray hitting a cell raises lo.

The drone flies only where hi + 1.5 m is below its layer, so **unknown space is never treated as free**.

**The gate (gated fleet).** It moves only if it *knows* the next 0.3 s plus a hard stop is safe against every place each
neighbour could be:
- an honest neighbour: a small ball around its report;
- a possible liar: the UWB range sphere, minus what the camera sees empty.

If not safe, the drone brakes, re-queries, gives way and eventually detours. If a proven liar is seen on camera, the
drone escapes away from it and routes around it.

**The liar** (`resilience/liar.py`) is honest during transit, then lies.

| Aspect | Options |
|---|---|
| Position lies | `true_pose`, `fake_sweep_naive` (claims to sweep its far corner; caught by range at once), `range_aware` (fits receivers' ranges), `radio_shadow` |
| Motion | `hover`, `loiter`, `shadow` (tails a drone; rams, so out of scope), `park`, `ambush` (waits still in a drone's path; `ambush_rise` > 0 waits just above the camera's view; yields and never moves into anyone) |
| Coverage | `claim_all` claims its whole corner at once; otherwise it claims cells as its fake sweep passes |

---

## 5. How to do common tasks

| Task | Command | Time |
|---|---|---|
| Run all fast tests | `make test` | about 30 s |
| Soundness under attack, baseline, dashboard smoke | `make slow` | about 10 min |
| One mission, three fleets, metrics printed | `python scripts/demo_once.py [--seed S] [--lie range_aware] [--T 240]` | about 3 min |
| **3D replay, main demo** (liar claims a corner, ambushes) | `python scripts/export_viewer.py` → `docs/media/viewer.html` | about 5 min |
| **3D replay, collision demo** (naive drones fly into the hidden liar) | `python scripts/export_viewer.py --preset collisions` → `docs/media/viewer_collisions.html` | about 6 min |
| Restyle a replay page without re-simulating | `python scripts/export_viewer.py --preset <name> --rebuild` | seconds |
| **MP4 video** (top-down, naive vs gated, live stats) | `python scripts/make_video.py [--preset collisions\|demo] [--speed 8] [--fps 20]` → `docs/media/video_<preset>.mp4` | 1–3 min |
| Slide figures (terrain, knowledge curves) | `python scripts/make_slide_figures.py` | about 4 min, cached afterwards |
| E1–E3 statistics and figures from kept rows | `python scripts/analyze_experiments.py` | seconds |
| Sweep statistics and figures from kept data | `python scripts/analyze_sweep.py` | seconds |
| Re-run an experiment (prefer the server, §8) | `cd scripts && python exp_sweep.py --workers 6` (also `exp_e1_headline.py`, `exp_e2_detection.py`, `exp_e3_caution.py`; `--quick` for a smoke test) | §7 |
| Theory studies (first build) | `cd scripts/lab && python exp_lambda2.py --quick` (also `exp_er_robustness.py`, `exp_degradation.py`) | minutes |
| Streamlit dashboard | `make dashboard` | slow on the first run of a config |

**The 3D replay pages** are a single self-contained HTML file each.
- **Deep links:** `viewer.html#t=60`, `#view=gated&cam=top&play=1`.
- **Presenting:** a "Next collision ⏭" button, clickable timeline marks, and View → Hide panels for a clean screen.
- **Internet:** they need it once for three.js. The MP4 videos need nothing.

**Adding a replay preset.** Add an entry to `PRESETS` in `scripts/export_viewer.py` (config overrides, run length,
output paths). `make_video.py` reads the matching `*_data.json`.

---

## 6. Configuration knobs

All knobs live in `aerial/core/config.py`. Override them with dotted strings, e.g.
`Config().override(["comms.R=30", "fleet=naive"])`. The ones that matter most:

| Knob | Default | Meaning |
|---|---|---|
| `N`, `F` | 10, 1 | drones; assumed maximum number of liars (≤ 2) |
| `fleet` | gated | `unfiltered` / `naive` / `gated` / `oracle` |
| `world.scenario` | district | `district`, `dense_rubble`, `open_field`, `demo_district` (frozen terrain) |
| `world.zone`, `world.arena` | 100, 120 m | search-zone side and arena side (arena = zone + 20) |
| `world.n_survivors`, `world.p_collapse` | 16, 0.4 | survivors; fraction of buildings collapsed |
| `sim.T`, `sim.dt`, `sim.seed` | 240 s, 0.1 s, 0 | run length, tick, seed |
| `sim.record_level` | standard | `minimal` (sweeps), `standard`, `full` |
| `comms.R`, `comms.p_drop` | 35 m, 0.1 | radio range; packet loss |
| `sensor.fov_h`, `sensor.r_detect` | 90°, 15 m | camera horizontal field of view; survivor detection range |
| `mission.audit_q` | 0.1 | fraction of each teammate's claims to re-check (the caution knob) |
| `mission.trust_gate` | **false** | naive and unfiltered also run the motion gate, believing everyone. **Set true for fair collision comparisons.** |
| `attack.n_compromised` | 0 | number of liars actually present (≤ F) |
| `attack.position_lie` | range_aware | `true_pose`, `fake_sweep_naive`, `range_aware`, `radio_shadow` |
| `attack.motion` | hover | `hover`, `loiter`, `shadow`, `park`, `ambush` |
| `attack.claim_all`, `attack.seek_quiet` | false, true | claim the whole corner at once; wait for a quiet spot before lying |
| `attack.ambush_lead`, `attack.ambush_rise` | 10 m, 0 m | ambush: how far ahead of the drone; how far above its height |
| `attack.start_time` | 12 s | when lying starts (after transit) |

---

## 7. Experiments and results

All of these ran on the lab GPU server (§8), except the localisation study. The raw data is in `results/`, and
`EXPERIMENTS.md` describes the formats. Means carry 95% Student-t intervals; naive vs gated is compared per seed with
a Wilcoxon signed-rank test.

| Experiment | Folder | Design | Headline |
|---|---|---|---|
| **Sweep** (newest reference) | `results/sweep/20261110-044831/` | Vary drones 5/10/15, liars 0/1/2 and city 70/100/130 m, one at a time; 20 seeds × naive and gated; 480 s limit; `trust_gate=true`. 280 runs. | Gated finds **100% in all 140 runs**; naive finds 82–87% (1 liar) and **56%** (2 liars). Collisions: 3 in 280 runs, no real difference. Gated is about +90 s slower (even with no liar), and with 5 drones or a 130 m city it rarely finishes within 480 s. Liar caught 100%. 0 false accusations. |
| E1 headline | `results/e1_headline/20261110-021208/` | 30 districts × 4 fleets, 360 s | Naive missed survivors in 30/30 runs, gated 0/30. Naive collided in 23/30 (these are crashes between honest drones, because naive had no safety check then). Gated 274 s vs naive 175 s. |
| E2 detection | `results/e2_detection/20261110-021208/` | 4 liar types × whole/gradual claims × 2 fleets × 20 seeds | Position liars caught 20/20 within 12–15 s by R1/R2. A liar that lies only about coverage: caught 13/20 by audits (R5), median 44 s. |
| E3 audit fraction | `results/e3_caution/20261110-024417/` | Hardest liar, q ∈ {0, .05, .1, .2, .3} × 30 seeds | Catch rate flat at 63% up to q = 0.2 (overlapping cameras already re-check >20%), then 80% at q = 0.3; missed survivors 1.1 → 0.5. |
| Localisation | `results/localisation/` | W-MSR vs mean, one lying observer | Mean dragged by the full lie; W-MSR within about 0.5 m. |
| **E1 rerun** (fair gate) | `results/e1_fair/20261111-110826/` | 30 districts × 4 fleets, 480 s, `trust_gate=true`, current code | Naive 82.7% = unfiltered 82.9% (W-MSR buys nothing for search); gated 100% = oracle 100%; gated 265 s vs oracle 271 s (p=0.94); 0 collisions. |
| **E2 rerun** | `results/e2_current/20261111-112553/` | as E2, `trust_gate=true`, current code | Same detection numbers as E2. All 22 collisions are vs the `fake_sweep_naive` hovering liar (gated 16 in 12/40 runs, naive 6 in 5/40): gated drones touch the proven liar because the shell check probes only along the motion direction. |
| **E3 rerun** | `results/e3_current/20261111-120513/` | as E3, current code | 63% for q ≤ 0.2 (identical runs: free overlap meets the budget), 83% at q = 0.3; missed 1.10 → 0.37. |

**Figures:**
- `figures/sweep_found.png`, `sweep_collisions.png`, `sweep_time.png`, `sweep_tradeoff.png`;
- `figures/e1_headline.png`, `e2_detection.png`, `e3_caution.png`;
- `figures/knowledge_curves.png`, `terrain.png`.

**Each sweep run keeps** `runs/run_NNNN.json` and `runs/run_NNNN.npz`:
- the metrics and each survivor's find time;
- every collision (who, when, where) and every proof of lying (who, against whom, which rule, when);
- landing times and the config;
- 1 Hz trajectories.

**Collision demo** (local, not a server experiment). With `ambush_rise=0.6`, `fake_sweep_naive` and
`trust_gate=true`, across 6 seeds:

| Fleet | Collisions with the liar | Survivors found |
|---|---|---|
| naive | **23**, in every seed | 12–14/16 |
| gated | 1 (the liar drifting into a stopped drone) | 16/16 |

The replay uses seed 0: naive hits the liar 6×, gated 0×.

---

## 8. The GPU server: what was run and how

> The password is **not** stored anywhere in this repository. Ask the team, or set up key login once with
> `ssh-copy-id <user>@<lab-gpu-server>`.

**Server facts** (checked 10 Nov 2026):
- **Host:** `<user>@<lab-gpu-server>`; the home directory is `<server-home>`.
- **GPUs:** 3 × NVIDIA RTX A5000 (24 GB each). **GPU 1 is used by another user, so use GPUs 0 and 2.**
- **CPUs:** 40 threads. **The lab asked that heavy jobs not load the CPU cores** (there was an admin notice after the
  first build's 24-process run).
- **Conda** is at `<server-home>/miniconda3` and is *not* on the PATH in a non-interactive SSH session. Run
  `eval "$(<server-home>/miniconda3/bin/conda shell.bash hook)"` first.
- **Env `arial`:** Python 3.11, numpy 2.4.6, scipy 1.17.1, **torch 2.5.1+cu121**. This env was created fresh on
  10 Nov: someone had renamed the original `arial` to `lalit` and deleted `~/Arial`, and neither was touched.
- **Code:** in `~/Arial`, an rsync copy of this repository with no git history. **Results:** in
  `~/Arial/results/{e1_headline, e2_detection, e3_caution, sweep}`.

**How the GPU is used.** The simulator is mostly CPU-bound (millions of small numpy calls on tiny arrays). Only the
camera ray casting is GPU-friendly, so it runs in torch when `AERIAL_DEVICE` is set:
- `cuda` means one GPU;
- `cuda:auto` spreads worker processes across the visible GPUs.

The CUDA path is float64 and matches numpy exactly (`tests/world/test_gpu.py`). Measured speed-up: **1.19× per run**
(102 s vs 122 s for one gated 360 s mission). Each worker still uses one CPU core, so **keep the number of processes
small**: 6 processes, 3 per GPU, at `nice -n 19`.

**Measured run times on the server** (10 drones, 100 m city): naive about 37 s per run (p95 50 s), gated about 62 s
(p95 92 s). For the cost of a run elsewhere, scale by drones × mission length.

**Recipe** (what was actually run):

```bash
# from the laptop: copy the code (never the results back over the server copy)
rsync -az --exclude .git --exclude '__pycache__' --exclude 'results*' --exclude 'docs/media' --exclude '*.pdf' \
      ./ <user>@<lab-gpu-server>:Arial/
# on the server
eval "$(<server-home>/miniconda3/bin/conda shell.bash hook)"; conda activate arial; cd ~/Arial
nvidia-smi                                   # check GPUs 0 and 2 are free
CUDA_VISIBLE_DEVICES=2 python -m pytest -q tests/world/test_gpu.py tests/contracts   # GPU path still matches
cd scripts
AERIAL_DEVICE=cuda:auto CUDA_VISIBLE_DEVICES=0,2 nohup nice -n 19 python exp_sweep.py --workers 6 --root ../results \
      > ../run_sweep.log 2>&1 &
grep -o "\[sweep\] [0-9]*/[0-9]*" ../run_sweep.log | tail -1      # progress; rows land in rows.jsonl as runs finish
# back on the laptop: pull results
rsync -az <user>@<lab-gpu-server>:Arial/results/sweep results/
```

**Timeline of what ran** (10 Nov 2026, IST):

| Time | What | Runs | Setup |
|---|---|---|---|
| 02:12–02:45 | E1 headline | 120 | GPU 0, 3 processes |
| 02:44–03:38 | E3 audit fraction | 150 | GPU 0, 3 processes |
| 02:12–03:34 | E2 detection | 320 | GPU 2, 3 processes |
| 04:48–05:35 | Parameter sweep | 280 | GPUs 0 and 2, 6 processes |

There were 0 failed runs. Stop everything with `pkill -u <user> -f exp_`.

---

## 9. Which code produced which result

The code kept improving after some experiments ran, so not every number comes from the current code. This matters
when quoting results:

| Result | Code state | What changed afterwards |
|---|---|---|
| E1–E3 (02:12–03:38) | v2 before the fixes | Naive had **no** motion gate (its collisions were between honest drones); gated could get **stuck** next to a hovering liar (7/30 E1 runs hit the time limit); the gate checked only one point ahead |
| Parameter sweep (04:48–05:35) | + `trust_gate` (fair safety check), two liars get separate corners, camera-based liar localisation with escape and route-around (stuck fix) | The gate still checked only the point straight ahead |
| Main replay `viewer.html` | between E1 and the sweep (`trust_gate` off, ambush without rise) | — |
| Collision replay `viewer_collisions.html` and videos | **current code**: the gate checks the whole reachable cone of the liar's range sphere; the liar yields and adopts a new spot after a standoff; ambush rise and lead | — |

**Update, 11 Nov 2026:** the repository code is identical (md5) to the server copy that ran the sweep, so the sweep *is*
current code; E1–E3 were rerun with it (`e1_fair`, `e2_current`, `e3_current`, statistics via
`scripts/analyze_report.py` → `results/report_summary.json`). **The code that produced `viewer_collisions.html` (ambush
rise/lead, liar standoff yield, cone gate, `export_viewer.py --preset`) is not in this repository or on the server**; only
its output survives. Treat that replay as an illustration.

To get every number from the current code, rerun the sweep (about 45 minutes on the server) and, if wanted, E1
(about 30 minutes). Expect the gated fleet to be somewhat slower with the stricter gate.

---

## 10. Presentation and deliverables

**`presentation.tex`** (beamer, metropolis theme). Compile on Overleaf with the `figures/` folder; there's no local
TeX engine. It has 13 content slides, about 11 minutes:

| # | Slide |
|---|---|
| 1 | Problem |
| 2 | System model |
| 3 | Assumptions |
| 4 | One mission end to end |
| 5 | Mapping and splitting the work |
| 6 | Knowing who lies (S5) |
| 7 | Proofs of lying |
| 8 | Naive vs gated |
| 9 | **Live demo** (play `docs/media/viewer.html` or `viewer_collisions.html`; the MP4s are the offline backup) |
| 10 | Results 1 (E1) |
| 11 | Results 2 (E2 + E3) |
| 12 | Limitations |
| 13 | Contributions (placeholders), then Takeaways |

**Other deliverables:**
- `docs/story.md`: a story-mode walk-through of one mission, take-off to landing.
- `EXPERIMENTS.md`: the experiment guide.
- `AERIAL_v2_bundle_*.zip`: a shareable bundle (code + data + figures + docs, without course PDFs or caches).
- Course deadlines from `MAS_Project.pdf`: report due 11 Nov 2026 23:59; presentation 12 Nov 2026.

---

## 11. Open items and next steps

1. **Slides vs newest results.** The Results slides still show E1–E3, which used the older code (§9). Swap in the
   sweep figures (`sweep_found.png`, `sweep_tradeoff.png`) and/or rerun with the current code.
2. **Placeholders:** member names and contributions in `presentation.tex`, and the project name (AEGIS was suggested).
3. **Not built yet** (from `PLAN.md`):
   - rule R7 (a coverage claim impossible from where the drone was);
   - heading-based altitude layers (layers are by drone id);
   - consensus-based regroup at the base (it's plain path planning);
   - `prioritise()` between safety and advance (they are summed);
   - full 3D detours.
4. **Model facts to state honestly:**
   - Trust costs *survivors*, not collisions, once both fleets share the safety check.
   - Caution costs about 90 s even with no liar.
   - The audit guarantee holds only for deliberately sampled audits.
5. **Git:** the repository has staged files but **no commit yet**. The first commit failed because GPG signing needs a
   passphrase prompt. Commit from a terminal where pinentry works.

---

6. **Report:** `report/` (IITH template, XeLaTeX + Biber + shell escape; `AerialReport_overleaf.zip`). Fill in the GitHub
   link (`\repolink` in `report/main.tex`) and check the group number in `report/Metadata/Metadata.tex`.

## 12. Lessons and gotchas

- **Floating-point edge in R1.** Range noise is clipped at exactly 4σ = ε′, so comparisons need `+1e-6`, or honest
  drones get accused.
- **Wall hits.** A ray hitting a wall proves only H ≥ z. Never store it as the height, or the air above towers looks
  free.
- **Range-consistent lies can't hide proximity.** A `range_aware` liar's fake position comes close whenever it really
  is close. To show naive drones flying into a liar, use `fake_sweep_naive`.
- **Make liars passive in demos** (yield, never move into a drone). Otherwise collisions are ramming, which the gate
  cannot prevent and which is out of scope.
- **Deadlocks.** A gated drone next to a proven liar can freeze. The fixes are the camera-located hazard plus escape
  plus route-around, and the liar giving way after a 5 s standoff.
- **Sweeps:** write rows as they finish (`rows.jsonl`) and catch per-run errors, so one crash doesn't lose a sweep.
- **Server sessions:** the SSH control socket can drop silently, so make watchers report connection loss. A `&`
  inside an `a && b &` chain puts `conda activate` in the background subshell.
- **Validate figures:** check each one visually (overlapping labels, zero-height bars that disappear).
