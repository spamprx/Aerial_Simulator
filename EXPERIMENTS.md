# AERIAL v2: experiments, data and figures

This bundle holds everything behind the AERIAL v2 results: the code, every experiment's raw data, the statistics,
the figures, the 3D replay, the slides source and the project documents.

## Contents

| Path | What it is |
|---|---|
| `README.md` | What the project is and how to run it |
| `docs/story.md` | Walk-through of one mission from take-off to landing: how the fleets differ, how the liar is caught |
| `CONTEXT.md` | **Start here:** the whole repository, every task, the GPU server, and which code produced which result |
| `PLAN.md`, `docs/decisions.md`, `docs/history/CONTEXT_first_build.md` | The plan with its task cards, the locked design decisions, and the first build's context |
| `proposal.pdf` | The original project proposal |
| `presentation.tex` + `figures/` | Slides (beamer, metropolis theme); compile on Overleaf together with `figures/` |
| `docs/media/viewer.html` | Interactive 3D replay of the demo mission, naive vs gated (open in a browser; needs internet once for three.js) |
| `docs/media/viewer_collisions.html` | 3D replay where naive drones fly into a liar hiding in their path, while gated drones hold back |
| `docs/media/video_*.mp4` | Offline MP4 versions of both replays (`scripts/make_video.py`) |
| `docs/media/viewer_data.json` | The data behind that replay |
| `results/` | Raw data and statistics for every experiment (below) |
| `aerial/`, `scripts/`, `tests/`, `viewer/`, `dashboard/`, `configs/` | Source code, experiment and figure scripts, tests (`pip install -e .[dev]`, then `pytest`) |

## Common setup

- **Fleet:** 10 UAVs, one of them a liar that fakes its position and claims a corner of the area as "searched".
- **Area:** 16 survivors in a randomly generated collapsed district.
- **Pairing:** the **naive** fleet (believes every claim) and the **gated** fleet (acts only on what it can prove) fly
  the *same* map with the *same* liar for each seed, so differences come from trust.
- **Statistics:** means with 95% Student-t intervals; naive vs gated compared per seed with Wilcoxon signed-rank tests.
- **Where it ran:** the lab server (6 low-priority processes; camera ray casting on GPUs 0 and 2).

## Experiments

### Sweep: one parameter at a time (newest, recommended reference)

- **Folder:** `results/sweep/20261110-044831/`. Statistics in `stats.csv` / `stats.json`.
- **Figures:** `figures/sweep_found.png`, `sweep_collisions.png`, `sweep_time.png`, `sweep_tradeoff.png`.
- **Design:** vary the number of drones (5/10/15), the number of liars (0/1/2) or the city size (70/100/130 m), with
  the other two at the baseline (10 drones, 1 liar, 100 m). 20 seeds per value, naive and gated, 480 s limit;
  280 runs in total.
- **Fair safety check:** the naive fleet runs the same predictive safety check as gated, but believes everyone
  (`mission.trust_gate=true`), so the fleets differ only in trust.

**Headline results**
- **Survivors found:** gated **100% in all 140 runs**. Naive 82–87% with 1 liar (worst run 69%), **56% with 2
  liars** (worst 44%), 100% with no liar. p < 0.002 everywhere a liar is present.
- **Collisions:** 3 in 280 runs (gated 2, both with the liar; naive 1). There is no real difference once both fleets
  have the safety check.
- **Completion time:** gated is slower, by about +90 s at the baseline (269 vs 175 s) and about +90 s even with no
  liar. With 5 drones, or in the 130 m city, gated rarely finishes within 480 s, so its time there is the limit.
- **Liar detection:** caught in 100% of gated runs. **0 false accusations.**

### E1: does gating help? (30 districts)

- **Folder:** `results/e1_headline/`. Figure: `figures/e1_headline.png`.
- **Design:** four fleets (unfiltered, naive, gated, oracle), one liar, 360 s limit.
- **Results:** naive missed survivors in 30/30 runs, gated in 0/30; naive collided in 23/30, gated 0/30; the liar was
  caught 30/30; gated mean time 274 vs 175 s.
- **Caveats:** run with older code. The naive fleet had **no** predictive safety check, so its collisions are crashes
  between honest drones (not trust-related), and the gated fleet had a stuck-near-the-liar bug that was fixed later.
  The sweep supersedes it.

### E2: which liars are caught? (20 runs per liar type × whole-area / gradual claims × 2 fleets)

- **Folder:** `results/e2_detection/`. Figure: `figures/e2_detection.png`.
- **Results:**
  - Liars that fake their position: caught 20/20 within 12–15 s, by the radio-range (R1) or speed (R2) checks.
  - A liar that flies its true position and fakes only coverage: caught 13/20 by audits (R5), after a median 44 s.
- **Caveat:** same older code as E1.

### E3: audit fraction q (30 runs per q, hardest liar)

- **Folder:** `results/e3_caution/`. Figure: `figures/e3_caution.png`.
- **Results:** catch rate 63% for q = 0–0.2 and 80% at q = 0.3 (survivors missed 1.1 → 0.5). Overlapping cameras
  already re-check more than 20% of the liar's claims, so audits only change the outcome at q ≥ 0.3.
- **Caveat:** same older code as E1.

### Collision demo (local, 6 seeds)

- **Config:** `export_viewer.py --preset collisions`. The liar claims to be sweeping its far corner, and waits, still,
  0.6 m above a drone's height in its path (just above the camera's view). Both fleets run the same safety check
  (`trust_gate`).
- **Result:** naive hit the liar 23 times, in every seed. Gated was hit once (the liar drifting into a stopped drone)
  and found 16/16 survivors every time.
- **Replay:** uses seed 0 (naive 6 hits, gated 0).

### Survivor localisation: W-MSR vs plain mean

- **Folder:** `results/localisation/`. Script: `scripts/exp_localisation.py`.
- **Design:** 2–8 drones estimate a survivor's position while one liar reports it offset by 0–20 m.
- **Result:** the plain mean is dragged by the full lie; W-MSR stays within about 0.5 m.

`results/summary.json` holds the E1–E3 statistics, and `results/*/rows.csv` the raw rows (one per run). All E1–E3
runs together: 590, with 0 false accusations.

## Data format (sweep)

Each run `k` has two files in `runs/`:

- `run_k.json`:
  - `metrics`: survivors found and missed, collisions, completion time, catch times, …
  - `found_times`: survivor cell → time first seen
  - `collisions`: time, the two drones, whether the liar was involved, position
  - `proofs`: time, accuser, accused, rule
  - `landed`: landing time of each drone
  - `config`: the full configuration used
- `run_k.npz`:
  - `t`, `pos` (time × drones × xyz, 1 Hz), `yaw`
  - `liars`, `survivor_cells`, `survivor_alive`, `claim_region`

## Where these ran

- **Server:** all E1–E3 and sweep runs ran on the lab GPU server, as 6 low-priority processes using GPUs 0 and 2.
  `CONTEXT.md` §8 has the recipe.
- **Code versions:** later code fixes mean the results come from different versions. `CONTEXT.md` §9 maps each result
  to its code state.

## Reproducing

```bash
pip install -e .[dev]
python scripts/analyze_sweep.py          # sweep statistics and figures from the kept data
python scripts/analyze_experiments.py    # E1-E3 statistics and figures
python scripts/exp_sweep.py --workers 6  # re-run the sweep (about 45 min on 6 processes)
python scripts/export_viewer.py          # re-make the 3D replay (about 5 min)
```
