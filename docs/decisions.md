# AERIAL v2 decisions

Locked decisions (V*) and defaults (A*), copied from PLAN.md §1–§2. A decision beats a task card. Add new decisions here with an ID and a one-line reason, and mention them in the commit message.

## 1. Decisions locked in the Q&A on 9 Nov

| ID | Decision |
|---|---|
| V1 | Full plan: about 3 weeks of building plus 1 buffer week, run as parallel streams. |
| V2 | The liar fakes its **coverage** (it claims cells were swept and empty) and its **position** (it broadcasts a fake trajectory). Fake maps, fake survivors and framing are not v2 attacks; the epistemic layer still resists framing by construction. |
| V3 | **Knowledge + audits.** A "cleared" claim is *known* if I saw the cell myself, if ≥ F_left + 1 distinct drones reported it, or if a known-honest drone did. **Audits** re-check a fraction **q** of each teammate's claims; q is the caution knob. |
| V4 | **Limited radio range with relaying**: gossip over a time-varying disk graph with drops. |
| V5 | **2.5D height-map world**, with full 3D flight. |
| V6 | **Alive/dead is a belief** built from heat and motion cues. Drones hover to confirm, using a sequential probability ratio test (SPRT). |
| V7 | **Streamlit + Plotly 3D.** |
| V8 | The liar is caught by **UWB ranging, camera sightings, shared ranges and audits**, plus accusations that drones relay and corroborate. |
| V9 | **Dynamic partition**: each drone's region is a geodesic Voronoi cell, with frontier search and work stealing. When measuring partition distances, unknown cells count as passable; navigation never treats them as passable. *The liar's swept area* = the cells whose "cleared" status rests **only** on the liar's claims, according to their provenance. |
| V10 | Naive and gated differ **only in trust**. There are four fleets: unfiltered, naive, gated and oracle. |
| V11 | Consensus runs inside the mission in three places: **transit formation**, **survivor localisation**, and **return & regroup**. Each uses either the mean or W-MSR. |
| V12 | If no liar is caught, **spend the audit budget, then stop**. Report known coverage and believed coverage separately, together with the residual risk. |

## 2. Defaults: override any of them before Phase 0 ends

| ID | Default | Why |
|---|---|---|
| A1 | Fresh repository in `new_project/`, with the package still named `aerial`. Validated generic modules are **ported along with their tests** (§7). The first build stays read-only as a reference. | Clean code without rewriting modules that are already proven. |
| A2 | **Agent-centric.** Each drone is a `DroneAgent` with private state. The simulator holds ground truth and gives each agent only its own observation. Physics and sensing are vectorised in the simulator. | Decentralisation by construction. |
| A3 | The λ₂, ER and degradation studies run in `aerial/lab/`, the first build's vectorised simulator for fixed topologies, ported. | Already validated, and they do not need the mission. |
| A4 | Before take-off, drones know only the arena box, the altitude band and the launch base. There is no prior map. | "Unmapped territory." |
| A5 | z ∈ [4, 15] m, a regulatory ceiling. Standing buildings (12–35 m) must be flown round; rubble (1–5 m) can be flown over. Three cruise layers (6, 9 and 12 m) are chosen by heading. | Gives "3D aerial traffic management" real content. |
| A6 | One gimbal camera serves both depth sensing and survivor detection. FOV_h is **90°**, with 150° as a study factor. FOV_v is 60° and the camera tilts 30° down, so it sees from 0° to 60° below horizontal. r_depth = 25 m, r_detect = 15 m, r_see = 20 m, refresh 5 Hz. | The user asked for 90° or 150°, so both are studied. |
| A7 | **One function, `viewed(c, pose)`, defines both coverage and survivor detection.** A cell's target point is the cell centre on top of its 2×2 terrain block, plus 0.3 m. The cell is viewed if that point is inside the frustum, within r_detect, in line of sight, and at an elevation of at least θ_min = 20°. A survivor is detected **iff** its cell is viewed, and the detector returns the **exact cell id**; the noisy position is used only for localisation. Drones are detected iff they are inside the frustum, within r_see and in line of sight, using the same line-of-sight code. Survivors are static and sit at cell centres. Only alive/dead is noisy. | This is what lets audits and camera checks yield **proofs**. |
| A8 | Message origin cannot be forged (signatures are modelled abstractly). Messages are broadcast only, with no equivocation. A liar can lie about itself and drop relays, but it cannot forge another drone's messages. | Keeps relaying sound and preserves the W-MSR assumptions. |
| A9 | Carried over from the first build: D2 (one Safe predicate), D3 (hold = brake), D4 (re-query) and D5 (UWB ranging, σ = 0.05 m clipped at 4σ, so ε′ = 0.2 m). | Validated in the first build. |
| A10 | Demo: N = 10 (9 honest + 1 liar), F = 1, R = 35 m, p_drop = 0.1, a 100 × 100 m district, 16 survivors (about 70% alive). Studies go up to N = 20 and F ∈ {1, 2}. | Within the proposal's limits. |
| A11 | Missions use dt = 0.1 s (the lab keeps 0.05 s) and T ≤ 240 s. Sensing and epistemic updates run at 5 Hz. The partition updates at 1 Hz and on events. Planning replans only when the current path is invalidated, staggered across drones. | Performance budgets (T4.3). |
| A12 | Dependencies: numpy, scipy (new; for csgraph, cKDTree and statistics), networkx, matplotlib, plotly, kaleido, streamlit, pytest and hypothesis. | |
| A13 | Recorder levels: `minimal` (metrics computed online), `standard` (per-step states and knowledge summaries) and `full` (per-agent coverage snapshots at 1 Hz). | Keeps sweeps light and the dashboard rich. |
| A14 | The liar has **no ground-truth access**: same sensors and same inbox as every other drone. | The user asked for the same privileges. |
| A15 | Terrain claims pass through the same provenance and knowledge machinery as coverage claims. | The gate's guarantee must not depend on which attack is running. |
| A16 | Partitioning is a pluggable strategy. `static_strips` is kept for debugging and as a fallback. | Mitigates risk R1. |
| A17 | **Timing model.** Each tick, every agent's `beacon()` claim is ranged and delivered within the same tick (age 0); adjacency and UWB ranges are taken at that tick. Events relay one hop per tick. A single helper, `staleness(age)`, uses v_max_3d = √(v_h² + v_z²) ≈ 5.39 m/s for cached beacons. R1 and R3 run on fresh beacons only. R2 checks the horizontal and vertical limits separately. | Stops honest drones being accused through timing slack (review finding 2). |
| A18 | **Terrain knowledge is an interval [lo, hi] for each 1 m cell.** A ray passing over a cell lowers hi; a ray hitting a cell raises lo. Rays traverse cells exactly (DDA). A cell is "free at z" iff hi + clearance ≤ z. | A wall hit proves only H ≥ z (review finding 3). |
| A19 | Every drone sends a beacon every tick, so silence carries information. A relayed `POSE` event at 1 Hz gives drones more than one hop away an approximate position for the partition. | Partition liveness (review finding 9). |
| A20 | **Right of way.** When two drones each hold because of the other, the lower-priority one backs away along an allowed heading until the other's gate opens. Priority: a drone climbing yields to one in level flight; otherwise the higher id yields. | Breaks symmetric deadlocks (review finding 8). |
| A21 | Performance fallback: if the bench budgets fail after vectorising, adopt numba. This needs team agreement. | |

