# AERIAL v2: rebuild plan and task cards

Written for the six Aerial team members and for the AI coding agents that will implement the cards. This version has
been revised after an independent red-team review, whose 14 findings are all included (see "What the review changed"
at the end).

## Context

The first build (`../project/Aerial-Simulator-main`: about 6.3k lines, 865 tests) met every target in `docs/history/CONTEXT_first_build.md` §2,
but it rested on assumptions the team now wants removed:

- the map was known, and routes were planned before take-off;
- survivors were "found" when a run was graded, not by any sensor;
- the radio graph was effectively complete (60 m range in an 80 m arena);
- a liar's sector was never re-swept;
- W-MSR did nothing in the demo;
- the code grew from flat modules into subpackages and still carries re-export shims.

**v2 rebuilds AERIAL from an empty repository**, with its interfaces planned first, around a new mission.

- A decentralised fleet flies into **unmapped 3D terrain** where buildings have collapsed at random.
- Each drone finds **survivors** with a **limited field-of-view camera** and judges whether they are **alive or dead**.
- Drones coordinate over **limited-range radio, relaying messages for one another**.
- The fleet **catches a liar** that fakes its coverage and its position, then **re-sweeps what the liar claimed**.

The thesis stands: *the filtering that makes a fleet resilient also makes it ignorant*. It is now measured as
survivors missed against mission time.

Every proposal promise is kept:

- double-integrator UAVs in ℝ³ on a time-varying disk graph with packet drops;
- the consensus behaviours, now inside the mission;
- W-MSR with F ≤ 2;
- (r,s)-robustness, plus the Erdős–Rényi (ER) and degradation studies;
- offline STL;
- possible worlds under S5 with nested knowledge, and a gate that changes behaviour;
- naive vs gated over seeded runs, with a trade-off curve;
- a Streamlit dashboard and reproducible scripts.

---

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

### Parameter defaults

| Item | Default |
|---|---|
| Arena | 120 × 120 m, made of a 100 × 100 m search zone and a base strip. Terrain grid 1 m (stored as intervals); coverage grid 2 m (2,500 cells). |
| Buildings | Blocks about 20 m wide; streets 8–12 m; heights 12–35 m; p_collapse = 0.4. Rubble is 1–5 m high and spills into the streets; some piles rise above the cruise layers. |
| Drone | v_h = 5 m/s, v_z = 2 m/s (v_max_3d ≈ 5.39 m/s), a_max = 4 m/s², yaw rate 90°/s, v_ref = 3 m/s. |
| Radio | R = 35 m, p_drop = 0.1, at most 64 events per message. Priority order: ACCUSE > SURVIVOR > CLEARED > AUDITED > POSE > TERRAIN > RANGES. |
| Silence rule | A drone not heard for L = ⌈ln 1e-9 / ln p_drop⌉ ticks is outside R − v_close·L·dt, with v_close = 2·v_max_3d. That is 25.3 m at R = 35 m, p = 0.1. This is the only probabilistic assumption. |
| Safety | d_safe = 1 m, d_obs = 1.5 m. r_sep, street width, layer spacing and formation spacing are **checked by T0.3** against the gate clearance. |
| Gate | Look-ahead τ = 0.3 s; patience 1 s. The grid (h = d_safe) is used only in the near field. |
| Alive/dead | Heat ~ N(34, 1) °C when alive; when dead it falls towards an ambient 25–30 °C. Motion probability per look: 0.6 conscious, 0.25 unconscious, 0.02 dead. SPRT thresholds at 0.95 and 0.05. CONFIRM is capped at T_conf = 8 s. |

---

## 3. System model

**World.** The ground truth, which agents never see:

- a 2.5D height map built from city blocks, with a random subset of buildings collapsed and spilling into the streets;
- a ground-truth `viewable` mask, the cells that can be viewed from some pose in the band;
- survivors placed only in viewable cells, at cell centres;
- each survivor's alive/dead status and vital-sign signature.

**Drone.** A 3D double integrator with separate horizontal and vertical limits, plus first-order yaw with a rate limit.
`step_double_integrator` is ported from the first build.

**Sensors** (vectorised in the simulator, at the sensor rate):

1. **Depth camera.** Exact traversal of the rays through the frustum produces terrain-interval updates (A18).
2. **Coverage.** `viewed(c, pose)` (A7).
3. **Survivor detector.** Returns the exact cell, a noisy position, and one heat reading and one motion bit per look.
4. **Drone sightings.** Anonymous positions inside the frustum, and a `VisibleVolume` whose `contains()` calls the
   same line-of-sight code. Because the camera looks only at or below horizontal, **camera proofs can only refute
   claims at or below the observer**.
5. **UWB range** on every beacon, taken at the transmit tick. A liar cannot alter it.

**Comms.** `A[i, j]` = i receives j if ‖pᵢ − pⱼ‖ ≤ R and the packet is not dropped. Drops are drawn as a full block
every tick.

- **Beacons** travel one hop, every tick, and carry the claimed pose, velocity and sequence number.
- **Events** are flooded with deduplication on (origin, seq), one hop per tick, under the priority cap:
  - `CLEARED(t, cells, claimed pose)`: cells swept and found empty;
  - `SURVIVOR(cell, est)`, `TRACK_EST` and `LOOKS`: survivor reports, position estimates and alive/dead evidence;
  - `POSE`: the 1 Hz relayed position;
  - `TERRAIN`: terrain observations;
  - `RANGES(j, seq_j, range, p_r)`: a measured range to j, with the reporter's own pose;
  - `AUDITED(j, cells)`: cells of j's that this drone re-checked;
  - `ACCUSE(j, rule, evidence refs)` and `KNOWN_FAULTY`: liar evidence and the sender's known-faulty set.
- **Re-query (D4):** a drone that is holding pulls fresh beacons and the events it is missing. The drops are drawn as a
  full block from `rng_requery` every tick.

**Liar.** A `LiarAgent` with the same sensors and inbox as everyone else. It **broadcasts its true pose during
transit**, so it is not caught before the mission starts.

- **Motion:** `hover`, `loiter`, `shadow` or `partial_sweep`.
- **Position lie:**
  - `fake_sweep_naive` broadcasts a lawnmower track and ignores ranging. This is the easy baseline for E5, caught by
    R1 within seconds.
  - `range_aware` keeps every fake pose consistent with the ranges its current receivers measure, where the geometry
    allows it.
  - `radio_shadow` stays out of direct range and lets its claims arrive only by relay.
  - `true_pose` with fake coverage only.
- **Coverage lie:** emits `CLEARED` claims for a claim region stored in the scenario, and suppresses every survivor
  report.

**Timing (A17), one tick:**

1. `beacon()` claims from every agent.
2. Sense at p(k): frames (every s ticks) and UWB ranges.
3. Communicate: adjacency at p(k); drops; deliver beacons (age 0), the events relayed from k − 1, and re-queries.
4. `step(obs)` for every agent, liars included.
5. Integrate to p(k+1), and log true separations and clearances.
6. Record at the configured level, computing online metrics.

**Randomness.** `SeedSequence(seed)` spawns:

- `world`: terrain, survivors, liar identity and claim region, launch poses;
- the environment streams `comm`, `attack`, `sense` and `requery`, each drawing **full blocks every tick**, checked by
  a `CountingGenerator`;
- `ctrl`: per agent, for audit sampling. It is not an environment stream.

---

## 4. The epistemic layer (the core; owned by Suhas)

**Worlds.** A world fixes:

- the faulty set S\* (|S\*| ≤ F);
- the terrain and the survivors;
- every drone's true trajectory.

Agent i's information set Iᵢ(w) factors as **Hᵢ** (the faulty sets consistent with i's evidence) × per-cell facts.
Hᵢ is enumerated exactly as uint32 masks, with i honest to itself:

| Fleet | |Hᵢ| |
|---|---|
| N = 10, F = 1 | 10 hypotheses |
| N = 20, F = 2 | 1 + 19 + 171 = 191 |

Knowledge of *who* is faulty is therefore exact S5: a fact is known when it is true in every world compatible with i's
information.

**If Hᵢ is empty, the assumption is broken** (more than F liars). Never compute `all()` over an empty set, because it
is vacuously true and would make every claim look known. Instead, raise `assumption_broken` and count **only i's own
observations** as knowledge.

**Evidence → constraints on S.** Hᵢ = {S : |S| ≤ F, i ∉ S, every constraint holds}.

| Rule | Evidence | Constraint | Rests on |
|---|---|---|---|
| R1 | A fresh beacon's pose disagrees with my UWB range taken at the same tick: \|‖p̂ⱼ − pᵢ‖ − rᵢⱼ\| > ε′ | j ∈ S | physics (A17) |
| R2 | j's own signed beacons imply a horizontal speed > v_h or a vertical speed > v_z | j ∈ S, and anyone can check it from j's own messages | physics |
| R3 | j heard on a fresh beacon claims a distance > R + ε′ | j ∈ S | physics |
| R4 | The whole honest-ball of j's claim, shrunk by v·(frame age), lies inside my visible volume, and nothing is sighted there | j ∈ S | A7, applies at or below the observer |
| R4b | An unexplained sighting at x | OR over every drone whose honest-ball does not contain x, **including silent ones** | A7 |
| R5 | **Audit**: I myself detect a survivor in a cell j claimed `CLEARED`. The cell is exact (A7). | j ∈ S | A7, static survivors |
| R6 | r's `RANGES(j, seq, ρ, p_r)` contradicts j's signed beacon `seq` | j ∈ S ∨ r ∈ S | physics, if r is honest |
| R7 | j's `CLEARED` at time t is impossible from every position consistent with my range history at t. Unknown terrain counts as see-through; any yaw is allowed. | j ∈ S | physics |
| Acc | ACCUSE(j) from drone a | a ∈ S ∨ j ∈ S | honest drones accuse only with proof |
| Self | Evidence the receiver can check itself, made only of j's own signed messages and the receiver's own knowledge | j ∈ S | as for the rule involved |

Consequences:

- **Framing never proves anything**: a false accusation only adds {liar ∨ victim}.
- **F + 1 distinct accusers prove it**, and so does a single accuser that is known honest.
- **Budget exhaustion**: once |KnownFaultyᵢ| = F, every other drone is known honest. With F = 1, catching the liar
  makes everyone else's claims knowledge in one update.

**How knowledge spreads, and why that matters.** At F = 1 a lone accusation leaves receivers knowing only "a or j". To
know which, a receiver needs a **second independent proof**. So gated drones **corroborate**:

- On receiving ACCUSE(a, j, evidence), a drone re-checks what it can itself:
  - its own range history for the same beacon `seq` (R1);
  - j's own signed messages (R2 and Self);
  - the claimed cell, if it can view it soon (R5).
- It also raises the audit priority on j.

Spreading is therefore a **threshold-2 (complex) contagion**, not plain SI. E3 compares it with both models.

At F = 2 with only one liar present, the budget is never exhausted. Completion then comes through audits and V12, and
a proof without self-verification needs 3 accusers. The report states this as a finding.

**Knowledge over claims.** Kᵢ(cleared c) holds if i viewed c itself, or if for every S ∈ Hᵢ some reporter of c is
outside S. The check runs on bitmasks, cells × hypotheses, recomputed incrementally. Kᵢ(free c) (from terrain
intervals) and Kᵢ(survivor at c) work the same way.

**Two standards, used by gated drones for different jobs:**

| Standard | Means | Used for |
|---|---|---|
| **Belief** Bᵢ | anyone reported it; this is also the naive fleet's only standard | allocating work: the partition and targets, so no work is duplicated |
| **Knowledge** Kᵢ | as defined above | audits, completion, re-sweeping, and survivor reports to rescuers |

**Audits (V3, V12).** A gated drone whose region is believed clear audits cells that are *believed but not known* to
be clear.

- **Sampling.** It samples uniformly within each teammate j's set of cells that only j vouched for, using its own
  `ctrl` stream.
- **Budget.** The budget for j is ⌈q·nⱼ⌉ cells. **An audit counts towards j's budget only if the auditor ≠ j.** The
  sample is **topped up** as new claims arrive.
- **Approach.** Auditors approach through the frontier, because the liar's block may be unmapped. If an audit cell
  turns out to be unreachable or unviewable, the drone resamples.
- **Stopping.** When the budget is spent with no catch, the drone stops. It reports:
  - the known-cleared and believed-cleared fractions;
  - the **achieved** audit fraction kⱼ/nⱼ for each teammate;
  - the residual risk: "one hidden survivor in j's cells would have been found with probability ≥ kⱼ/nⱼ".
- **What counts in the risk report.** Only audits by i itself, by known-honest drones, or by F_left + 1 distinct
  auditors.

**Cells that cannot be viewed.** If a cell is still not viewed after k attempts from the best reachable viewpoints,
the drone abandons it and records it as *unviewable (believed)*. It is reported separately and never counted as
cleared. Metrics use the ground-truth `viewable` mask.

**Exclusion and re-sweep (V9).** As soon as Kᵢ(j faulty) holds:

1. j is dropped from i's partition, so its region falls to i and the other honest neighbours;
2. j's claims are discarded, so cells **only j vouched for** revert to uncovered and are re-swept by normal
   exploration;
3. j becomes a hazard whose position is uncertain;
4. j is excluded from W-MSR, from survivor tracks and from belief fusion.

Partitions may disagree while knowledge spreads. This is harmless: at worst some work is duplicated for a while.

**The gate (it changes behaviour).**

**(a) Motion.** The plan checked is one tick of advance + safety, followed by maximum braking until τ. Kᵢ(Safe(plan))
must hold for every neighbour, for **every status the neighbour can have** in some S ∈ Hᵢ, and for every possible
position under that status:

- **Status honest:** a ball of radius ½·a_max·t² around p̂ + v̂·t. Using the reported velocity is sound under that
  status, because an honest report is true. This is much tighter than the old v_max·τ ball.
- **Status faulty:** the UWB shell around me, intersected with the points consistent with all but `slack` of the
  shared shells (the first build's rule, in place of least squares). It is limited to cells **not known to be
  occupied**, and the parts I can see are empty are removed, after shrinking those parts by v·(frame age + τ).
- **A possibly-faulty neighbour** gets the **union** of both sets.
- **A silent drone** gets "outside R − v_close·L·dt" (the silence rule).

Shells wholly out of reach are rejected analytically; grid cells are used only in the near field. If Kᵢ(Safe) fails,
the drone brakes, keeps the never-gated safety terms, and issues a re-query. Right of way (A20) applies; after
`patience` the drone detours, using the first build's rule that backing away is allowed only away from every
possibly-occupied region.

**(b) Clearing.** Only Kᵢ-cleared cells count as done.

**(c) Report.** A survivor is *confirmed* only if Kᵢ(survivor at c) holds and the alive/dead belief has been decided
(or capped as "unsure").

**(d) Done.** Every coverable cell is Kᵢ-cleared or abandoned as unviewable, or the audit budget is spent.

**Belief versus knowledge in the report.** Alive/dead is a **belief** Bᵢ (an SPRT posterior). Liar identity and
clearance are **knowledge** Kᵢ (S5).

**Nested and group knowledge (a proposal promise, so Must).**

- Kᵢ Kⱼ(L faulty) ≈ i holds j's `KNOWN_FAULTY` payload naming L, and Kᵢ(j honest). This is sound but not complete,
  and is checked against the exact Kripke value.
- Ground-truth Eᴳ and depth-k Eᵏ.
- The report explains why common knowledge Cᴳ is unattainable when packets can drop (coordinated attack).

**Abstraction size (a proposal promise).**

|Iᵢ| = Σ_{S∈Hᵢ} [ Π_j |cellsⱼ(status_S(j))| ] · 2^{u(S)}

Here u(S) is the number of cells still uncertain under S. It is computed in log space (logsumexp), is optional, and is
off in sweeps. Simply adding log|Hᵢ| to the uncertain-cell count over-counts.

**Soundness invariants.** For every honest i at every step:

- S\* ∈ Hᵢ;
- no honest drone is ever in KnownFaultyᵢ;
- every Kᵢ fact is true in ground truth;
- Hᵢ = ∅ ⇒ only own observations are treated as known.

These run first on the **replay harness** (T0.10) and later on 50 full attacked runs.

---

## 5. Mission flow and the demo story

**State machine:**

- TAKEOFF → TRANSIT (formation with W-MSR, broadcasting the true pose) → EXPLORE ⇄ CONFIRM (capped at T_conf)
  → AUDIT (gated only) → RETURN → REGROUP → LANDED.
- **Back-edges:** AUDIT → EXPLORE and RETURN → EXPLORE whenever new uncovered work appears, for example cells that
  revert after a late catch.

**`demo_district`.** A frozen seed: N = 10 with drone 7 as the liar, F = 1, R = 35 m, liar mode `range_aware` +
`hover`. The scenario stores the liar's claim region, which holds 3 survivors. The times below are illustrative, not
promised.

1. **Transit.** The line-abreast formation flies to the edge of the zone. In the *unfiltered* fleet a lying
   neighbour's offset bends the formation; W-MSR trims it.
2. **Exploration.** The liar hovers at a junction in partial radio shadow. It broadcasts a range-consistent fake sweep
   of the NE block and claims that block `CLEARED`.
   - **Naive fleet:** accepts the claims, finishes first, and misses all 3 survivors.
   - **Gated fleet:** the liar is caught by whichever rule fires first. Typically this is R6, once three or more drones
     range it from different places so that no fake pose fits every shell, or an R5 audit. Corroboration spreads the
     proof, the budget is exhausted, and the known-cleared fraction jumps. The liar-only cells revert, the partition
     re-forms, and two drones sweep the NE block, find the 3 survivors and confirm which are alive. The fleet then
     regroups and finishes later, with 0 collisions.

---

## 6. Architecture

```
aerial/
  core/        config.py  rng.py  types.py  dynamics.py  simulator.py  result.py  registry.py
  world/       terrain.py  survivors.py  geometry.py  scenario.py          (ground truth; agents never import it)
  sensing/     camera.py  detector.py  sightings.py  ranging.py
  comms/       graph.py (ported)  network.py  messages.py  gossip.py
  mapping/     provenance.py  terrain_map.py  coverage.py
  planning/    astar.py  partition.py  explorer.py
  agents/      base.py  controller.py  mission_fsm.py  transit.py  tracks.py  regroup.py  honest.py  gated.py  fleets.py
  resilience/  aggregators.py (ported)  robustness.py (ported)  liar.py
  epistemic/   kripke.py (ported)  hypotheses.py  evidence.py  multilateration.py  audit.py  accusations.py
               knowledge.py  gate.py  belief.py  nested.py  replay.py (test harness)
  safety.py    ONE Safe predicate (D2) + never-gated terms
  spec/        stl.py (ported)  mission_specs.py
  analysis/    mission.py  online.py  stats.py
  experiments/ runner.py
  lab/         first build's vectorised consensus simulator (λ₂ / ER / degradation)
  viz/         scene3d.py  plots.py  dashboard_data.py
dashboard/app.py   scripts/{exp_*,fig_*,validate_*}.py   configs/   tests/ (mirrors packages + contracts/ + bench/)   docs/
```

**Interfaces fixed in Phase 0.** Stubs come first; later cards only fill them in.

```python
@dataclass(frozen=True)
class AgentObservation:
    t: float; k: int; i: int
    own: DroneState                      # true pos, vel, yaw of agent i only
    frame: Optional[SensorFrame]         # terrain intervals, viewed cells, survivor looks (exact cell), sightings, visible volume
    beacons: Dict[int, ReceivedBeacon]   # sender -> claimed state, seq, age (0 = fresh), uwb_range (fresh only)
    events: List[Event]                  # newly received relayed events (deduplicated)
    static: StaticInfo                   # arena box, z band, base, N, F, own id; nothing about terrain

class DroneAgent(ABC):
    def reset(self, ctx: AgentContext) -> None: ...
    def beacon(self, own: DroneState, k: int) -> ClaimedState: ...  # honest: the truth; LiarAgent: its lie
    def step(self, obs: AgentObservation) -> Action: ...             # Action(u, yaw_rate, outbox, requery)
    def diagnostics(self) -> Dict[str, Any]: ...                     # must follow the extras schema

@dataclass(frozen=True)
class Event:
    origin: int; seq: int; t: float; kind: EventKind; payload: Any   # origin cannot be forged (A8)
```

**Extras schema** (T0.5). Each entry is recorded at `standard` and above; at `minimal` it is summarised online:

| Extra | Shape |
|---|---|
| `known_faulty` | (K, N, N) |
| `known_honest` | (K, N, N) |
| `n_hyp` | (K, N) |
| `gate_open` | (K, N), i.e. Kᵢ(Safe) |
| `holds` | (K, N) |
| `proof_log` | rows of (k, i, j, rule) |
| `wmsr_rejected` | (K, N, N) in transit and localisation |
| `formation_err` | (K,) |
| `known_cleared_frac` | (K, N) |
| `believed_cleared_frac` | (K, N) |
| `msgs_sent` | (K, N) |
| `mode` | (K, N) |
| `robust_ok` | sampled |

**Structural guard.** `agents/`, `epistemic/`, `mapping/` and `planning/` must not import `aerial.world`, apart from
shared types. A test enforces this.

---

## 7. What to port from the first build (`../project/Aerial-Simulator-main`)

| First build | v2 | Keep |
|---|---|---|
| `aerial/stl.py` | `aerial/spec/stl.py` | `Pred`, `Not`, `And`, `Or`, `Always`, `Eventually`, `Until`, `robustness`, `prefix_robustness` |
| `aerial/epistemic/kripke.py` | `aerial/epistemic/kripke.py` | `KripkeModel`, `Atom`, `K`, `E`, `C` (for the cross-check and the muddy-children example) |
| `aerial/resilience/robustness.py` | same path | `is_rs_robust`, `robustness_witness`, `max_r`, `resilience_conditions` |
| `aerial/resilience/aggregators.py` | same path | `MeanAggregator`, `WMSRAggregator`, `AggResult` |
| `aerial/network/graph.py` | `aerial/comms/graph.py` | `disk_adjacency`, `packet_keep_mask`, `laplacian`, `algebraic_connectivity`, `make_topology` |
| `aerial/core/dynamics.py` | same path | `step_double_integrator`, `clip_norm`, plus vertical limits and yaw |
| `aerial/stability.py`, `coordination/{consensus,formation,boids}.py`, `resilience/attacks.py`, `core/simulator.py` | `aerial/lab/` | the lab; transit reuses `formation_offsets` |
| `aerial/epistemic/worlds.py` (rules), `gate.py`, `detour.py` | reference only | shared-shell rule with slack, silence rule, back-away and detour rules |
| `scripts/exp_{lambda2,er_robustness,degradation}.py` and their `fig_*.py` | `scripts/` | rerun in the lab; compare against `docs/report/data/` |

`docs/history/CONTEXT_first_build.md` §8 and §10 still apply:

- separation uses only the drone's own velocity;
- DP-MSR damping under W-MSR;
- leader–follower pins 2F + 1 agents;
- W-MSR masks are **never** evidence of lying;
- r_sep must exceed the gate clearance;
- draw full random blocks every tick.

---

## 8. Task cards

Each card lists: Owner · Priority · Depends on · Files · Do · Done when.

**Priorities:**

- **Must**: on the critical path or behind a success criterion;
- **Should**: planned, and the first to be cut;
- **Could**: only if time allows.

**Owners**, following the proposal:

| Owner | Area |
|---|---|
| Rithvik (lead) | core, comms, runner, integration |
| Praneeth | navigation and coordination |
| Sanjana | resilience |
| Suhas | epistemic layer |
| Rohith | specifications, visualisation, analysis |
| Prabhu | world, sensing, scenarios, metrics |

**Can start immediately, in parallel:** T0.1–T0.6, T0.9, T1.1, T2.1, T3.3, and T0.10 once T0.5 exists.

### Rules for anyone (human or AI) picking up a card

1. Read §1–§4 and §6 first. A locked decision (V\*) or a default (A\*) beats a card.
2. Conventions:
   - `A[i, j]` means i receives j;
   - shapes are (N, 3) per step and (K+1, N, 3) for histories;
   - agents see only their own `AgentObservation`;
   - a liar is an agent, not a filter;
   - environment random streams draw full blocks every tick;
   - never iterate over a `set` where order matters.
3. Every card adds tests and leaves `pytest -m "not slow"` green in about 3 minutes or less. Anything slower is marked
   `@pytest.mark.slow`.
4. An optional feature must be **bit-identical when switched off**, and a test must prove it. Every number in the
   report comes from a script and from rows stored in the repository.
5. Record each new decision in `docs/decisions.md` and in the commit message.

> **Prompt template for an AI agent:** "You are working in the `aerial` v2 repository. Read `PLAN.md` §1–§4 and §6,
> then card **T?.?**. Implement that card only. Add the tests listed under *Done when*, run `pytest -m 'not slow'`,
> and report what you changed and any decision you had to make."

### Phase 0: foundations and contracts (week 1, days 1–3)

#### T0.1 Repository, tooling and CI · Rithvik · Must

- **Depends on:** nothing.
- **Files:** `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `.gitignore`, `README.md`.
- **Do:**
  - run `git init` and create the GitHub repository;
  - add the `[dev, dashboard]` extras (A12) and the pytest markers `slow` and `bench`;
  - add the make targets `test | slow | figures | dashboard | bench`;
  - set up CI on Python 3.10 and 3.12.
- **Done when:** on a fresh clone, `pip install -e .[dev,dashboard] && make test` passes, and CI is green.

#### T0.2 Decisions and architecture documents · Rithvik + all · Must

- **Depends on:** nothing.
- **Files:** `docs/decisions.md`, `docs/architecture.md`, `PLAN.md`.
- **Do:** give every V\* and A\* its own ID and a one-line reason, and draw a diagram of the tick and the agent stack.
- **Done when:** all six members have reviewed them.

#### T0.3 Configuration and geometry checks · Rithvik · Must

- **Depends on:** nothing.
- **Files:** `aerial/core/config.py`, `configs/default.json`.
- **Do:** write nested dataclasses, with JSON round-trips and dotted overrides. Validation must reject or flag:
  - N > 20;
  - F > 2;
  - n_compromised > F, unless `allow_beyond_assumption` is set;
  - an FOV outside (0°, 180°];
  - a z band that does not fit inside the arena.

  Also compute the gate clearance and **check it against r_sep, street width, layer spacing and formation spacing**,
  and **warn when R − v_close·L·dt ≤ the clearance**.
- **Done when:** every invalid case raises with a clear message, and the geometry checks are unit-tested.

#### T0.4 Random streams and pairing · Rithvik · Must

- **Depends on:** nothing.
- **Files:** `aerial/core/rng.py`.
- **Do:** implement the named streams and a `CountingGenerator`.
- **Done when:** two different fleets make identical environment draws.

#### T0.5 Core interfaces as stubs · Rithvik + Suhas + Praneeth · Must

- **Depends on:** T0.3.
- **Files:** `aerial/core/types.py`, `aerial/agents/base.py`, `aerial/comms/messages.py`, `aerial/core/result.py`,
  `aerial/analysis/online.py`, `aerial/registry.py`.
- **Do:** implement:
  - everything in §6, including `beacon()`, the extras schema, and the online-metrics accumulator used at `minimal`;
  - the `Recorder` with its levels;
  - `SimResult.save` and `load`.
- **Done when:** a fleet of `ZeroAgent`s round-trips through save and load.

#### T0.6 3D dynamics · Rithvik · Must

- **Depends on:** nothing.
- **Files:** `aerial/core/dynamics.py`.
- **Do:** port the first build's dynamics, then add vertical limits and the yaw-rate limit.
- **Done when:** the closed-form tests pass, including saturation and yaw wrap-around.

#### T0.7 Simulator loop · Rithvik · Must

- **Depends on:** T0.3–T0.6.
- **Files:** `aerial/core/simulator.py`.
- **Do:** implement the tick order in §3 (A17), with hooks for later cards.
- **Done when:** `ZeroAgent` and a toy waypoint agent both run, and a regression pin on their positions holds.

#### T0.8 Contract test harnesses · Rithvik + Suhas · Must

- **Depends on:** T0.7.
- **Files:** `tests/contracts/test_decentralised.py`, `test_pairing.py`, `test_bit_identity.py`,
  `test_import_guard.py`.
- **Do:** test that:
  - perturbing ground truth outside agent i's sensors and radio leaves i's action unchanged, for every registered
    agent;
  - an intentionally unpaired agent trips the pairing assertion;
  - the import guard holds.
- **Done when:** all harnesses run over `ZeroAgent`.

#### T0.9 Port the validated generic modules · Must

**Owners:** Rohith (STL), Suhas (Kripke), Sanjana (robustness, aggregators, attacks), Praneeth (lab, stability).

- **Depends on:** T0.1.
- **Files:** the paths in §7.
- **Do:** copy each module with its tests, tidy it, and change no behaviour.
- **Done when:** the ported tests pass with only their imports changed, and the lab reproduces the W-MSR hull property
  and the closed-form λ₂ values.

#### T0.10 Epistemic replay harness · Suhas · Must

- **Depends on:** T0.5.
- **Files:** `aerial/epistemic/replay.py`, `tests/epistemic/`.
- **Do:** build a scripted mini-world, with no physics, that emits honest claims, liar claims (with controllable
  timing), ranges, sightings and drops. Feed it to the epistemic modules, and fuzz evidence streams with hypothesis.
- **Done when:** the soundness invariants (§4) can run against T3.3–T3.9 **before the mission exists**.

### Phase 1: world, sensing, comms (weeks 1–2)

#### T1.1 Terrain generator · Prabhu · Must

- **Depends on:** T0.3, T0.4.
- **Files:** `aerial/world/terrain.py`.
- **Do:** generate:
  - blocks, streets and buildings;
  - a seeded collapse, with rubble spilling into the streets;
  - the coverable mask;
  - the **ground-truth `viewable` mask** (a cell counts as viewable if it is viewable from some pose in the band);
  - a connectivity check from the base to every viewable cell, re-drawing if it fails.
- **Done when:** generation is deterministic, takes under 0.5 s, and the checks pass.

#### T1.2 Survivors · Prabhu · Must

- **Depends on:** T1.1.
- **Files:** `aerial/world/survivors.py`.
- **Do:** place survivors at the centres of **viewable** cells; assign alive/dead; give each a vital-sign signature.
- **Done when:** over 1,000 seeds, the placement constraints and the statistics hold.

#### T1.3 Geometry kernel · Prabhu + Rithvik · Must

- **Depends on:** T1.1.
- **Files:** `aerial/world/geometry.py`.
- **Do:** implement:
  - **exact grid traversal (DDA)** of many rays at once, returning the cells passed over and the cell hit;
  - one line-of-sight function, shared by coverage, detection and sightings;
  - segment–terrain clearance;
  - distance to blocking columns, layer by layer.
- **Done when:** thin 1 m columns are never skipped (property test), and 20 drones × 400 rays take ≤ 5 ms.

#### T1.4 Camera · Prabhu · Must

- **Depends on:** T1.3.
- **Files:** `aerial/sensing/camera.py`.
- **Do:** build the frustum and produce terrain-**interval** observations (A18) and `viewed(c, pose)` (A7).
- **Done when:** FOV edges, occlusion and range limits are covered by tests; wall hits never lower lo or set hi; the
  footprint is within 5% of the geometry.

#### T1.5 Survivor detector · Prabhu · Must

- **Depends on:** T1.2, T1.4.
- **Files:** `aerial/sensing/detector.py`.
- **Do:** detect presence through `viewed(c)`, and return the **exact cell id**, a noisy position and the vital-sign
  looks. There are no false positives.
- **Done when:** a property test shows viewed ∧ survivor ⇔ detected.

#### T1.6 Drone sightings and the visible volume · Prabhu · Must

- **Depends on:** T1.4.
- **Files:** `aerial/sensing/sightings.py`.
- **Do:** detect drones within the frustum, within r_see and in line of sight. `VisibleVolume.contains(x, margin)`
  uses the same code.
- **Done when:** a property test shows contains ⇒ detected.

#### T1.7 UWB ranging · Sanjana · Must

- **Depends on:** T0.4.
- **Files:** `aerial/sensing/ranging.py`.
- **Do:** produce a range per beacon at the transmit tick, drawing a full block every tick.
- **Done when:** the error is within 4σ, and turning ranging on leaves trajectories bit-identical when no agent uses it.

#### T1.8 Network · Rithvik · Must

- **Depends on:** T0.9.
- **Files:** `aerial/comms/network.py`.
- **Do:** build the disk graph with drops, and log λ₂ and the connected components each tick.
- **Done when:** the first build's graph tests and the pairing test pass.

#### T1.9 Messaging and relaying · Rithvik · Must

- **Depends on:** T0.5, T1.8.
- **Files:** `aerial/comms/gossip.py`, `aerial/comms/messages.py`.
- **Do:**
  - send mandatory beacons every tick;
  - relay with deduplication, one hop per tick, under the priority bandwidth cap;
  - give every message an unforgeable origin;
  - add the 1 Hz relayed `POSE` and the full `RANGES` payload;
  - implement re-query (D4).
- **Done when:**
  - an event crosses a connected graph within its diameter in ticks;
  - the cap is respected;
  - forging another drone's message is impossible.

#### T1.10 Scenario generator · Prabhu · Must

- **Depends on:** T1.1, T1.2.
- **Files:** `aerial/world/scenario.py`, `configs/scenarios/*.json`.
- **Do:** implement `generate_scenario(name, seed)`, returning the world, survivors, launch poses, the **liar's
  identity and claim region**, and the viewable mask. The scenarios are `open_field`, `district`, `dense_rubble`, and
  `demo_district`, in which the claim region holds 3 or more survivors.
- **Done when:** every scenario validates and is deterministic.

### Phase 2: the honest mission end to end (weeks 2–3)

#### T2.1 Provenance store · Suhas · Must

- **Depends on:** T0.5.
- **Files:** `aerial/mapping/provenance.py`.
- **Do:** store, for each cell and claim kind, a bitmask of reporters, an own-observation flag and timestamps. Answer
  vectorised queries: anyone, at least k reporters, and "some reporter outside S" for a batch of S.
- **Done when:** merging is idempotent and order-free, and the results match brute force.

#### T2.2 Terrain map · Prabhu · Must

- **Depends on:** T1.4, T2.1.
- **Files:** `aerial/mapping/terrain_map.py`.
- **Do:** keep [lo, hi] intervals from the drone's own observations and from reports with provenance, and answer
  `free_at(z, standard)`.
- **Done when:** observed cells converge, and a cell is "free" only if its hi is known.

#### T2.3 Coverage map · Prabhu · Must

- **Depends on:** T1.4, T2.1.
- **Files:** `aerial/mapping/coverage.py`.
- **Do:** track the drone's own views, `CLEARED` and `SURVIVOR` claims with provenance, `cleared(standard)`, and
  `abandoned`.
- **Done when:** the drone's own views match the ground truth.

#### T2.4 3D path planner · Praneeth · Must

- **Depends on:** T2.2.
- **Files:** `aerial/planning/astar.py`.
- **Do:** plan with A\* over (x, y, layer), treating unknown cells as blocked, with a climb penalty and shortcutting.
  Replan **only when the path is invalidated**, staggered across drones. Use scipy csgraph where it helps.
- **Done when:**
  - paths never cross occupied or unknown cells;
  - a typical plan takes **≤ 5 ms**, and the worst case on a 100 × 100 × 3 grid takes ≤ 30 ms.

#### T2.5 Dynamic partition · Praneeth · Must

- **Depends on:** T2.3, T2.4.
- **Files:** `aerial/planning/partition.py`.
- **Do:**
  - compute geodesic Voronoi labels with **multi-source Dijkstra (scipy csgraph, `min_only`)** over the coverage grid;
  - treat **unknown cells as passable** when measuring distance, and known-occupied cells as blocked;
  - take sources from the drone's own position, fresh beacons and relayed `POSE` events;
  - exclude drones the fleet's policy treats as faulty;
  - add **target hysteresis**;
  - provide the `static_strips` alternative.
- **Done when:**
  - identical information gives identical labels;
  - excluding a drone redistributes its region;
  - one update takes ≤ 2 ms.

#### T2.6 Exploration and target choice · Praneeth + Prabhu · Must

- **Depends on:** T2.5.
- **Files:** `aerial/planning/explorer.py`.
- **Do:**
  - pick the best cell that is uncovered under the belief standard in the drone's own region;
  - when the region is empty, steal work;
  - look before moving;
  - abandon a cell after k failed views;
  - Should: the heading-to-layer rule (A5).
- **Done when:** target selection, work stealing and abandonment pass **unit tests on synthetic maps**. The fleet-level
  coverage criterion belongs to T2.14.

#### T2.7 Safety layer · Praneeth · Must

- **Depends on:** T1.3.
- **Files:** `aerial/safety.py`.
- **Do:** define Safe as separation ≥ d_safe for all pairs ∧ clearance ≥ d_obs ∧ inside the arena ∧ inside the z band.
  Implement the never-gated terms (separation from own velocity, terrain repulsion, geofence, brake) and `prioritise`.
- **Done when:** the first build's safety tests pass after porting.

#### T2.8 Controller and mission state machine · Praneeth · Must

- **Depends on:** T2.4, T2.6, T2.7.
- **Files:** `aerial/agents/controller.py`, `aerial/agents/mission_fsm.py`.
- **Do:** write a PD follower with a speed governor, split into advance and safety, and the state machine from §5
  **with its back-edges and the CONFIRM cap**.
- **Done when:** a drone flies three legs with replanning, and every transition has a unit test.

#### T2.9 Transit in formation · Praneeth + Sanjana · Must

- **Depends on:** T0.9, T2.8.
- **Files:** `aerial/agents/transit.py`.
- **Do:** use line-abreast offsets and consensus tracking to a common waypoint, with W-MSR per coordinate, 2F + 1
  agents pinned and DP-MSR damping. Log `formation_err` and a sampled `robust_ok` (whether the graph is (2F+1)-robust).
- **Done when:** the error is < 0.5 m with no attack, and W-MSR stays bounded against a 10 m liar while the mean is
  dragged away.

#### T2.10 Survivor tracks and resilient localisation · Sanjana + Suhas · Must

- **Depends on:** T1.5, T1.9, T0.9.
- **Files:** `aerial/agents/tracks.py`.
- **Do:** associate reports into tracks, with the cell as the key; run consensus on each position estimate using the
  mean or W-MSR; add observers through confirmation visits. The liar's estimate-offset knob is off by default and used
  only in E6.
- **Done when:** with 3 or more observers and an 8 m liar, the W-MSR error is ≤ 1.5 × the honest noise.

#### T2.11 Alive/dead belief · Prabhu (Suhas reviews) · Must

- **Depends on:** T1.5.
- **Files:** `aerial/epistemic/belief.py`.
- **Do:** compute log-odds from heat and motion; fuse under the fleet's standard; run the SPRT; cap at T_conf, then
  "unsure", which prompts a second look.
- **Done when:** accuracy is ≥ 95% within k looks on average, and the error rates match SPRT theory.

#### T2.12 Return and regroup · Praneeth · Must (a V11 promise)

- **Depends on:** T2.8.
- **Files:** `aerial/agents/regroup.py`.
- **Do:** fly back with A\*, then settle into landing slots with formation or rendezvous consensus under W-MSR. Log
  `formation_err`.
- **Done when:** every honest drone ends in its own slot with 0 separation violations.

#### T2.13 Mission metrics · Prabhu · Must

- **Depends on:** T1.2, T0.5.
- **Files:** `aerial/analysis/mission.py`.
- **Do:** compute the §9 metrics, both offline and online. Mission metrics count honest drones only; safety counts all
  drones; coverage uses the ground-truth viewable mask.
- **Done when:** the fixtures give the expected numbers.

#### T2.14 Honest baseline and benchmark (Phase 2 exit) · Prabhu + Rithvik · Must

- **Depends on:** every Must card in Phase 2.
- **Files:** `scripts/validate_baseline.py`, `tests/test_baseline.py` (slow), `tests/bench/`.
- **Do:** run 3 scenarios × 20 seeds with no liar.
- **Done when:**
  - ≥ 95% of viewable coverable cells are covered;
  - ≥ 95% of survivors are found;
  - there are 0 collisions and ρ_safety > 0;
  - each run takes ≤ 20 s at N = 10 (benchmark).

### Phase 3: the liar and the epistemic layer (weeks 2–3; the epistemic core starts in week 1)

#### T3.1 Liar agent · Sanjana · Must

- **Depends on:** T2.8, T1.9, T1.7.
- **Files:** `aerial/resilience/liar.py`.
- **Do:** implement the modes from §3: the true pose during transit, `beacon()` lies, `CLEARED` claims for the claim
  region, and survivor suppression.
- **Done when:** each mode emits the expected broadcasts; `range_aware` stays consistent with 1–2 receivers and fails
  with 3 or more; the decentralisation and import-guard tests pass.

#### T3.2 Legacy position attacks · Sanjana · Should

- **Depends on:** T0.9.
- **Files:** `aerial/lab/attacks.py`.
- **Do:** port the first build's attacks for the lab and for E7.
- **Done when:** the ported tests pass.

#### T3.3 Fault hypotheses Hᵢ · Suhas · Must

- **Depends on:** T0.5.
- **Files:** `aerial/epistemic/hypotheses.py`.
- **Do:** implement:
  - the masks;
  - the constraints `Unit` and `Clause`;
  - the queries `known_faulty`, `known_honest`, `possibly_faulty`, `budget_left` and `n_hypotheses`;
  - `assumption_broken`, with the **empty-set fallback** from §4.
- **Done when:** hand-worked examples at F = 1 and F = 2 pass, and replay fuzzing shows S\* is never eliminated.

#### T3.4 Physics proofs R1–R3 · Suhas + Sanjana · Must

- **Depends on:** T3.3, T1.7.
- **Files:** `aerial/epistemic/evidence.py`.
- **Do:** implement the rules on fresh beacons only (A17), with the `staleness()` helper and separate horizontal and
  vertical limits in R2. Put each rule's soundness argument in its docstring.
- **Done when:** the rules fire 0 times in 200 honest runs, including drones climbing and drones entering radio range,
  and each violating liar mode triggers them.

#### T3.5 Camera proofs R4 and R4b · Suhas · Must

- **Depends on:** T1.6, T3.3.
- **Files:** `aerial/epistemic/evidence.py`.
- **Do:**
  - **R4:** the claim's honest-ball, shrunk by the frame age, lies inside the visible volume, and nothing is sighted
    there.
  - **R4b:** an OR-clause over every drone that cannot explain the sighting, silent drones included.
- **Done when:** R4 never fires on honest drones, including when beacons are dropped, and it catches a fake pose below
  the observer.

#### T3.6 Shared ranges · Sanjana + Suhas · Must

- **Depends on:** T1.9, T3.3.
- **Files:** `aerial/epistemic/multilateration.py`.
- **Do:** add `Clause({j, r})` for each contradicting `RANGES` report. For the gate's faulty-status set, use the
  shared-shell intersection with slack (the first build's rule). A least-squares fix is for display only.
- **Done when:** with 2 contradicting reporters at F = 1, j becomes known faulty, and range noise never produces an
  accusation.

#### T3.7 Audits R5 and R7 · Suhas + Praneeth · Must

- **Depends on:** T2.6, T2.3, T3.3.
- **Files:** `aerial/epistemic/audit.py`, plus a hook in the explorer.
- **Do:** implement §4's audits: auditor ≠ j, top-ups, approach through the frontier, resampling, and reporting of the
  achieved fraction and the residual risk. Implement R5 from the drone's own exact-cell detections. Implement R7, with
  unknown terrain counted as see-through.
- **Done when:**
  - the catch rate for one hidden survivor matches the achieved fraction within statistical tolerance;
  - there are 0 false accusations;
  - the audit budget always finishes in the liar's unmapped block.

#### T3.8 Accusations and corroboration · Suhas · Must

- **Depends on:** T1.9, T3.3, T3.4.
- **Files:** `aerial/epistemic/accusations.py`.
- **Do:** turn each `ACCUSE` into `Clause({a, j})`, with the evidence references attached. A `Unit` comes only from
  j's own signed messages or the receiver's own knowledge. Implement corroboration (§4): re-check the same beacon
  `seq` against the drone's own range history, and raise audit priority on the accused. Broadcast `KNOWN_FAULTY`.
- **Done when:**
  - a lone liar's framing never makes an honest drone known faulty;
  - a true catch reaches every honest drone of a connected graph through corroboration.

#### T3.9 Knowledge over provenance · Suhas · Must

- **Depends on:** T2.1, T3.3.
- **Files:** `aerial/epistemic/knowledge.py`.
- **Do:** compute Kᵢ for cleared, free and survivor, incrementally, together with the Bᵢ standards. Plug them into
  T2.2, T2.3 and T2.5, with the empty-set fallback.
- **Done when:** results match brute force, and exhausting the budget flips knowledge in one update.

#### T3.10 Soundness invariants · Suhas · Must

- **Depends on:** T0.10 and T3.3–T3.9 for the early version; T3.15 for the full version.
- **Files:** `tests/epistemic/test_soundness.py`.
- **Do:** run the §4 invariants on the replay harness straight away (fast, in CI), then on 50 full attacked runs over
  every liar mode (slow).
- **Done when:** green, and rerun after every change to `epistemic/`.

#### T3.11 Kripke cross-check · Suhas · Must

- **Depends on:** T0.9, T3.3, T3.9.
- **Files:** `tests/epistemic/test_kripke_crosscheck.py`.
- **Do:** for N ≤ 5, build an explicit model and compare Kᵢ, Kᵢ Kⱼ and Eᴳ with the fast code.
- **Done when:** they agree on 300 random instances.

#### T3.12 The knowledge gate · Suhas · Must

- **Depends on:** T3.9, T2.7, T3.6, T3.7, T2.11.
- **Files:** `aerial/epistemic/gate.py`.
- **Do:** implement gate parts (a)–(d) from §4:
  - the honest ball built from the reported velocity;
  - the union for possibly-faulty drones;
  - the silence rule;
  - shrinking of the parts seen empty;
  - right of way (A20);
  - re-query and detour.
- **Done when:**
  - with F = 0, the gate holds only when something is truly close;
  - two honest drones crossing in an 8 m street on adjacent layers never deadlock;
  - the residual risk matches Monte Carlo;
  - the old back-away regression case passes.

#### T3.13 Exclusion and re-sweep · Suhas + Praneeth · Must

- **Depends on:** T3.8, T3.9, T2.5, T2.8.
- **Files:** `aerial/agents/gated.py`.
- **Do:** implement the four steps in §4, together with the state machine's back-edges.
- **Done when:** in the demo, every cell only the liar vouched for is viewed again before the gated fleet finishes,
  including after a late catch.

#### T3.14 Nested knowledge, group knowledge and abstraction size · Suhas · Must

- **Depends on:** T3.8, T3.9.
- **Files:** `aerial/epistemic/nested.py`.
- **Do:** compute Kᵢ Kⱼ, the time of Eᴳ and the depth of Eᵏ, the data for the knowledge matrix, and the exact
  log-space world count from §4.
- **Done when:** sound against the exact value on small instances, and the world count matches enumeration for N ≤ 4.

#### T3.15 Fleets · Rithvik + Suhas · Must

- **Depends on:** T3.1–T3.13.
- **Files:** `aerial/agents/fleets.py`.
- **Do:** define the fleets:

  | Fleet | Behaviour |
  |---|---|
  | unfiltered | trusts everything, uses the mean |
  | naive | trusts every claim, uses W-MSR |
  | gated | W-MSR, proofs, corroboration, audits, K standards, the gate, re-sweep |
  | oracle | knows S\* at t = 0 |
  | ablations | detect-only; no camera; no shared ranges; q = 0 |

- **Done when:** one config switch selects each fleet, and a test shows the fleets share the same sensors and the same
  terrain caution.

#### T3.16 Demo acceptance (Phase 3 exit) · Suhas + Prabhu · Must

- **Depends on:** every Must card in Phase 3.
- **Files:** `tests/test_demo_acceptance.py` (slow), `scripts/validate_demo.py`.
- **Done when:** the demo criteria in §10 hold over 30 seeds.

### Phase 4: specifications, evaluation, dashboard (weeks 3–4)

#### T4.1 Mission STL specs · Rohith · Must

- **Depends on:** T0.9, T2.7, T2.13.
- **Files:** `aerial/spec/mission_specs.py`.
- **Do:** define the specs, each with ρ and ρ(t):

  | Spec | Formula |
  |---|---|
  | φ_safe | G[0,T](Safe) |
  | φ_find(s) | F[0,T_find](s detected) |
  | φ_alive | F[0,T_conf](s classified), for each live s |
  | φ_know | F[0,T_k](∀ honest i: Kᵢ(L faulty)), from the `known_faulty` extra |
  | φ_cover | F[0,T](known-cleared ≥ 0.95) |

- **Done when:** the results match hand-computed traces.

#### T4.2 Runner, pairing and sweeps · Rithvik · Must

- **Depends on:** T3.15.
- **Files:** `aerial/experiments/runner.py`.
- **Do:** implement `run_one`; `run_pair`, which asserts pairing; and `sweep`, with at most 8 workers by default.
  Write `results/<exp>/<stamp>/{config,grid,rows,env}` and support resuming.
- **Done when:** an unpaired agent trips the assertion, and an interrupted sweep resumes.

#### T4.3 Performance pass · Rithvik · Must

- **Depends on:** T3.15.
- **Files:** `tests/bench/`.
- **Do:** profile the code and enforce budgets for each component:

  | Component | Budget |
  |---|---|
  | camera | 20 drones × 400 rays in ≤ 5 ms |
  | A\* | ≤ 5 ms typical |
  | partition | ≤ 2 ms |
  | knowledge update (cells × hypotheses) | ≤ 1 ms |
  | 150 s mission at N = 10 | ≤ 20 s |
  | 150 s mission at N = 20 | ≤ 60 s |

  World counting stays off in sweeps. Use numba (A21) only with the team's agreement.
- **Done when:** the bench tests enforce every budget.

#### T4.4 Experiment scripts · owners in §9 · Must/Should

- **Depends on:** T4.2.
- **Files:** `scripts/exp_*.py`.
- **Do:** give every script `--quick` and `--workers`, and store the rows it produces in `docs/report/data/`.
- **Done when:** every `--quick` run finishes in under 5 minutes.

#### T4.5 Figures · Rohith + owners · Must

- **Depends on:** T4.4.
- **Files:** `scripts/fig_*.py`.
- **Do:** one script per figure, drawing only from the stored rows.
- **Done when:** `make figures QUICK=1` finishes in under 5 minutes.

#### T4.6 Plotly 3D scene · Rohith · Must

- **Depends on:** T0.5, T1.1.
- **Files:** `aerial/viz/scene3d.py`.
- **Do:** draw:
  - merged building meshes;
  - drones with FOV frustums and trails;
  - ground coverage, known against believed;
  - survivor states;
  - a halo on the liar once it is caught;
  - frames at 2 Hz;
  - side-by-side panels for a paired run.

  Export static images through kaleido.
- **Done when:** a 150 s run renders in under 3 s.

#### T4.7 Streamlit dashboard · Rohith · Must

- **Depends on:** T4.6, T4.2.
- **Files:** `dashboard/app.py`, `aerial/viz/dashboard_data.py`.
- **Do:** build five tabs:

  | Tab | Content |
  |---|---|
  | Mission | side-by-side 3D; sliders for N, F, p_drop, R, FOV, liar mode, q and the fleet pair |
  | Knowledge | the Kᵢ(j faulty) matrix and the **Kᵢ(Safe) gate-open matrix** over time; \|Hᵢ\|; known against believed coverage; the fraction of drones that know, against the SI and threshold fits |
  | Coordination | **formation and consensus error over time**, for transit, localisation and regroup; W-MSR rejection heatmaps |
  | Safety | ρ(t) for both fleets against the zero line |
  | Batch | the trade-off curve |

- **Done when:** the `AppTest` smoke test passes, and the demo loads in under 5 s from a cached run.

### Phase 5: deliverables (week 4)

| Card | Owner · Priority | What |
|---|---|---|
| **T5.1** | Rithvik · Must | End-to-end acceptance on a fresh clone: `make test`, the slow suite, `make figures QUICK=1`, and the dashboard smoke test. |
| **T5.2** | Rithvik · Must | README with the architecture diagram and the commands to run everything. |
| **T5.3** | all; Rohith assembles · Must | LaTeX report with one chapter per owner, the individual-contribution page, and an AI-usage statement (both required by the course). |
| **T5.4** | Rohith + Suhas · Must | Demo video, and `docs/story.md` with a frame-by-frame narrative. |
| **T5.5** | all · Must | 15-minute presentation. |

### Timeline and critical path

| Week | Work |
|---|---|
| 1 | Phase 0, the Phase 1 streams, and the epistemic core on the replay harness (T0.10, T2.1, T3.3, T3.9, T3.11) |
| 2 | Phase 2, the liar (T3.1), and the evidence rules (T3.4–T3.8) |
| 3 | Phase 3 integration up to demo acceptance; in parallel, the Phase 4 runner, scene and specs |
| 4 | Sweeps, figures, dashboard, and Phase 5. The second half of week 4 is buffer. |

**Critical path:** T0.5 → T0.7 → T1.1/T1.3/T1.4 → T2.2/T2.3 → T2.4–T2.6 → T2.8 → T2.14 → T3.1 → T3.4–T3.8 →
T3.12/T3.13 → T3.15 → T3.16 → E2 → figures and report.

**Start the E2 sweep by the start of week 4 at the latest.**

---

## 9. Experiments, metrics and figures

**Metrics recorded for every run:**

- **survivors:** found (all, and alive only); time to find each; missed, split into "in the liar's claimed area" and
  "elsewhere"; alive/dead accuracy;
- **time:** completion time;
- **safety:** collisions, minimum separation, minimum clearance;
- **coverage:** true coverage of viewable cells; known-cleared and believed-cleared fractions; achieved audit fractions
  and residual risk;
- **liar:** time to the first proof; time until every honest drone knows; false accusations (must be 0);
- **cost:** hesitation time, audit cost, re-sweep cost, messages sent.

| Study | Owner | Design | Question |
|---|---|---|---|
| E1 Demo | Suhas, Prabhu | `demo_district`, 4 fleets × 30 seeds | Does the gated fleet find the survivors that the naive fleet misses, and at what cost in time? |
| E2 Trade-off | Rohith | q ∈ {0, .05, .1, .2, .3} × liar ∈ {range_aware, radio_shadow, true_pose + fake coverage} × p_drop ∈ {0, .1, .3} × R ∈ {30, 35, 50} × F ∈ {1, 2} (fractional design) × 30 seeds; about 5k runs. Every R and p_drop combination keeps R − v_close·L·dt above the clearance. | Caution against completion time against survivors missed. The naive and gated families should form separate groups. |
| E3 Knowledge spread | Suhas | F = 1. The fraction of honest drones that know, from the first proof onward, as R and p_drop vary. | Does it fit SI, or threshold-2 contagion (the Infection lecture)? |
| E4 No liar | Prabhu | n_compromised = 0, F ∈ {1, 2}, sweep over q | What does caution cost when nobody lies? Known against believed coverage, and residual risk. |
| E5 Detection ablation | Sanjana, Suhas | Each rule alone and in combination, against all liar modes including `fake_sweep_naive` | Which technique catches which lie, and how fast? |
| E6 Localisation | Sanjana | mean vs W-MSR × offset size × number of observers | Are the survivor coordinates sent to rescuers protected? |
| E7 Transit | Praneeth, Sanjana | The ported attacks, with and without W-MSR, plus the `robust_ok` log | What W-MSR is worth inside the mission, and whether its graph condition holds. |
| E8 FOV and altitude | Prabhu | 90° vs 150° × cruise layer | Coverage rate against detection and safety. |
| E9 Topology | Rithvik | λ₂(t), connected components and time disconnected, against R | How connectivity drives the spread of knowledge. |
| E10–E12 Ported | Praneeth (λ₂), Sanjana (ER, degradation) | The first build's designs, run in the lab | Reproduce the first build's headline numbers. |

**Report figures:**

- 3D snapshots of the demo, side by side;
- end-of-run maps of known against believed coverage;
- the knowledge matrix over time;
- spread curves with the SI and threshold fits;
- the trade-off curve;
- ρ(t);
- detection-time CDFs;
- localisation error;
- formation error in transit and regroup;
- the λ₂, ER and degradation plots;
- a table of abstraction sizes.

---

## 10. Success criteria (v2)

| Criterion | Target |
|---|---|
| Honest baseline (no liar, 3 scenarios × 20 seeds) | ≥ 95% of viewable coverable cells covered and ≥ 95% of survivors found; 0 collisions; ρ_safety > 0 in every run |
| Naive fleet on the demo | misses ≥ 1 survivor in ≥ 70% of 30 seeds |
| Gated fleet on the demo | catches the liar in ≥ 90% of seeds; misses 0 survivors in every seed where it caught the liar; 0 collisions |
| Termination | every gated run ends by T_max, through completion or the V12 end rule, with no deadlocks |
| Soundness (every attacked run, every study) | 0 false accusations; S\* ∈ Hᵢ always; every Kᵢ fact true in ground truth |
| Cost of caution with no liar, at q = 0.1 | measured and reported; soft target ≤ 25% extra completion time |
| Trade-off curve | naive and gated families form visibly separate groups |
| W-MSR in the mission | honest error stays bounded in transit and localisation, while the mean is dragged away |
| Ported studies | reproduce the first build's headline numbers within statistical tolerance |
| Performance | N = 10, 150 s ≤ 20 s; N = 20 ≤ 60 s |

## 11. Testing and engineering rules (changes from `docs/history/CONTEXT_first_build.md` §12 and §14)

Carry over everything in `docs/history/CONTEXT_first_build.md` §12 and §14. On top of that, add:

- the soundness invariants, which run on the replay harness from week 1;
- the empty-Hᵢ fallback test;
- the no-false-accusation property across all studies;
- timing tests for drones climbing and drones entering radio range;
- the wall-hit and thin-column tests for terrain intervals;
- the Kripke cross-check;
- a statistical audit test;
- the import guard;
- the bench budgets.

The default suite should take about 3 minutes. The slow suite holds the baseline, demo acceptance, the 50-run
soundness test and `AppTest`.

## 12. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Under the dynamic partition, the liar uses a fake position to claim territory and stall coverage. | R4, R6, R7 and audits; corroboration; the V12 end rule; `static_strips` as a fallback (A16). |
| R2 | The per-agent Python code is too slow. | Vectorised sensing, scipy csgraph, replanning only when a path is invalidated, per-component budgets (T4.3); numba (A21) as a last resort. |
| R3 | A soundness bug in the epistemic layer. | The replay harness from week 1, invariant tests, the Kripke cross-check, and a docstring proof for every rule. |
| R4 | The camera sees the liar too, so naive collisions are rare. | The headline contrast is survivors missed, so collisions are not promised. Use the liar in blind spots (sideways and above the observer) in the demo. |
| R5 | Honest drones hold each other in narrow streets. | The honest ball from reported velocity, right of way (A20), and the T0.3 geometry checks. |
| R6 | Knowledge spreads slowly at F = 2. | Report it as a finding (§4). E3 runs at F = 1. |
| R7 | Lab compute rules. | At most 8 workers; check GPU needs before E2. |
| R8 | Scope creep. | Cut Should and Could cards first; the Must cards cover every success criterion. |

## 13. Verification (end to end, once the cards are done)

1. `pip install -e .[dev,dashboard]` on a fresh clone, then `make test`: green in about 3 minutes.
2. `pytest -m slow`: the baseline, demo acceptance, 50-run soundness and `AppTest` all pass.
3. `python scripts/validate_demo.py` prints the §10 demo rows over 30 seeds.
4. `make figures QUICK=1` finishes in under 5 minutes; the full figures are redrawn from `docs/report/data/`.
5. `make dashboard`: pick `demo_district`, naive against gated. Watch the gated fleet catch the liar and re-sweep the
   NE block while the knowledge matrix fills in.

## 14. Next steps once this plan is approved

1. Write this plan into the repository as `new_project/PLAN.md`, with a checkbox on each card, and write
   `docs/decisions.md`.
2. `git init`, and create the GitHub repository (T0.1).
3. Hand out the Phase 0 cards in parallel, using the prompt template.
4. Optionally, publish the plan as a page the team can share.

## What the review changed

An independent red-team pass found 14 issues, all fixed above.

**Blockers:**

- Survivors detected by noisy position could be filed under the wrong cell and wrongly accuse an honest drone, so
  detection now uses the exact cell (A7).
- Timing slack let R1–R3 accuse honest drones that were climbing or entering radio range, so beacons are now same-tick
  and every rule uses a 3D `staleness` helper (A17).
- A camera ray hitting a wall was stored as the building's height, which made the air above tall buildings look free.
  Terrain is now stored as height intervals with exact ray traversal (A18).

**Majors:**

- the gate's possible-position sets: the union over both statuses, the silence rule, and shrinking what was seen empty;
- R4b clauses, and R4 being limited to claims at or below the observer;
- audit loopholes and unreachable audit cells;
- missing back-edges in the state machine, and cells that cannot be viewed;
- honest drones deadlocking in narrow streets (fixed by A20 and the T0.3 checks);
- partition liveness (A19, unknown cells passable for distances, hysteresis);
- A\* and partition costs (component budgets);
- a demo story in which the liar would have been caught at t ≈ 1 s;
- card-order errors (the replay harness T0.10, the extras schema, the `beacon()` hook).

**Minors:**

- Unit evidence must be checkable from j's own messages or the receiver's own knowledge;
- empty Hᵢ;
- knowledge at F = 2 spreading only through 3 accusers;
- missing proposal items (nested knowledge and regroup promoted to Must, the exact abstraction size, the formation-error
  and Kᵢ(Safe) plots, a robustness check on the mission graph).
