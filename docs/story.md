# AERIAL v2: the story of one mission

This document follows one simulated mission from take-off to landing. At each step it explains what the drones do,
what each drone knows, which multi-agent algorithm is running, and how the **naive** and **gated** fleets differ.
It describes the code as it is now built. Places where the code differs from `PLAN.md` are marked **(not built
yet)** or **(simplified)**.

The numbers come from the demo run that `scripts/export_viewer.py` renders into the 3D viewer,
`docs/media/viewer.html`:

| Fleet | Survivors found | Missed in the liar's area | Collisions | Liar known to all honest drones | All landed |
|---|---|---|---|---|---|
| naive | 10 / 16 | 6 | 8 (all between honest drones) | never | 189 s |
| gated | 16 / 16 | 0 | 0 | 13.3 s | 192 s |

This is one seed, chosen to show the effect. Its liar is set to be blatant: it claims its whole area at once
(`attack.claim_all`), starts lying without waiting for a quiet spot (`attack.seek_quiet=false`), and physically
sits in drones' paths (`attack.motion=ambush`).

---

## Contents

- [Part 0. The setup](#part-0-the-setup)
- [Part 1. Take-off and transit](#part-1-take-off-and-transit-0-12-s)
- [Part 2. Exploring unmapped terrain](#part-2-exploring-unmapped-terrain)
- [Part 3. Finding survivors](#part-3-finding-survivors)
- [Part 4. The liar starts lying](#part-4-the-liar-starts-lying-from-about-12-s)
- [Part 5. Catching the liar](#part-5-catching-the-liar-gated-fleet-only-about-13-s)
- [Part 6. After the catch: re-sweep](#part-6-after-the-catch-exclusion-and-re-sweep)
- [Part 7. Collisions and the knowledge gate](#part-7-collisions-and-the-knowledge-gate)
- [Part 8. Audits](#part-8-audits-what-the-gated-fleet-does-when-it-cannot-catch-anyone)
- [Part 9. Deciding to stop and coming home](#part-9-deciding-to-stop-and-coming-home)
- [Part 10. Grading the run](#part-10-grading-the-run)
- [Summary tables](#summary-tables)
- [Gaps between this build and the plan](#gaps-between-this-build-and-the-plan)

---

## Part 0. The setup

### The world: ground truth, which no drone can read

`aerial/world/terrain.py` and `aerial/world/scenario.py` build the world.

- **Arena.** 120 × 120 m. The **search zone** is the 100 × 100 m square from (10, 20) to (110, 120). Below it, the
  strip y < 20 is the **launch base**.
- **City.** City blocks with streets 8–12 m wide between them. Each block holds one or two buildings, 12–35 m tall.
- **Collapse.** Each building collapses with probability 0.4.
  - A collapsed building becomes a mound of rubble 1–5 m high.
  - Debris spills into a street as an ellipse in a random direction.
  - About a third of collapses also leave a tall pile of 13–18 m.
- **Representation.** The world is a **2.5D height map** H(x, y) on a 1 m grid. A point is solid if its height is
  below H. This model cannot represent overhangs.
- **Coverage grid.** A 2 m grid over the zone, 50 × 50 = 2,500 cells. A survivor sits at the centre of a cell.
  - The simulator also computes which cells can be seen from at least one legal pose. This is the *viewable* mask,
    and it is used only for grading.
  - Cells under a standing roof are not *coverable*.
- **Survivors.** 16 per run, placed only in viewable, coverable cells.
  - Each is alive with probability 0.7. A live survivor is conscious with probability 0.6.
  - Each has a vital signature: body heat about N(34, 1) °C if alive, 25–30 °C if dead. The chance that one camera
    look sees movement is 0.6 (conscious), 0.25 (unconscious) or 0.02 (dead, for example cloth in the wind).
- **The demo scenario.** In `demo_district` the generator also picks a **claim region**: a 50 × 50 m corner of the
  zone that the liar will claim it swept. It places at least 3 survivors deep inside that corner.

### The drones

There are 10 drones (`N = 10`). One of them is a **liar**, chosen at random; in the demo it is drone 0.

**Dynamics** (`aerial/core/dynamics.py`)
- A 3D double integrator: the drone commands an acceleration u, with |u| ≤ 4 m/s².
- Horizontal speed ≤ 5 m/s, vertical speed ≤ 2 m/s.
- Heading (yaw) turns at most 90°/s.

**Altitude rules ("3D aerial traffic management")**
- Every drone stays between 4 and 15 m.
- Drone i cruises on layer `[6, 9, 12][i % 3]` m, so neighbours with different ids are separated vertically.
- Rubble lower than the layer can be flown over. Standing buildings and tall piles must be flown round.
- **(simplified)** The plan chose the layer from the drone's heading; the code assigns it by id.

**Sensors** (`aerial/sensing/camera.py`)

A single gimbal camera faces the drone's heading.

| Property | Value |
|---|---|
| Horizontal field of view | 90° (150° can be set in the config) |
| Vertical field of view | 60°, tilted 30° down, so it sees from the horizon to 60° below it |
| Frame rate | 5 frames per second |

From each frame the drone gets:
- **Depth rays.** 32 × 12 rays, marched over the height map out to 25 m.
- **Viewed cells.** A coverage cell counts as *viewed* if its centre point is inside the frustum, within 15 m, in
  line of sight, and seen at least 20° below the horizon.
- **Survivor looks.** Each survivor in a viewed cell gives a noisy position, a heat reading and a motion yes/no.
- **Drone sightings.** Any drone inside the frustum, within 20 m and in line of sight is detected, but its identity
  is not known.

**Radio**
- Every received packet carries a **UWB time-of-flight range**: the true distance plus noise clipped at ±0.2 m.
  Nobody, the liar included, can fake this range, because it comes from physics, not from the message.

### Communication (`aerial/comms/network.py`)

- **Radio range.** R = 35 m. Drone i hears drone j only if they are within 35 m of each other.
- **Packet loss.** Each packet is also dropped with probability p = 0.1. Losses are drawn as a fresh 10 × 10 random
  block every tick, so the graph changes over time and can split.
- **Beacons.** Every tick, each drone broadcasts a **beacon** with its claimed position and velocity. A liar is free
  to lie in its beacon.
- **Events and relaying.** Everything else is an **event** that drones **relay** to one another:
  - each drone forwards each event once per hop, one hop per tick;
  - at most 64 events per message, highest priority first: accusations, then survivor reports, then cleared cells,
    and so on;
  - the event's **origin cannot be forged**, so a liar can lie about itself and refuse to relay, but cannot put
    words in another drone's mouth.
- **Re-query.** A drone that is holding can ask its neighbours to resend. The resent packets get their own,
  independent drop draws.

### One tick (every 0.1 s, `aerial/core/simulator.py`)

1. Every drone states its **beacon** claim.
2. **Sense.** Camera frames are produced (every second tick), and UWB ranges are measured.
3. **Communicate.** Beacons and relayed events are delivered over the current graph.
4. **Decide.** Each drone computes its acceleration, its yaw rate, and the events it wants to send. It uses **only
   its own observation**. A test checks this, and agent code is not allowed to import the ground-truth world.
5. **Physics.** All drones move.
6. **Record.** The step is logged.

### The four fleets

All four fleets use the same drone code, `aerial/agents/mission.py`. They differ **only in what they trust**.

| | unfiltered | **naive** | **gated** | oracle |
|---|---|---|---|---|
| Believes teammates' claims | all of them, always | all of them | all, except drones it has **proven** to be liars | all, except the true liar |
| Consensus rule | plain mean | W-MSR | W-MSR | W-MSR |
| Runs lie detection | no | no | **yes** | not needed: it is told the true liar at t = 0 |
| Audits teammates' claims | no | no | **yes** | no |
| Knowledge gate on motion | no | no | **yes** | yes |

**What "naive" means.** The naive fleet is the resilient fleet from the proposal. It filters numeric values with
W-MSR, but it takes every report at face value.

**What "gated" means.** The gated fleet acts on what it can **prove**. The oracle is a reference point: it shows the
best possible outcome, and measures how much the gated fleet pays for not knowing who the liar is.

### The assumptions

1. **At most F liars.** There are at most F = 1 liars. The liar is an ordinary drone: it has the same sensors, the
   same messages and a sector like everyone else, and **no access to ground truth**.
2. **What a liar can do.** It can lie about its own **position** (in beacons) and its own **coverage** ("I swept
   these cells and found nobody"). It can hide survivors and refuse to relay messages. It cannot forge another
   drone's message, cannot fake a UWB range, and cannot send different lies to different drones (messages are
   broadcast only).
3. **Survivors.** Survivors never move. If a survivor's cell is viewed, the survivor **is** detected; there are no
   false positives. Only the alive/dead judgement is noisy.
4. **Drone sightings.** If another drone is inside my camera frustum, within 20 m and in line of sight, I **see** it.
5. **The world.** The terrain does not change during a run. There is no wind and no aerodynamics, and drones are
   points.
6. **Prior knowledge.** Before take-off a drone knows only the arena box, the altitude band, the launch base, the
   zone it must search, its own landing slot and its own entry slot. **There is no map.**
7. **The one probabilistic assumption.** If a drone has not been heard for L = 9 ticks (at p = 0.1, L =
   ⌈ln 10⁻⁹ / ln p⌉), it is assumed to be out of radio range at some point in that window. The chance that this is
   wrong is below 10⁻⁹.

### What every drone carries in its head

These are the fields of `MissionAgent`.

| Memory | What it holds |
|---|---|
| **Terrain map** (`mapping/maps.py`, `TerrainMap`) | For every 1 m cell, an **interval [lo, hi]** for the unknown height: hi starts at +∞ and lo at 0. For each cruise layer, which teammates reported the cell free (a bitmask of drone ids). Which teammates reported a wall (H ≥ 6 m). |
| **Coverage map** (`CoverageMap`) | For each of the 2,500 cells: did *I* view it, which teammates claimed it swept and empty (bitmask), where I saw a survivor, which teammates reported a survivor there (bitmask), cells known to lie under a roof, cells given up as unviewable, and how many viewing attempts have failed. |
| **Fault hypotheses** (`epistemic/hypotheses.py`) | Every set of possible liars still consistent with what I have seen. With N = 10 and F = 1 there are 10: "no liar", "drone 1 lies", …, "drone 9 lies", where drone 0 is me. |
| **Survivor tracks** | For each survivor cell: my alive/dead belief (log-odds), my own position estimates, and my teammates' estimates. |
| **Pose book** | The last known position of every teammate, from beacons and from relayed POSE events. |
| **Claim history** | Each teammate's recent beacon claims, numbered by sequence, used for consistency checks. |
| **Audit bookkeeping** | The cells I have sampled to re-check, and the cells that anyone has audited. |
| **Proof log** | The proofs I have produced: (time, me, accused, rule). |
| **Mode** | TRANSIT, EXPLORE, CONFIRM, AUDIT, RETURN or LANDED. |

**Every claim records who made it**, as a bitmask of reporters. That one design choice is what lets a drone later
ask either question: "does *anyone* say so?" (belief) or "does someone I can *prove* honest say so?" (knowledge).

---

## Part 1. Take-off and transit (0–12 s)

**Starting position.** The ten drones sit on the base at y = 10, 8 m apart, 4 m up, all facing north into the zone.

**The goal.** Each drone has an **entry slot**. The slots are spread evenly across the zone's southern edge
(y ≈ 22), at x = 15, 25, …, 105, each at that drone's own layer. Spreading the drones out like this gives them a
sensible starting split of the area.

**Multi-agent algorithm 1: consensus formation with W-MSR** (`_formation_goal`). The drones should reach the line
abreast, so nobody arrives far ahead. Each tick, every drone:

1. takes the forward progress (y offset from the base) of each neighbour it heard;
2. combines them with **W-MSR**, the resilient consensus filter from the course (LeBlanc et al. 2013):
   - it sorts the values;
   - it throws away up to F values above its own and up to F below it;
   - it averages the rest together with its own value;
3. uses the result to set its own forward goal: it stays at most 4 m ahead and is pulled by half of the gap between
   it and the filtered value.

The unfiltered fleet uses the plain mean instead. A neighbour that lies about its position drags the mean, but W-MSR
trims it.

**The liar during transit.** In the demo the liar flies **honestly** and broadcasts its true position during
transit. A good liar does not reveal itself early.

Transit ends when a drone is within 2 m of its entry slot, or after 30 s at the latest. The drone then switches to
EXPLORE.

**(simplified)** The consensus is applied only to forward progress, not to the full 3D formation.

---

## Part 2. Exploring unmapped terrain

### How a drone builds its map from nothing (`TerrainMap.observe`)

Five times a second, the camera's depth rays sweep the scene in front of the drone. Each ray is marched forward until
it hits something.

- **A ray passing over a cell** at height z proves that the ground there is **lower** than z. So hi ← min(hi, z).
- **A ray hitting a cell** at height z proves that the ground there is **at least** z. So lo ← max(lo, z). A wall hit
  proves only "at least this tall". Treating it as the building's height would make the air above a 35 m tower look
  free.
- **The drone's own position.** Being at altitude z above a cell proves the cell is lower than z.
- **The launch strip** is known to be clear from the start.

A cell counts as **free on my layer** if hi + 1.5 m ≤ the layer height. **Unknown cells are never treated as free for
flying.** A drone flies only into space it, or someone it trusts, has actually seen to be open.

**Sharing the map.** Every second, each drone relays a TERRAIN event:
- the cells it newly knows to be free on each layer;
- the cells where it has seen a wall at least 6 m high.

Receivers record each report under its reporter's bit.

**Telling ground from roof.** A coverage cell whose terrain includes a known wall cell is marked **roof**, meaning
"not ground, nothing to search". Cells entirely enclosed by walls, the inside of a building, are filled in as roof
too (`binary_fill_holes`). Without this, building interiors that no camera can ever see would keep the mission
running for ever.

### Who goes where

**Multi-agent algorithm 2: decentralised geodesic Voronoi partition** (`planning/nav.py`, `partition_labels`).

There are no fixed sectors. Once a second, **every drone computes its own partition, locally**:

1. It takes its own position and the last known positions of its teammates: beacons it heard, plus POSE events
   relayed from farther away, all at most 15 s old. The gated fleet leaves out any drone it has proven to be a liar.
2. It runs a single **multi-source Dijkstra** search over the 2 m navigation grid, starting from all those positions
   at once. Walls block the search. **Unknown cells count as passable**, so distances to unexplored space are still
   meaningful.
3. Every coverage cell is labelled with the drone that can reach it soonest **by path, not as the crow flies**.

The cells labelled with my own id are **my region**.

No negotiation is needed. Drones that hold the same information compute the same partition. When their information
differs, the regions overlap a little or leave small gaps, which costs some duplicate work and nothing else.

### Choosing the next target (`_plan`)

1. **Candidates** are the cells in my region that are not yet *done*. A cell is *done* if it is swept (by me, or by a
   teammate's claim my fleet accepts), holds a survivor someone reported, is roof, or has been given up.
2. **Work stealing.** If my region is finished, the candidates are the unfinished cells **anywhere**. Drones that
   finish early go and help.
3. **Scoring.** I run Dijkstra from my own position over the cells free on my layer. A candidate scores its path
   length if I can reach it, or 1.5 × its straight-line distance + 20 m if I cannot. I take the lowest score.
4. **Hysteresis.** I keep my current target unless another one is more than 6 m better. This stops drones dithering
   when partitions shift.
5. **Unreachable targets.** If the target is in unknown space, I fly to the reachable cell nearest to it: the
   **frontier**. On the way, the camera turns unknown space into known space, and the next plan gets further.
6. **The path** is shortened into straight legs, and I fly them at 3 m/s. I replan every few seconds, or as soon as
   my target is done.

**Look before you move.** If the next leg needs a turn of more than 45°, the drone first rotates on the spot so the
camera sees where it is going, and only then moves.

**Giving up on a cell.** Suppose I reach the end of my path and the target still hasn't been viewed.
- That counts as one failed attempt. After 3 failed attempts I give up on the cell.
- If the cell was unreachable and I was already close to it, I give up at once.

Abandoned cells are never counted as swept. They are reported separately.

### Does every drone fly over the whole map?

**No.** Each drone flies its own region and shares what it has swept as CLEARED events. Every other drone then marks
those cells done, **if it trusts the reporter**. That condition is where the fleets diverge (Parts 4–6).

Regions do overlap near their borders, because the camera sees about 12 m ahead. So many cells end up viewed by two
or more drones without anyone planning it. That turns out to matter for audits (Part 8).

---

## Part 3. Finding survivors

**Detection.** When a survivor's cell is viewed, the camera produces a *look*:
- a position estimate with noise of 0.2 m + 0.05 × distance;
- a heat reading;
- a motion bit.

The drone then:
1. opens a **track** for that cell and broadcasts SURVIVOR(cell, estimate) once;
2. broadcasts LOOKS(cell, heat, motion) for every look it takes;
3. stops counting that cell as "swept and empty". A cell with a survivor in it is never reported as CLEARED.

### Alive or dead (`epistemic/belief.py`)

This is a **belief**, not knowledge.

1. **Evidence from each look.** Each look adds a log-likelihood ratio log p(look | alive) / p(look | dead), computed
   from the heat reading (two Gaussians) and the motion bit.
2. **Hovering to confirm.** If the belief is undecided, the drone switches to **CONFIRM** mode: it hovers and looks
   at the survivor.
3. **Deciding, with an SPRT** (sequential probability ratio test). The drone declares *alive* once P(alive) > 0.95,
   and *dead* once it drops below 0.05. After 8 s it gives up and records "unsure".
4. **Teammates' looks.** The naive fleet adds every teammate's looks to its belief. The gated fleet adds only looks
   from drones it knows are honest.

In tests the SPRT is right at least 95% of the time and needs about 2–4 looks. The report contrasts this
*probabilistic belief* B_i with the *certain knowledge* K_i used for liars.

### Where exactly is the survivor?

**Multi-agent algorithm 3: resilient distributed estimation.** Every 2 s, each drone with a track broadcasts its
estimate (TRACK_EST). Each drone combines its own mean estimate with its teammates' estimates, again using **W-MSR**
(or the mean, in the unfiltered fleet). This is the distributed state estimation problem from the course, made
resilient.

In `scripts/exp_localisation.py`, one lying observer drags the plain mean all the way to its lie: an 8 m lie means
8 m of error. W-MSR stays within about 0.5 m, however large the lie.

### What happens to the count of survivors?

**Each drone keeps its own list** of survivor tracks: the ones it saw and the ones it was told about. There is **no
fleet-wide consensus** on a single "number of survivors"; every drone's list fills up as SURVIVOR events spread.

The number in the table above comes from the **grader** (`analysis/mission.py`), not from the drones. A survivor
counts as **found** once any honest drone has detected it with its own camera. The grader also reports:
- how many survivors were alive, and how many of them were found;
- how many were missed **inside the liar's claimed area** and how many elsewhere;
- whether the drones' alive/dead decisions were correct.

---

## Part 4. The liar starts lying (from about 12 s)

`aerial/resilience/liar.py`. Transit is over and `attack.start_time` has passed. In the demo the liar:

1. **Moves physically (ambush):** every 14 s it picks the nearest moving honest drone, flies politely (keeping its
   separation) to a spot about 10 m ahead of it at the same height, and sits still there. It never moves into a
   drone.
2. **Lies about its position:** its beacons claim it is flying a lawnmower pattern over its claim region, starting
   from where it really is, so there is no instant jump to give it away. Its claimed speed is 0.9 of the maximum.
   - In `range_aware` mode it bends each fake point so that it matches the true ranges that the drones hearing it
     will measure, wherever geometry allows.
   - With three or more listeners that is impossible, and the fake point collapses onto the truth.
3. **Lies about coverage:** it sends one CLEARED event covering **every cell** of its 50 × 50 m claim region, which
   says "swept, nobody here".
4. **Hides survivors:** it never sends SURVIVOR or LOOKS.

Within a few ticks, that CLEARED event has been relayed to every drone.

### The naive fleet

1. Every naive drone marks the whole corner as **done**, because "anyone reported it" is enough for the naive fleet.
2. The partition still hands parts of the corner to nearby drones, but those cells are no longer candidates.
3. Nobody flies into the corner, and nobody's camera reaches deep inside it.
4. **Six survivors in the corner are never found.** They stay red in the animation, and at the end the naive fleet
   lands believing the whole zone has been searched.

### The gated fleet

The gated fleet also *believes* the claim at first. It uses belief to share out work, otherwise it would re-sweep
everything twice. But it does not stop there.

---

## Part 5. Catching the liar (gated fleet only, about 13 s)

**Multi-agent algorithm 4: S5 possible-worlds reasoning** (`epistemic/hypotheses.py`).

Each gated drone keeps the list of every possible set of at most F liars:

| Fleet size | Possible liar sets |
|---|---|
| N = 10, F = 1 | 10 (no liar, or exactly one of the 9 other drones) |
| N = 20, F = 2 | 191 |

Evidence **removes** sets from that list. Knowledge follows the definition from the epistemic-logic lecture:

- "I **know** j is a liar" means j is in **every** set I have left.
- "I **know** j is honest" means j is in **none** of them.

### Proofs of lying

Each rule below can **never** be triggered by an honest drone (`epistemic/evidence.py`), so none of them can wrongly
accuse anyone.

| Rule | What I check | Why an honest drone can't trigger it |
|---|---|---|
| **R1** range | j's beacon puts it at distance d from me, but the UWB range of that same packet says r, and \|d − r\| > 0.2 m | An honest beacon carries the true position, and range noise is at most 0.2 m. |
| **R2** speed | j's own beacons imply a speed above 5 m/s horizontally or 2 m/s vertically, or a jump between beacons | Honest drones obey the limits. |
| **R3** horizon | j claims to be more than 35 m away, but I heard it | Radio does not reach beyond 35 m. |
| **R4** camera | j's claimed spot is inside the part of my camera view that I can see is empty, and I see no drone there | Drones in view are always seen (assumption 4). |
| **R5** audit | I found a survivor in a cell that j claimed swept and empty | Survivors are always detected in viewed cells, and they never move. |

Each of these proofs **eliminates every world in which j is honest**.

Two weaker kinds of evidence only narrow things down:

| Evidence | What it proves |
|---|---|
| **R6** shared ranges: drone r reports its own range to j for a given beacon, and it contradicts that beacon | "j or r is lying" |
| **An accusation** from drone a against j | "a or j is lying". Honest drones accuse only with proof, so if a is honest then j is guilty, but a might be the liar framing j. |

### What happened in the demo

1. At about 13 s, a gated drone within radio range compared the liar's beacon (which put the liar over the corner)
   with the UWB range of that same packet (the liar was actually close by). The gap was far beyond 0.2 m, so **R1
   fired**.
2. That drone's list shrank to the single world "drone 0 lies". It now knows the liar, and it broadcasts ACCUSE.
3. **How the knowledge spreads.** With F = 1, a single accusation is **not** enough for other drones, because the
   accuser could itself be the liar framing drone 0. Each receiver learns only "the accuser or drone 0 is lying".
   They become certain in one of two ways:
   - **their own proof:** every drone that also hears the liar runs R1 on the same lies, which is what happened here;
   - **a second, independent accuser:** two accusers of the same drone leave only one possible liar when F = 1.

   This is a *threshold-2 contagion*, not the simple SI epidemic from the course. A lone liar's false accusation can
   never make an honest drone "known guilty".
4. By **13.3 s every honest drone knew drone 0 was the liar.** In the animation the liar turns orange.

**Budget exhaustion.** With F = 1, once the liar is known every other drone is **known honest**. Every claim from
those drones instantly counts as knowledge.

---

## Part 6. After the catch: exclusion and re-sweep

Once a gated drone knows drone 0 is the liar:

1. **Its claims are discarded.** The belief standard now leaves out reporters proven guilty. Every cell whose "swept"
   status rested **only** on drone 0 goes back to "not done". That set of cells is exactly what "the liar's swept
   area" means: cells nobody else vouched for.
2. **It leaves the partition.** Drone 0 is no longer a Voronoi source, so its region is absorbed by honest
   neighbours.
3. **Its broadcasts are no longer trusted.** Its position reports stop counting as truth: the motion gate (Part 7)
   reasons about where it could *really* be. Its survivor estimates and looks are ignored too.
4. **Re-sweep.** The corner is now full of undone cells, so the normal Part 2 machinery sends drones there. They find
   the 6 survivors the naive fleet missed, and hover to judge whether each is alive.

No special "re-sweep protocol" was needed. Exclusion plus the dynamic partition does it on its own.

---

## Part 7. Collisions and the knowledge gate

### The safety layer, used by every fleet and never switched off (`_safety`)

- **Separation.** The drone is pushed away from any neighbour closer than 3 m, harder if it is closing in. "Closing
  in" is measured with the drone's **own** velocity, never a reported one, so a liar can't fake an approach. The
  neighbours used are the positions in recent beacons plus anything the camera sees.
- **Terrain.** Known walls within 3 m push the drone away.
- **Geofence.** The drone is kept inside the arena.

### Why the naive fleet collides

**Not with the liar.** Naive drones ignore where the liar *really* is, because they believe its fake position. Even so,
in our runs they don't hit it: the liar sits *in front* of them, their camera sees it, and the safety layer steers
around anything the camera sees.

**With each other.** All 8 of the naive fleet's collisions were between honest drones. The safety layer's 3 m push is
only a reflex. When the liar's false claim takes a whole corner out of the search, the rest of the fleet crowds into
less space, and drones converging on the same cells come closer than 1 m.

**What the gate changes.** The gated fleet's knowledge gate refuses any move it can't prove is safe, so its drones
hold or give way instead, and it had 0 collisions.

**A liar that deliberately flies into drones** (not modelled here, `attack.motion=shadow` comes close) beats *both*
fleets, because the gate controls only the drone's own move. That stays out of scope, as stated in the plan.

### The knowledge gate (gated fleet; `epistemic/gate.py`)

Before each move, the drone checks that it **knows** the next short stretch of flight is safe. That stretch is one
tick at the commanded acceleration, then hard braking for 0.3 s. The check is done against every place each
neighbour *could* be.

**Where a neighbour could be**, for each status it might have:
- **If it is honest,** its report is true, so it is within a small ball around its reported position, extrapolated
  with its reported velocity, of radius ½ a_max t².
- **If it might be the liar,** all I know is my UWB range to it, so it is somewhere on a **sphere of that radius
  around me**. The sphere is thickened by its possible movement. Parts of the sphere that my camera sees empty are
  removed.
- **If I haven't heard it for 9 ticks,** it is out of reach (assumption 7).

**If the move is safe,** the drone flies it. Otherwise:
1. The drone **holds**: it brakes, while the safety layer stays on.
2. It **re-queries** its neighbours for fresh data.
3. **Right of way.** After 1 s of holding, the drone with the higher id backs away from the drone blocking it.
4. **Detour.** After 3 s of holding, the drone gives up its current target and replans elsewhere.

**Result.** Gated drones hold, back off or detour, and in this run none of them collided: **0 collisions**. The cost
is time spent holding, but here it was small: the gated fleet had landed by 192 s against the naive fleet's 189 s,
while also re-sweeping the corner the naive fleet skipped.

**(simplified)** The check against the range sphere is analytic and looks only in the direction of motion. It is not
the full cell-by-cell check described in the plan. The rule "a coverage claim that is impossible from where the drone
actually was" (R7) is **not built yet**.

---

## Part 8. Audits: what the gated fleet does when it cannot catch anyone

Some liars never fake their position. They fly where they say they are, and lie only about what they saw. R1–R4
cannot catch them. **Audits** can.

**When.** A gated drone audits once its own work looks done.

**The budget, per teammate j.** Re-check ⌈q · n_j⌉ of the n_j cells that j claimed. q is the audit fraction
(default 0.1) and is the caution knob.
- **What counts as an audit:** any view of a claimed cell by somebody other than j. That includes my own views, a
  second teammate's claim of the same cell, and AUDITED events.
- **The liar cannot audit its own claims.**
- **Sampling.** The cells still to be audited are sampled **uniformly at random** from j's unchecked claims, using
  the drone's private random stream. This is what makes "a hidden survivor is caught with probability ≥ the audited
  fraction" a true statement.
- **Visiting.** Sampled cells are visited nearest-first. The order of visits does not change the guarantee.
- **Catching.** Finding a survivor in an audited cell is proof R5.

**A finding from the no-liar runs.** Dynamic regions overlap so much that the audit budget is usually met **for
free**, so with no liar the gated fleet costs only about +0.5 s over naive (128.6 s vs 128.1 s).

**The weakness.** With q = 0.1 and only a few hidden survivors in a large claimed area, an audit can easily miss them.
In an earlier run with a liar that lied only about coverage, it was never caught. That is why the demo uses a liar
that also fakes its position.

**What is reported.** `audit_report()` gives the fraction of each teammate's claims *I* checked myself. That is the
residual risk: "a single hidden survivor in j's area would have been found with probability ≥ that fraction".

---

## Part 9. Deciding to stop and coming home

### The mode machine (`_update_mode`)

| From | Condition | To |
|---|---|---|
| TRANSIT | near the entry slot, or after 30 s | EXPLORE |
| EXPLORE or AUDIT | a track I have seen is still undecided and less than 8 s old | CONFIRM |
| CONFIRM | the survivor is decided alive or dead, or 8 s have passed | EXPLORE |
| EXPLORE | no undone cell left (under my fleet's standard) | AUDIT if audits are pending (gated only), otherwise RETURN |
| AUDIT | audit budget spent | RETURN |
| RETURN | within 1.5 m of my landing slot | LANDED |
| AUDIT, RETURN or LANDED | **undone cells reappear**, for example after a late catch reverts the liar's cells | EXPLORE |

### When does a drone stop?

When, as far as **it** can tell, every cell in the zone is swept, holds a reported survivor, is roof, or was given up
as unviewable, and, in the gated fleet, the audit budget has been spent.

That is a **local** decision; there is no vote. The mission ends when **every honest drone has landed**, or when the
time limit runs out.

- The naive fleet stops after 210 s, believing the liar's corner was searched. It wasn't.
- The gated fleet would only have stopped after re-sweeping the corner and spending its audit budget. In this run it
  was still working at 240 s.

### Coming home

In RETURN, the drone plans a path to its own **landing slot** on the base, flies it on its layer, descends to 4 m in
the last 3 m, and switches to LANDED.

**(simplified)** The plan described the regroup as a formation-consensus manoeuvre into the slots; the code uses
ordinary path planning.

**What each drone knows at the end:**
- its terrain intervals;
- its coverage map;
- which cells are only *believed* swept and which are *known* swept (gated fleet);
- its survivor tracks with their alive/dead decisions and localised positions;
- in the gated fleet, the proven identity of the liar.

---

## Part 10. Grading the run

The drones never see this part (`analysis/mission.py`, `spec/mission_specs.py`).

**Safety**
- The minimum distance between any two drones, counted over all drones, liars included.
- **Collisions**: each time a pair falls below d_safe = 1 m.
- **Terrain hits.**

**Mission** (honest drones only)
- Survivors found.
- Survivors missed, split into inside the liar's area and elsewhere.
- Whether the alive/dead decisions were correct.
- Coverage of viewable cells.
- Completion time.

**Knowledge**
- When the first honest drone proved who the liar was, and when all of them knew.
- Number of false accusations. This **must always be 0**, and is 0 in every run so far.

**Signal Temporal Logic** (offline monitor, ported from the first build)

| Formula | Meaning |
|---|---|
| φ_safe = G[0,T] (separation ≥ 1 m ∧ above the terrain ∧ inside the arena) | never unsafe |
| φ_find = F[0,T] (survivor detected), one per survivor | each survivor is eventually found |
| φ_know = F[0,T] (every honest drone knows the liar) | knowledge eventually spreads |
| φ_cover = F[0,T] (the fraction known to be cleared reaches 95%) | the area is eventually known searched |

Each formula gives a robustness score ρ. ρ < 0 means the run violated it. The dashboard plots ρ_safe(t) over time for
both fleets.

---

## Summary tables

### Multi-agent algorithms used

| Algorithm | Where | Course topic |
|---|---|---|
| Consensus formation with W-MSR (and mean for comparison) | transit (`_formation_goal`) | consensus, formation control, resilient consensus |
| Resilient distributed estimation (W-MSR on survivor positions) | survivor tracks (`track_estimate`) | distributed state estimation |
| Decentralised geodesic Voronoi partition, plus work stealing | who goes where (`partition_labels`, `_plan`) | coverage and task allocation |
| Gossip / flooding with one-hop latency, priorities and re-query | all sharing (`comms/network.py`) | information spread (SI infection model) |
| S5 possible worlds over sets of liars; knowledge as truth in all worlds | liar identity, claim trust (`hypotheses.py`, `knowledge.py`) | epistemic logic, Kripke models |
| Nested knowledge and exact world counting | `epistemic/nested.py`, checked against the Kripke engine | nested knowledge |
| Accusations as threshold-2 contagion | `ACCUSE` handling | infection model, compared |
| SPRT belief for alive/dead | `belief.py` | belief vs knowledge |
| Knowledge gate on motion, hold, re-query, right of way | `gate.py` | behavioural planning with epistemic logic |
| STL robustness grading | `spec/` | STL |
| λ₂ convergence, (r,s)-robustness, ER and degradation studies | `aerial/lab/` (first build, ported) | graph theory, resilient consensus |

### What each fleet trusts

| Claim | naive | gated |
|---|---|---|
| "I swept these cells, nobody there" | always | yes, unless the reporter is a proven liar; also audited |
| "This airspace is free" (terrain) | always | same rule |
| "There is a survivor here" | always | same rule |
| "I am at position p" (beacon) | always, and uses it to avoid the drone | checked against UWB range, camera and speed limits; once j is proven a liar, its position becomes "somewhere on my range sphere" |
| A teammate's alive/dead looks | always | only from drones known honest |
| A teammate's survivor-position estimate | W-MSR | W-MSR, excluding proven liars |
| "j is a liar" (accusation) | ignored (no detection) | means "accuser or j"; certain only with a second independent proof |

---

## Gaps between this build and the plan

All of these are honest current limits:

- **No 30-seed sweep yet.** The numbers in this document come from one demo seed.
- **R7 is not built.** A coverage claim that is impossible from where the liar actually was cannot yet be used as
  proof.
- **Completion uses belief plus audits.** The plan wanted completion to require *known*-cleared cells. The code
  reports the known-cleared fraction but does not wait for it.
- **Simplified pieces:** the regroup at the base, the altitude-layer rule, and the gate's analytic sphere check.
- **Demo liar settings.** The demo liar is configured to be blatant (`claim_all`, no quiet-spot seeking). A subtler
  liar that never fakes its position is caught only by audits, and in our runs it often was not.
