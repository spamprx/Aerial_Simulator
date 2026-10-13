# AERIAL: project context and rebuild plan

This file holds everything needed to build AERIAL again from an empty repository, with the planning done up front. It
contains the problem, the objectives and how each one is met, the architecture and its interfaces, the decisions and why
they were made, the traps the first build fell into, and a phased plan with acceptance criteria.

It was written from the finished first build: `TODO.md` (the task cards), `docs/decisions.md`, `docs/story.md` (the demo
walked through step by step), `docs/report/*.md` (measured results) and `docs/proposal.pdf`. Wherever this file and the
code disagree, the code and `docs/decisions.md` describe what was actually built. This file describes what should be built.

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Thesis and what counts as success](#2-thesis-and-what-counts-as-success)
3. [Scope: committed, and out of scope](#3-scope-committed-and-out-of-scope)
4. [Objectives, and how each is met](#4-objectives-and-how-each-is-met)
5. [System model](#5-system-model)
6. [Architecture](#6-architecture)
7. [Core interfaces (fix these first)](#7-core-interfaces-fix-these-first)
8. [Design decisions (lock on day 1)](#8-design-decisions-lock-on-day-1)
9. [The epistemic layer in detail](#9-the-epistemic-layer-in-detail)
10. [Lessons from the first build](#10-lessons-from-the-first-build)
11. [Build plan: phases, dependencies, owners](#11-build-plan-phases-dependencies-owners)
12. [Testing strategy](#12-testing-strategy)
13. [Experiments and the results to reproduce](#13-experiments-and-the-results-to-reproduce)
14. [Engineering rules](#14-engineering-rules)
15. [Known limits and future work](#15-known-limits-and-future-work)
16. [Parameter reference](#16-parameter-reference)

---

## 1. Problem statement

The course is AI3403 (Multi-Agent Systems), and the group is Aerial. The project title is *Resilient 3D Aerial Traffic
Management and Knowledge-Gated Coordination for UAV Fleets in Disaster Response*.

After a disaster, a fleet of UAVs has to sweep damaged sectors and find survivors. Ground communication is down and there is
no central controller. Each UAV talks only to neighbours within its short radio range, so the communication graph G(t)
changes as UAVs move and packets are lost. Two things go wrong:

1. **Links drop at random.** The graph is time-varying and can disconnect.
2. **Some UAVs are compromised** and broadcast false positions. That breaks ordinary consensus completely: one liar can drag
   the whole fleet more than 100 m off.

The standard defence is **W-MSR** filtering, in which each UAV throws away the F most extreme received values on each side
before it averages. This restores convergence, but it also throws information away. A covert liar that keeps its report
inside the honest range is never trimmed. A filtered estimate can therefore say "the airspace ahead is clear" when some
states consistent with what the UAV received are not clear.

## 2. Thesis and what counts as success

> **The filtering that makes a fleet resilient also makes it ignorant. We quantify that trade-off.**

A UAV should **advance only when it knows the move is safe**: when the move is safe in every world consistent with what it
genuinely observed. Otherwise it holds (brakes) and re-queries. This is an epistemic knowledge gate, Kᵢ(φ) ≡ Iᵢ(w) ⊆ ⟦φ⟧
under S5, placed above the resilient coordination layer. STL robustness grades every run, and the cost of caution is
measured as extra mission time.

**The central demo** is one image. Two fleets fly side by side under an identical attack. The naive fleet, on the left,
moves fast and then breaks safety because it trusts spoofed data. The gated fleet, on the right, hesitates, re-queries,
finishes later and stays safe.

**Success criteria for the rebuild.** The first build met all of these, and the rebuild must meet them again:

| Criterion | Target | First build |
|---|---|---|
| Naive fleet breaks d_safe on the demo | ≥ 70% of 30 seeds | 100% |
| Gated fleet breaks d_safe on the demo | 0% of 30 seeds | 0% |
| What the gate costs when nobody lies | within 10% of naive completion | +0.03 s (0.07%) |
| Naive and gated families in the trade-off curve | visibly separate groups | all 24 naive configs at ρ < 0, all 216 gated at ρ > 0 |
| W-MSR when the sufficient condition holds | 100% success | 100% of 2,325 runs |
| One gated 30 s run at N = 20 | ≤ 10 s | 8.2 s (2-D), 8.8 s (3-D) |

## 3. Scope: committed, and out of scope

**Committed, from the proposal:**
- A Python simulator (numpy, networkx, matplotlib) for up to **N = 20** UAVs, in 2-D first and then 3-D, with time-varying
  disk graphs and packet drops. Fixed-step integration, with no aerodynamics.
- Coordination: rendezvous, velocity alignment, formation with offsets, leader–follower pinning, BOIDS, a behaviour-switching
  controller, step-size and stability checks, and a study of convergence against λ₂(L).
- Resilience: W-MSR in every primitive, F-local attacks with **F ≤ 2**, three attack types (constant offset, slow drift and
  random spoof), an (r, s)-robustness checker, Erdős–Rényi experiments, and degradation as F grows.
- Specification: an **offline** STL robustness monitor written from the quantitative semantics. There is no online synthesis.
- Knowledge: possible worlds over a **deliberately coarse, finite** abstraction (≤ F faulty plus a coarse grid), with its
  size reported, an S5 evaluator including nested knowledge, and a gate that changes behaviour.
- Evaluation: naive against gated over repeated seeded runs, plus the caution-versus-completion-time trade-off curve.
- A Streamlit dashboard, a report, and reproducible scripts on GitHub.

**Out of scope, and stated as such in the report:** aerodynamics; online STL synthesis; equivocation (a liar sending
different lies to different receivers); sensing obstacles (the map is known); live survivor detection (survivors are found
in grading); handing work over in flight; rams by liars.

**Dependencies:** numpy, networkx, matplotlib, pytest and streamlit. Anything else needs the team's agreement; plotly
is allowed if the team agrees.

## 4. Objectives, and how each is met

| # | Objective (proposal) | How it is met | Evidence |
|---|---|---|---|
| O1 | Double-integrator fleet on a time-varying disk graph with drops | `step_double_integrator`, which is exact under zero-order hold when unsaturated and trapezoidal when saturated. `DiskComm`: A_ij = 1[‖pᵢ−pⱼ‖ ≤ R]·Bernoulli(1−p). | dynamics and graph tests, λ₂ checked against closed forms |
| O2 | Consensus behaviours | One `Aggregator` interface (`mean` or `wmsr`) shared by consensus, formation, leader–follower, BOIDS and mission. Normalised gains (D11). | formation error < 1e-2; BOIDS keeps d_safe |
| O3 | Stability checks | The exact one-step matrix M(λ), its spectral radius over λ ∈ (0, λ_max], and a warning in `Simulator`. Worst case λ_max = min(2·d_max, N). | predicts that N=20 complete with k_v=3 diverges and k_v=1.5 converges |
| O4 | Convergence against λ₂ | `exp_lambda2.py`. Prediction from **every** eigenvalue of L, or of (1−p)L under loss. | 1.7% mean error with no loss; λ₂ alone gives up to 45% error on dense graphs |
| O5 | W-MSR in every primitive | Standard W-MSR (LeBlanc 2013), per coordinate, with the agent's own value included. Damping with the agent's own velocity (DP-MSR, Dibaji and Ishii 2015). 2F+1 pinned leaders. | the unfiltered fleet drifts 2,000 m; the filtered fleet stays in the honest hull |
| O6 | F-local attacks, three types plus covert | `Attack.corrupt(...)` falsifies the broadcast only. `covert` stays in the inner honest range so W-MSR accepts it. | covert rejected on 0% of steps in 4 test cases |
| O7 | (r, s)-robustness checker | A bitmask and subset-zeta version in O(N·2^N), checked against a 3^N brute force for N ≤ 10. | 20 nodes in under 30 s; agrees on 300 random graphs |
| O8 | ER experiments and degradation | `exp_er_robustness.py`, `exp_degradation.py`. | 100% success when (2F+1)-robust; W-MSR discards 29% of honest messages at F=1 and 51% at F=2 |
| O9 | Offline STL | An AST (`Pred`, `Not`, `And`, `Or`, `Always`, `Eventually`, `Until`), sliding-window O(K), and `prefix_robustness` for ρ(t). The specs are built from the **same** predicate as the gate. | hand-computed tests per operator |
| O10 | Possible worlds, S5, nested knowledge | `PossibleWorlds` built from own state, received reports, **physical ranges** and own history. `knows_safe` decides per neighbour, with no enumeration. Checked against a generic S5 Kripke engine. | the actual world lies in Iᵢ at every step of 50 attacked runs |
| O11 | Gate changes behaviour | `KnowledgeGate` splits u into `advance` (gated) and `safety` (never gated). Hold = brake + safety, plus a re-query. Detours after a patience period. | demo: 0 collisions and ρ = +1.87, against 2 collisions and ρ = −0.69 |
| O12 | Scenarios and mission metrics | `generate_scenario(name, seed)`: sectors, a wall with a corridor, survivors, a launch base, lanes and lawnmower routes. Metrics cover honest agents only (D9). | every survivor reachable by BFS; baseline within 10% |
| O13 | Naive against gated, trade-off curve | `run_pair` with **asserted** pairing, and `exp_tradeoff.py` (14,400 runs). | the gated run has the higher ρ in all 6,480 pairs |
| O14 | Dashboard | `dashboard/app.py`: side-by-side fleets, a time slider, four plots and a batch tab. Data preparation is plain functions, tested headless. | `AppTest` checklist in `docs/acceptance.md` |
| O15 | Reproducible deliverables | `results/<exp>/<time>/` holding config, grid, rows and env (git hash, versions, seeds). `make test figures dashboard`. CI on Python 3.10 and 3.12. | a fresh clone regenerates every figure |

## 5. System model

**Agents.** N ≤ 20 point UAVs, with ṗ = v and v̇ = u, in 2-D or 3-D. Per tick:
`u ← sat(u, a_max)`, `v' = sat(v + u·dt, v_max)`, `p' = p + ½(v + v')·dt`.

**Communication.** Broadcast only (D7). A[i, j] = True means *i received j*, so row i lists i's in-neighbours. Drops are
drawn as a full (N, N) block every step, whatever the controller does. A message cache with a TTL is keyed by
**(receiver, sender)**.

**Sensing.** Every received packet carries a physical-layer range rᵢⱼ = ‖pᵢ − pⱼ‖ + σ·clip(n, −4, 4), which no liar can
fake (D5). Optionally, UAVs share the ranges they measured, through a payload that arrives one tick late.

**Adversary.** Up to F compromised UAVs (F-local on fixed graphs, F-total on disk graphs). They falsify their **broadcast
only**, never their true state. Their physical motion is `same`, `hover` or `intercept` (D6).

**World.** An arena box (the geofence), spherical obstacles ("collapsed buildings") and survivor targets. The map is known
to every controller.

**Randomness.** `SeedSequence(seed).spawn(6)` gives `init`, `comm`, `attack`, `ctrl`, `requery` and `sense`. No stream
may be drawn from in a way that depends on the controller. That is what makes naive and gated runs of the same seed
comparable run by run. Keep the first three in this order, so old seeds reproduce.

## 6. Architecture

```
aerial/
  core/          config.py, dynamics.py, world.py, simulator.py   (sense → decide → integrate; SimResult + extras)
  network/       graph.py (topologies, Laplacian, λ₂), comm.py (DiskComm, drops), messages.py (per-receiver TTL cache)
  coordination/  base.py (Controller, SplitController), consensus.py, formation.py, boids.py, mission.py, switching.py
  resilience/    aggregators.py (mean, wmsr), attacks.py, adversary.py, flocal.py, robustness.py
  epistemic/     kripke.py (S5 engine), abstraction.py (Grid), worlds.py (I_i, proofs of lying, knows_safe),
                 gate.py (gated, oracle), detour.py (going round), nested.py (K_i K_j)
  safety.py      ONE Safe predicate (D2) + never-gated terms: separation, obstacle, geofence, brake, prioritise, steer
  stl.py         generic STL engine;  specs_library.py  mission specs built on safety.py
  stability.py   M(λ), spectral radius, max stable dt, predicted rate
  scenarios.py   generate_scenario, fleet_overrides (the four fleets of D1)
  analysis/      metrics.py, mission.py (mission metrics), stl.py (per-run numbers)
  experiments/   runner.py: run_one, run_batch, sweep (parallel), run_pair, pair_sweep, record_experiment
  viz/           plots.py, live.py
  registry.py    Registry("controller" | "attack" | "aggregator"); everything is imported from aerial/__init__.py
dashboard/app.py   scripts/{exp_*,fig_*,validate_*,make_*}.py   configs/   tests/   docs/
```

**Use this layout from day 1.** The first build began with flat modules (`sim.py`, `controllers.py`, …), moved to
subpackages part-way through, and still carries re-export shims. Starting with the subpackages avoids that.

**One step of the simulator:**
1. **Sense.** Broadcast the true states, pass the compromised rows through `Attack.corrupt`, draw drops, apply last step's
   re-queries, measure ranges, update the per-receiver cache, and build an `Observation`.
2. **Decide.** `u = controller.control(obs)`, then the rows for compromised agents are replaced by the adversary motion.
   Diagnostics, payload and re-query requests are collected.
3. **Integrate.** Apply the double integrator. If `record_extras` is set, log the extras.

**The controller stack, outermost first:** `KnowledgeGate` → `Mission` (BOIDS or formation along routes, through an
aggregator) → safety terms. The gate may withhold only the `advance` part.

## 7. Core interfaces (fix these first)

The first build added most of these hooks late, card by card. Define all of them in week 1, so that every later card only
fills them in.

```python
@dataclass
class Observation:
    t: float; k: int
    pos: np.ndarray; vel: np.ndarray          # TRUE states: a controller may read row i only for agent i
    adj: np.ndarray                           # (N,N) bool, adj[i,j] = i holds j's message
    msg_pos: np.ndarray; msg_vel: np.ndarray  # broadcasts (N,d), or per receiver (N,N,d) when ttl > 0
    msg_age: np.ndarray                       # (N,N) ticks since receipt
    ranges: Optional[np.ndarray]              # (N,N), inf where nothing was received
    msg_payload: Dict[str, np.ndarray]        # last tick's controller payloads, same availability as adj
    world: World; cfg: SimConfig
    def received(self, i) -> (ids, pos, vel, age)   # exactly what agent i holds; controllers use only this

class Controller(ABC):
    def reset(self, sim): ...
    def control(self, obs) -> np.ndarray: ...               # (N,d)
    def diagnostics(self) -> Dict[str, np.ndarray]: ...     # stacked into SimResult.extras[key], shape (K, ...)
    def broadcast_payload(self) -> Dict[str, np.ndarray]: ...  # delivered next tick (one-step latency)
    def requery_requests(self) -> Optional[np.ndarray]: ... # (N,) bool, served next tick (D4)
    def normalised_gains(self) -> Optional[Tuple[float, float]]: ...  # for the stability warning

class SplitController(Controller):
    def control_parts(self, obs) -> {"advance": (N,d), "safety": (N,d)}
    def control(self, obs): return prioritise(parts["advance"], parts["safety"], a_max)

class Aggregator(ABC):
    def aggregate(self, own, values, ids) -> AggResult(value, trusted, rejected, count)

class Attack(ABC):
    def reset(self, sim): ...
    def corrupt(self, t, true_pos, true_vel, compromised, rng, adj=None) -> (bcast_pos, bcast_vel)
    def corrupt_payload(self, payload, compromised) -> payload   # identity by default

@dataclass
class SimResult:
    t, pos (K+1,N,d), vel, u (K,N,d), adj (K,N,N), lambda2 (K,), compromised, config
    extras: Dict[str, np.ndarray]   # msg_pos, msg_vel, drop_keep, attack_noise, + every diagnostics key
    def save(path) / load(path)     # .npz with extras__<key>
```

**The extras the dashboard and the tests read:** `msg_pos`, `msg_vel`, `drop_keep`, `attack_noise`, `accepted*`,
`rejected`, `gate_open`, `k_safe`, `known_faulty`, `log10_worlds`, `n_hypotheses`, `requery`, `mode`, `f_local_ok`.

**Configuration** is one flat `SimConfig` dataclass, saved as JSON, with dotted overrides (`controller_params.k_p=2`).
Validation enforces N ≤ 20, F ≤ 2, `n_compromised ≤ F` unless `allow_beyond_assumption` is set, `comm_radius > 0`,
`d_safe > 0`, the shapes of `init_pos` and `init_vel`, and, once the gate is on,
`comm_radius > d_safe + 2·v_max·τ + h`. Nested dicts: `world`, `gate`, `sensing`, `mission`, `controller_params`,
`attack_params`.

## 8. Design decisions (lock on day 1)

The first build settled these before Phase 2. The rebuild should settle them **before writing code**. Each one affects
several modules. The full reasoning is in `docs/decisions.md`.

| ID | Decision | Why |
|---|---|---|
| D1 | **Naive** = mission controller + W-MSR, gate off. **Gated** = the same with the gate. Two references: **unfiltered** (no W-MSR) and **oracle** (told the true faulty set). The four differ only in the controller (`fleet_overrides`). | Filtered against filtered-and-gated is the proposal's comparison. The oracle measures the price of ignorance. |
| D2 | **One Safe predicate**: ‖pᵢ−pⱼ‖ ≥ d_safe for *all* pairs (liars included), obstacle clearance ≥ 0, inside the arena. It lives in `safety.py`, and both the gate and STL read it. | If the gate and the grade disagree, the results cannot be explained. A collision is physical. |
| D3 | **Hold = brake**: `u = sat(−v/dt, a_max)` + the never-gated safety terms. Coasting is never a hold. | u = 0 on a double integrator coasts into things. |
| D4 | **Re-query**: a held agent gets one extra retransmission attempt from each in-range neighbour on the next tick, with fresh ranges, at age 0. `rng_requery` draws a full (N, N) block **every** tick. | Pairing has to survive it. |
| D5 | **Ranging** (option A): a liar cannot fake the distance. If j is faulty, j lies on the shell \|‖x−pᵢ‖−rᵢⱼ\| ≤ ε′, with ε′ = max(ε, 4σ). Both fleets receive the ranges; only the gate uses them. | Without an unspoofable signal a faulty neighbour could be anywhere in radio range, and a sound gate never opens for F ≥ 1. Option B (reachable sets) grows without bound and freezes the fleet. |
| D6 | **Adversary motion**: `same`, `hover` or `intercept`. It uses true states only. | The demo needs a liar that is physically in the way while it reports being elsewhere. |
| D7 | **Broadcast only**, no equivocation. | Scope. W-MSR's F-local results still apply. |
| D8 | **Assumed F** separate from **actual n_compromised**. F-local is checked on fixed graphs; on disk graphs the total must be ≤ F. | The degradation study has to break the assumption on purpose. |
| D9 | Mission metrics cover **honest** agents only. Safety covers **all** agents. | A liar has no duty to arrive, but it can still be hit. |
| D10 | Grid h = d_safe, look-ahead τ = 5·dt, ε = 0.1 m, all under `gate`. h and τ are the caution knobs. | The proposal promised a coarse abstraction. |
| D11 | New controllers use **mean** (normalised) weights. The summed `consensus` controller stays for the λ₂ study. | Summed gains grow with degree, and N=20 complete diverged to 1e119. |

**Decisions the first build reached only while building. Adopt them from the start:**
- Separation uses **only the agent's own velocity** for closing speed, and never a reported velocity. A liar claiming
  20 m/s otherwise took all control authority from 50 m away.
- Under W-MSR, damp with the **agent's own velocity** (DP-MSR). W-MSR does not conserve mean velocity, and the fleet
  otherwise sailed 210 m away together.
- Leader–follower under W-MSR pins **2F + 1** agents, because W-MSR drops a lone leader as an extreme.
- A UAV on its own route has cohesion and alignment **off**. Cohesion pulls it back out of its sector.
- **W-MSR's accept and reject masks are never evidence of lying.** It trims honest values too: 29% of them at F = 1.
- **r_sep > the gate clearance C** = d_safe + h√d + max(h, 2ε′) + v_max·τ (4.66 m at the defaults). Otherwise two honest
  UAVs hold each other for ever. Lane spacing s = r_sep + 1.
- **Shared ranges plus detours** are needed to get past a hovering liar. With a range alone there is no bearing, and a
  liar within about 3.75 m of a route blocks it for ever.
- **Backing away** with the gate shut is allowed only on a heading that moves away from **every** possibly-occupied cell.
  A rule that only moved away from the nearest cell flew a UAV into a liar.
- Routes are planned as **straight clear legs** (BFS plus shortcutting) before take-off. Pure potential fields trap a UAV
  at a wall in front of its goal.
- **Each launch row gets its own corridor lane.** With one shared corridor waypoint, three UAVs sat in an 80 s standoff with
  nobody lying.
- Sectors go only to honest UAVs. Base rows map to sector rows, and the UAV in front takes the far sector, so routes never
  cross.
- The stability worst case is min(2·d_max, N), not Gershgorin's 2·d_max, which called stable fleets unstable.
- The pairing check also covers the attack stream's **raw draws** (by wrapping the generator), not only the drops.

## 9. The epistemic layer in detail

This is the core of the project and the part most worth getting right on paper first.

**What agent i may use:** its own true state, the messages it holds (with their ages), its ranges, the payloads it
received, and its own past observations. Nothing else. A parametrised test enforces this for every registered controller:
perturb an agent that i does not hear and assert that u[i] does not change.

**Proofs of lying.** Each rule is something an honest agent can never trigger, so no rule can accuse an honest agent.
Once caught, an agent stays caught.
1. The report disagrees with the range: \|‖p̂ⱼ − pᵢ‖ − rᵢⱼ\| > ε′ + h√d.
2. An impossible speed: ‖v̂ⱼ‖ > v_max.
3. A teleport: ‖p̂ⱼ(k) − p̂ⱼ(k′)‖ > v_max·dt·(k − k′) + h.
4. It claims to be beyond radio range, yet it was heard: ‖p̂ⱼ − pᵢ‖ > R.
5. Contradicted by others' shared ranges, by more reporters than the remaining faulty budget allows. This rule rests on F;
   the other four rest on physics.

**Counting.** Once |KnownFaultyᵢ| = F, every other agent is known honest, and the budget is 0.

**Where each neighbour can be** (cells of side h, dilated for the look-ahead):
- **Honest:** a ball around the report, of radius h + age·v_max·dt + v_max·τ, intersected with the ball around the
  previous report.
- **Faulty:** the range shell around where i measured, of half-width ε′ + age·v_max·dt + v_max·τ, intersected with every
  shared ring that enough reporters agree on (at least #reporters − slack).
- **Silent agents:** tracked from their last message, growing by v_max·dt per tick. An agent never heard (early in a run
  with loss) could be anywhere, so the gate stays shut. After L = ⌈ln 1e-9 / ln p_drop⌉ silent ticks the agent is taken
  to be out of range. This is the gate's **one probabilistic assumption**.

**Kᵢ(Safe) without enumeration.** Safe is a conjunction over (neighbour, cell). So Kᵢ(Safe) holds if and only if, for
every neighbour j, every status j can have under some consistent H, and every cell in cells_j(status), the plan clears
that cell centre by d_safe + h√d/2. The F-coupling between hypotheses affects only the world count and nested knowledge,
not this check. This must be verified against brute-force enumeration through the S5 Kripke engine on N ≤ 4.

**The plan** that is checked is one tick of `advance + safety`, then maximum braking until τ, on the same saturated
dynamics as the simulator.

**The gate per agent, per tick:** compute the parts → build Iᵢ → if `knows_safe(plan(advance + safety))`, apply it;
otherwise apply brake + safety and set re-query. After `patience` (1 s) of holding, detour: BFS round every possibly
occupied cell, try headings at 0, ±30°, … from the route, each of which must pass Kᵢ(Safe) along a 2·v_ref·τ probe; back
away only along headings that move away from every occupied cell; skip a blocked waypoint within r_sep.

**Nested knowledge.** Kᵢ Kⱼ Safeⱼ ≈ (j is known honest to i) ∧ (i holds j's payload) ∧ (j said k_safe). This is sound but
not complete. It is compared with the exact S5 value on small instances. It is shown on the dashboard, and the gate does not
act on it.

**Sizes to report** (2-D, N = 8, 40 m arena, h = 1): 2,116 cells and about 10^10.8 worlds at F = 1 with the liar caught.
N = 20 in 3-D reaches about 10^44. Counting worlds scans every cell and is expensive, so sweeps turn it off.

**The soundness invariant**, which is the most important test in the repository: over 50 seeded runs with attacks, at
every step and for every agent, the actual world lies in Iᵢ. That is, (true faulty set ∩ Nᵢ) is one of the hypotheses,
and each neighbour's true trajectory over τ lies in its cells for its actual status. **Write this test as soon as
`worlds.py` exists, and run it after every change to the epistemic layer.**

## 10. Lessons from the first build

**Bugs found in the base code (B1–B6)**, so the rebuild avoids them from the start:

| ID | Bug | Rule that avoids it |
|---|---|---|
| B1 | W-MSR indexed the local value array with global agent ids, which averaged the wrong values and crashed when an id ≥ the degree. | Keep local positions and agent ids separate. Test the numeric `u`, not only the masks. |
| B2 | The message cache was keyed by sender, so a receiver saw a fresh value that another receiver got. | Key the cache by (receiver, sender). |
| B3 | Summed gains grew with degree and diverged at N = 20. | D11, plus the stability check and warning. |
| B4 | The "W-MSR" was not standard: it trimmed unconditionally, left out the own value, used mean weights, and counted a neighbour as accepted if it survived in any coordinate. | Implement LeBlanc's version. `trusted` = kept in **every** coordinate. |
| B5 | Mission metrics required liars to reach targets, and hesitation counted only steps when every agent held. | D9, and per-agent hesitation from the gate log. |
| B6 | No re-query existed (the TTL cache reuses old messages, which is roughly the opposite), and "hold" with u = 0 coasts. | D3 and D4. |

**Process lessons:**
- **The default mission must not be rendezvous.** It drives everyone to one point, so the safety spec fails by design.
  Start from formation or flocking with separation.
- **Without a goal-directed controller there is no result.** No completion time, no hesitation cost, no trade-off curve.
  Build `Mission` early.
- **Without an unspoofable signal there is no knowledge.** Decide D5 first.
- **Without a covert attack and controllable adversary motion there is no demo.** W-MSR rejects obvious liars, and a liar
  flying the honest law hits the fleet only by luck.
- **Scenario geometry follows from the gate's numbers.** Size r_sep, lane spacing and corridor width from C. Do not tune
  them by hand.
- **Profile the epistemic layer early.** Vectorise over neighbours and cells, and make world counting optional.
- **Record everything per step from day 1** (`extras`). The dashboard, the pairing checks and the figures all depend on it.
- **Ship the rows behind every report figure** (`docs/report/data/`), so a fresh clone can redraw them without rerunning a
  6-hour sweep.

## 11. Build plan: phases, dependencies, owners

Owners follow the proposal's split, so that nobody blocks anybody. Each phase lists its exit criteria. Card numbers refer to
`TODO.md`, which keeps the full task cards.

| Owner | Area |
|---|---|
| Rithvik Jami (lead) | simulator core, graph layer, config, runner, repository and CI, integration |
| Praneeth Chamarthy | coordination primitives, stability, the λ₂ study, the switching controller |
| Gona Sanjana | attacks, W-MSR in every primitive, (r, s)-robustness, ER and degradation studies |
| Muvvala Sai Ram Suhas | the epistemic layer: worlds, S5, nested knowledge, the gate |
| Jakkala Naga Rohith Kumar | STL, specifications, dashboard, batch plots, the trade-off analysis |
| Prabhu Teja | scenarios, mission metrics, seeds, baseline validation |

### Phase 0, week 1: foundations and contracts
- Repository, subpackage layout, `pyproject.toml` with `[dev, dashboard]` extras, CI (`pytest -m "not slow"` on 3.10 and
  3.12), a Makefile (`test`, `figures`, `dashboard`), and a `slow` pytest marker.
- **Write the decisions document (section 8) and every interface in section 7 as stubs.**
- `SimConfig` with all its fields and validation; the six RNG streams; `Registry`.
- Double integrator, disk graph with drops, fixed topologies, Laplacian and λ₂; the per-receiver message cache; `World`
  (spheres, exact segment–sphere test); `SimResult` with extras and save/load.
- `safety.py` predicate (D2) and `brake_command`.
- **Exit:** a `ZeroController` run round-trips through save/load; pairing test (drops identical across two different
  controllers); the decentralisation harness exists and runs over every registered controller.

### Phase 1, weeks 2–3: coordination, and the independent building blocks
- The aggregators (`mean`, standard and trimmed `wmsr`), then consensus, formation, leader–follower, BOIDS, the
  obstacle and geofence terms, `Mission` (split into advance and safety), `switching`. T1.1–T1.7.
- `stability.py` (T1.8) and the λ₂ study (T1.9).
- Things that depend on nothing and can run in parallel: the STL engine (T3.1), the S5 Kripke engine (T4.1), `Grid`
  (T4.3), the (r, s)-robustness checker (T2.3) and the scenario generator (T5.1).
- **Exit:** with no attack and obstacles present, every honest UAV completes, ρ_safety ≥ 0 and clearance > 0.

### Phase 2, weeks 3–4: resilience and adversaries
- W-MSR wired into every primitive (T2.1), F-local checks and sampling (T2.2), the covert attack (T2.4), adversary motion
  (T2.5).
- The ER robustness study (T2.6) and the degradation study, including the honest-information-loss plot (T2.7).
- `specs_library.py` built on `safety.py` (T3.2).
- **Exit:** one liar drags unfiltered consensus off, while the filtered fleet stays in the honest hull; covert is accepted
  on ≥ 90% of steps.

### Phase 3, weeks 4–6: the epistemic layer
- Ranging (T4.2) → possible worlds and the proofs of lying (T4.4), **with the soundness invariant test** → `knows_safe`,
  checked against brute force (T4.5) → the gate and oracle (T4.7) → re-query (T4.8) → payload and nested knowledge (T4.6) →
  shared ranges and detours → the acceptance tests (T4.9).
- **Exit:** a covert liar hovering in a corridor makes the gated UAV hold and then pass, with no collision; the invariant
  holds over 50 runs; with F = 0 the gate holds only when something is truly close.

### Phase 4, weeks 6–7: scenarios, evaluation, dashboard
- Mission metrics (T5.2), baseline validation (T5.3), the frozen `demo_corridor` (T5.4), `run_pair` with asserted pairing
  (T6.1), parallel sweeps and result bookkeeping (T6.2), the trade-off study (T6.3), one figure script per report figure
  (T6.4), the dashboard (T7.1).
- **Exit:** the success table in section 2 is met; `make figures QUICK=1` finishes in under 5 min.

### Phase 5, week 8: acceptance and deliverables
- The end-to-end acceptance test (T8.1), the README with its architecture diagram, the report outline, the decisions
  document, the demo video, and `docs/story.md`. T8.2.
- **Exit:** a fresh clone runs `pip install -e .[dev,dashboard]` and `make test figures dashboard` with no errors.

**Critical path:** interfaces → Mission → ranging → worlds plus the invariant → knows_safe → gate → demo tuning → trade-off
sweep. The sweep alone took 6 h 18 min on 24 workers, so **start it by the middle of week 7 at the latest.**

## 12. Testing strategy

- **Unit tests** for every function with a closed form: λ₂ of standard graphs, the W-MSR scalar cases (fewer than F
  larger values, ties, empty neighbourhood), each STL operator on hand-computed signals of 5–10 samples, grid
  ball and shell covers, and the robustness checker on Kₙ, rings and lines.
- **Property tests:**
  - the W-MSR hull property, with ≤ F arbitrary liars;
  - the fast robustness checker agreeing with brute force on 300 random graphs;
  - the S5 axioms T, 4 and 5 on random Kripke models;
  - **the actual world in Iᵢ** (section 9);
  - `knows_safe` agreeing with brute-force enumeration;
  - whenever `k_safe[i]` is true, the realised next step being safe in ground truth;
  - a re-query never increasing the world-set size.
- **Contract tests:** decentralisation over every registered controller; pairing (an intentionally unpaired controller
  must trip `run_pair`'s assertion); bit-identity when an optional feature is off (TTL 0, ranging on or off).
- **Regression pins:** pin `res.pos[-1]` for one config and seed before any refactor of the simulator.
- **Slow tests** (`@pytest.mark.slow`): baseline validation on 3 seeds, the frozen demo inequalities, acceptance, and the
  dashboard through `streamlit.testing.v1.AppTest`.
- **Budget:** the default suite in about 2 min or less. The first build had about 870 test functions.

## 13. Experiments and the results to reproduce

Every experiment writes `results/<exp>/<YYYYmmdd-HHMMSS>/{config.json, grid.json, rows.csv, env.json}`. Every script
takes `--quick` and `--workers`. Statistics: ≥ 30 seeds per configuration, means with 95% Student's-t intervals, and
paired per-seed differences between naive and gated.

| Study | Design | First-build headline |
|---|---|---|
| λ₂ (T1.9) | summed consensus, no saturation, N ∈ {10, 20}, 6 topologies + ER(p ∈ {0.2, 0.4, 0.6}), 10 seeds, loss ∈ {0, 0.1, 0.3}; 480 runs | The rate rises with λ₂ up to about 0.9 (Spearman 0.96), then plateaus near k_p/k_v. Predicted from the full spectrum of (1−p)L: under 2% mean error. The mean of λ₂(t) is the wrong summary under loss (70% error at 30%). |
| ER robustness (T2.6) | N ∈ {10, 15, 20}, p ∈ 0.1…0.9, 30 graphs/cell, F ∈ {1, 2}, 3 attacks; 4,860 runs | (2F+1)-robust gives 100% success. It is sufficient, not necessary: about 32–35% success when it fails. |
| Degradation (T2.7) | N = 20, ER p = 0.6, assumed F ∈ {0, 1, 2} × actual {0..3}, 3 attacks, 30 seeds | 100% success while actual ≤ F, then it collapses. W-MSR discards 0 / 29 / 51% of honest messages at F = 0 / 1 / 2, costing only 12% in convergence time. |
| Baseline (T5.3) | 3 scenarios × 20 seeds, no attack, F = 0 | Gated within 0.03–0.07 s of naive; 0 collisions. |
| Demo (T5.4) | `demo_corridor`, N = 9 (8 honest + 1 covert hovering liar), 30 seeds | Naive breaks d_safe in 100% of seeds; gated in 0%. 43.7 s against 67.9 s. |
| Trade-off (T6.3) | 4 attacks × F ∈ {1, 2} × loss ∈ {0, 0.1, 0.3} × τ ∈ {2, 5, 10}·dt × h ∈ {0.5, 1, 2}·d_safe × 30 seeds, four fleets; 14,400 runs | The gated run has the higher ρ in all 6,480 pairs, at a mean cost of 30.4 s. All 97 gated violations are at τ = 0.1 s, from never-gated repulsion pushing a held UAV. The oracle gains nothing: the price is *where* the liar is, not *who*. |

**Report figures (T6.4):** λ₂ against convergence; ER robustness against success; degradation, with the information-loss
curve; the trade-off curve; demo snapshots side by side; ρ(t) for both fleets with the zero line; world-set size and
Kᵢ(Safe) over time; the W-MSR rejection heatmap.

**Compute.** Long sweeps run on the lab server. The lab has asked that heavy jobs there use the GPU and not load every CPU
core (a notice followed the CPU-heavy T6.3 run). The code is numpy-only, so either cap the number of workers or agree on a
GPU array backend with the team before the trade-off sweep. Keep world counting off in sweeps.

## 14. Engineering rules

1. Conventions: A[i, j] = i receives j; shapes are (N, d) per step and (K+1, N, d) for histories; controllers are
   decentralised; attacks falsify broadcasts only; RNG streams are paired.
2. Every registered component is imported from `aerial/__init__.py`, so that its name works from JSON.
3. Every card adds tests and leaves `pytest` green. Anything slow carries `@pytest.mark.slow`.
4. Do not rename public names. If a module moves, re-export from its old location.
5. A locked decision beats a task card. Any new decision goes in `docs/decisions.md` and in the commit message.
6. A new optional feature must be **bit-identical when off**, proven by a test.
7. Every random draw happens whatever the controller is doing. Draw the full block every step.
8. Every number quoted in a report is produced by a script from rows in the repository.

## 15. Known limits and future work

These are the honest limits of the first build. Each is a candidate improvement for the rebuild.

| Limit | Consequence | Possible fix |
|---|---|---|
| Never-gated repulsion can push a held UAV | caused all 97 gated violations at τ = 0.1 s | withhold repulsion that pushes towards a cell a liar could occupy, or add a repulsion from the liar's located position |
| A ramming liar (`intercept`) beats the gate | the gate controls only the honest UAV's own move | outside the knowledge model; state it as a limit |
| Accusations do not spread | each UAV must catch the liar itself | accept an accusation from a known-honest agent, or from F + 1 agents (both sound) |
| No handing over of work in flight | a compromised UAV's sector would go unswept | agree on the known-faulty set, then rerun the assignment rule on what is left |
| A lone UAV with a hovering liar near its route stays blocked | it needs ≥ 3 bearings to locate the liar | a wide-reaching radio (as the demo uses), or bearing sensing |
| Detours work only in x-y | a UAV cannot climb over in 3-D | a 3-D search |
| One probabilistic assumption | 9 silent ticks are taken to mean out of range (p = 1e-9) | state it |
| The map is known; survivors are found in grading | no live replanning | out of scope |
| W-MSR does nothing in the demo | the unfiltered fleet flies exactly as naive does, since flocking terms are off | show W-MSR's value in the consensus studies, or add a demo variant with formation terms on |

## 16. Parameter reference

| Symbol | Meaning | Demo value |
|---|---|---|
| N, F | UAVs (liar included), liars assumed and present | 9, 1 |
| dt, T | tick, run length | 0.05 s, 95 s |
| v_max, a_max | speed and acceleration limits | 5 m/s, 4 m/s² |
| v_ref, v_lim | cruising speed, speed governor | 2.5 m/s, 3 m/s |
| d_safe | minimum separation | 1 m |
| r_sep, s | separation reach, lane spacing | 5.51 m, 6.51 m |
| R, p_drop | radio range, packet loss | 60 m, 0.1 |
| σ, ε, ε′ | range noise (clipped at 4σ), shell tolerance, effective half-width | 0.05, 0.1, 0.2 m |
| L | silent ticks meaning out of range | 9 |
| h, τ | grid cell, look-ahead | 1 m, 0.25 s |
| C | gate clearance | 4.66 m |
| — | plan clearance from a cell centre, d_safe + h√d/2 | 1.71 m |
| patience | hold time before a detour | 1 s |
| r_detect | survivor sensor radius | 4 m |
| arena, wall | 80 m; circles of R = 6 m at x = 33, 9 m apart; corridor 26.5 m | — |
| k_goal, k_obs d₀ | seek gain, obstacle influence | 2, 5 m |
