"""The search-and-rescue drone (T2.6–T2.12, T3.7–T3.13). One class serves all four fleets (V10):

=============  ============  ===========  ==========  ======  =====  =============
fleet          work alloc.   completion   aggregator  proofs  audits gate
=============  ============  ===========  ==========  ======  =====  =============
unfiltered     TrustAll      TrustAll     mean        no      no     no
naive          Belief        Belief       W-MSR       no      no     no
gated          Belief        Knowledge    W-MSR       yes     yes    yes
oracle         Knowledge*    Knowledge*   W-MSR       --      no     yes
=============  ============  ===========  ==========  ======  =====  =============

(*) the oracle's H_i holds only the true faulty set, so its knowledge is exact from t = 0.

Mission state machine (T2.8): TRANSIT -> EXPLORE <-> CONFIRM -> AUDIT -> RETURN -> LANDED, with back-edges from
AUDIT/RETURN to EXPLORE whenever uncovered work reappears (e.g. cells revert after a late catch).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..comms.messages import Draft, EventKind
from ..core.types import Action, AgentObservation
from ..epistemic.belief import AliveBelief
from ..epistemic.evidence import check_audit, check_beacons, check_camera, check_shared_range
from ..epistemic.gate import knows_safe
from ..epistemic.hypotheses import Hypotheses
from ..epistemic.knowledge import Belief, Knowledge, TrustAll
from ..mapping.maps import CoverageMap, TerrainMap
from ..planning.nav import NavGrid, coarsen, partition_labels
from ..registry import agent_registry
from ..resilience.aggregators import MeanAggregator, WMSRAggregator
from .base import AgentContext, DroneAgent

MODES = ("TRANSIT", "EXPLORE", "CONFIRM", "AUDIT", "RETURN", "LANDED")


@agent_registry.register("mission")
class MissionAgent(DroneAgent):
    def __init__(self, fleet: str = "gated", oracle_truth=None):
        self.fleet = fleet
        self.oracle_truth = oracle_truth

    # ================================================================== setup
    def reset(self, ctx: AgentContext) -> None:
        super().reset(ctx)
        cfg, st = ctx.cfg, ctx.static
        self.cfg = cfg
        self.layers = np.asarray(cfg.world.layers, float)
        self.layer_idx = int(np.argmin(np.abs(self.layers - st.layer)))
        self.hyp = Hypotheses(st.N, cfg.F, st.i, truth=self.oracle_truth if self.fleet == "oracle" else None)
        f = self.fleet
        self.detect = f == "gated"
        self.audits_on = f == "gated" and cfg.mission.audit_q > 0
        self.gate_on = f in ("gated", "oracle") or (cfg.mission.trust_gate and f in ("naive", "unfiltered"))
        self.trusting = f in ("naive", "unfiltered")
        alloc = {"unfiltered": TrustAll, "naive": Belief, "gated": Belief, "oracle": Knowledge}[f]
        comp = {"unfiltered": TrustAll, "naive": Belief, "gated": Knowledge, "oracle": Knowledge}[f]
        self.std_alloc, self.std_done = alloc(self.hyp), comp(self.hyp)
        self.agg = MeanAggregator() if f == "unfiltered" else WMSRAggregator(f=cfg.F)
        self.terrain = TerrainMap(st.arena, st.terrain_cell, self.layers, cfg.gate.d_obs)
        if st.zone_origin is None:          # mission geometry not supplied (e.g. contract tests): derive defaults
            import dataclasses
            zo0 = np.array([(st.arena - st.zone) / 2, st.arena - st.zone])
            st = dataclasses.replace(st, zone_origin=zo0,
                                     entry=np.array([zo0[0] + (st.i + 0.5) * st.zone / st.N, zo0[1] + 2.0, st.layer]),
                                     slot=np.array([st.base[0], st.base[1], st.z_min]))
            self.ctx.static = st
        zo = np.asarray(st.zone_origin, float)
        self.terrain.mark_free_rect(0, 0, st.arena, zo[1], z=0.0)        # the launch strip is known (A4)
        self.cov = CoverageMap(zo, st.coverage_cell, st.zone)
        self.nav_cell = st.coverage_cell
        k = int(round(self.nav_cell / st.terrain_cell))
        self._k = k
        self.n_nav = int(round(st.arena / self.nav_cell))
        a = np.arange(self.cov.n)
        A, B = np.meshgrid(a, a, indexing="ij")
        off = (zo / self.nav_cell).astype(int)
        self.cov_to_nav = ((A.ravel() + off[0]) * self.n_nav + (B.ravel() + off[1])).astype(int)
        self.mode = "TRANSIT"
        self.history: Dict[int, dict] = {}
        self.poses: Dict[int, tuple] = {}                 # id -> (pos, t) from beacons and POSE events
        self.tracks: Dict[int, dict] = {}                 # survivor cell -> {belief, est_own, est_others, t0}
        self.sent_cleared = np.zeros(self.cov.n_cells, bool)
        self.audited: Dict[int, set] = {}                 # teammate -> cells audited (by me or others)
        self.audited_own: Dict[int, set] = {}
        self.audit_target: Optional[tuple] = None
        self.samples: Dict[int, set] = {}
        self._sole_cache: Dict[int, list] = {}
        self._pos = np.zeros(3)
        self.proofs: List[tuple] = []
        self.accused = set()
        self.target: Optional[int] = None
        self.goal_nav: Optional[int] = None
        self.path: List[np.ndarray] = []
        self.last_plan_t = -1e9
        self.hold_time = 0.0
        self.holds = 0
        self.gate_open = True
        self.confirm_cell: Optional[int] = None
        self.confirm_t0 = 0.0
        self.done_t: Optional[float] = None
        self.formation_err = 0.0
        self._ranges_out: list = []
        self._kf_sent = 0
        self._labels = None
        self._labels_t = -1e9
        self._target_reachable = True
        self.hazard = None                                # (position, time): a proven liar seen by my camera
        self.escape = None                                # (unit direction (2,), until time)

    # ================================================================== main loop
    def step(self, obs: AgentObservation) -> Action:
        out: List[Draft] = []
        self.t = obs.t
        self._pos = obs.own.pos
        self._ingest_events(obs, out)
        self._ingest_beacons(obs)
        if obs.frame is not None:
            self._ingest_frame(obs, out)
        if self.detect:
            self._run_evidence(obs, out)
            self._track_hazards(obs)
        self._update_mode(obs)
        u, yaw_rate = self._control(obs)
        u, requery = self._apply_gate(obs, u)
        self._share(obs, out)
        return Action(u=u, yaw_rate=yaw_rate, outbox=out, requery=requery)

    # ================================================================== perception
    def _ingest_beacons(self, obs):
        for j, b in obs.beacons.items():
            self.poses[j] = (b.claim.pos.copy(), obs.t - b.age * obs.static.dt)
            if b.age == 0 and b.uwb_range is not None:
                self._ranges_out.append((j, b.seq, b.uwb_range))

    def _ingest_frame(self, obs, out):
        f = obs.frame
        self.terrain.observe(f.terrain_over, f.terrain_hit, obs.own.pos)
        surv_cells = {lk.cell for lk in f.looks}
        for lk in f.looks:
            tr = self.tracks.setdefault(lk.cell, {"belief": AliveBelief(), "est": [], "others": {}, "t0": obs.t,
                                                  "reported": False})
            tr["belief"].update(lk.heat, lk.motion)
            tr["est"].append(lk.pos_est)
            self.cov.survivor_own[lk.cell] = True
            out.append(Draft(EventKind.LOOKS, (lk.cell, lk.heat, lk.motion)))
            if not tr["reported"]:
                tr["reported"] = True
                out.append(Draft(EventKind.SURVIVOR, (lk.cell, lk.pos_est.copy())))
        v = np.array([c for c in f.viewed if c not in surv_cells], dtype=int)
        self.cov.own[v] = True
        # audits: viewing an audit cell completes it
        seen = set(f.viewed.tolist()) | surv_cells
        for j, smp in self.samples.items():
            hit = smp & seen
            if hit:
                self.audited.setdefault(j, set()).update(hit)
                out.append(Draft(EventKind.AUDITED, (j, tuple(sorted(hit)))))
        if self.audit_target is not None and self.audit_target[1] in self.audited.get(self.audit_target[0], set()):
            self.audit_target = None
        if obs.k % 10 == 0:
            self.cov.update_roofs(self.terrain, self.std_alloc)

    def _ingest_events(self, obs, out):
        me = self.i
        for e in obs.events:
            o, p = e.origin, e.payload
            if e.kind == EventKind.CLEARED:
                self.cov.cleared_rep[np.asarray(p["cells"], int)] |= np.uint64(1 << o)
            elif e.kind == EventKind.SURVIVOR:
                cell, est = p
                self.cov.survivor_rep[cell] |= np.uint64(1 << o)
                tr = self.tracks.setdefault(cell, {"belief": AliveBelief(), "est": [], "others": {}, "t0": e.t,
                                                   "reported": True})
                tr["others"][o] = np.asarray(est)
            elif e.kind == EventKind.TRACK_EST:
                cell, est = p
                if cell in self.tracks:
                    self.tracks[cell]["others"][o] = np.asarray(est)
            elif e.kind == EventKind.LOOKS:
                cell, heat, motion = p
                if self.fleet != "gated" or self.hyp.known_honest()[o]:
                    tr = self.tracks.setdefault(cell, {"belief": AliveBelief(), "est": [], "others": {}, "t0": e.t,
                                                       "reported": True})
                    tr["belief"].update(heat, motion)
            elif e.kind == EventKind.TERRAIN:
                for layer, cells in p["free"]:
                    self.terrain.add_report(o, layer, cells)
                if len(p["walls"]):
                    self.terrain.add_wall_report(o, p["walls"])
            elif e.kind == EventKind.POSE:
                if o not in self.poses or self.poses[o][1] < e.t:
                    self.poses[o] = (np.asarray(p), e.t)
            elif e.kind == EventKind.AUDITED:
                j, cells = p
                if o != j:                                     # an audit never counts for the auditee
                    self.audited.setdefault(j, set()).update(cells)
            elif e.kind == EventKind.ACCUSE and self.detect:
                j = p[0]
                self.hyp.add_clause((o, j), why=f"ACCUSE {o}->{j}")
            elif e.kind == EventKind.RANGES and self.detect:
                for (j, seq, r, p_r) in p:
                    if j != o and j != me:
                        for rule, mem in check_shared_range(o, j, seq, r, np.asarray(p_r), self.history,
                                                            self.cfg.eps_prime):
                            self.hyp.add_clause(mem, why=rule)

    def _run_evidence(self, obs, out):
        cfg = self.cfg
        proofs = check_beacons(obs, cfg.eps_prime, cfg.comms.R, cfg.drone.v_h, cfg.drone.v_z, self.history)
        proofs += check_camera(obs)
        own_surv = np.flatnonzero(self.cov.survivor_own)
        if len(own_surv):
            proofs += check_audit(own_surv, self.cov.cleared_rep, self.i)
        for rule, mem in proofs:
            j = mem[0]
            if j == self.i:
                continue
            self.hyp.add_unit(j, why=rule)
            if j not in self.accused:
                self.accused.add(j)
                self.proofs.append((obs.k, self.i, j, rule))
                out.append(Draft(EventKind.ACCUSE, (j, rule)))

    # ================================================================== knowledge summaries
    def faulty_ids(self) -> np.ndarray:
        return np.flatnonzero(self.hyp.known_faulty())

    def uncovered(self) -> np.ndarray:
        """Coverage cells still needing work under the allocation standard (belief for gated)."""
        return ~self.cov.done(self.std_alloc)

    # ================================================================== mode machine
    def _update_mode(self, obs):
        st, cfg = obs.static, self.cfg
        pos = obs.own.pos
        if self.mode == "TRANSIT":
            if np.linalg.norm(pos[:2] - st.entry[:2]) < 2.0 or obs.t > 30.0:
                self.mode = "EXPLORE"
            return
        # CONFIRM: hover over an undecided survivor until the SPRT decides or t_conf runs out
        if self.mode in ("EXPLORE", "AUDIT"):
            for cell, tr in self.tracks.items():
                if tr["est"] and tr["belief"].decision is None and obs.t - tr["t0"] < cfg.mission.t_conf:
                    self.mode, self.confirm_cell = "CONFIRM", cell
                    break
        if self.mode == "CONFIRM":
            tr = self.tracks[self.confirm_cell]
            if tr["belief"].decision is not None or obs.t - tr["t0"] >= cfg.mission.t_conf:
                self.mode = "EXPLORE"
            else:
                return
        work = self.uncovered().any()
        if self.mode in ("AUDIT", "RETURN", "LANDED") and work:
            self.mode = "EXPLORE"                               # back-edge
        if self.mode == "EXPLORE" and not work:
            self.mode = "AUDIT" if self.audits_on and self._audit_pending() else "RETURN"
        if self.mode == "AUDIT" and not self._audit_pending():
            self.mode = "RETURN"
        if self.mode == "RETURN" and np.linalg.norm(pos[:2] - st.slot[:2]) < 1.5:
            self.mode = "LANDED"
            if self.done_t is None:
                self.done_t = obs.t
        if self.mode != "LANDED":
            self.done_t = None if self.mode != "LANDED" else self.done_t

    # ---------------------------------------------------------------- audits (V3, V12)
    def _claims_of(self, j) -> np.ndarray:
        return ((self.cov.cleared_rep >> np.uint64(j)) & np.uint64(1)).astype(bool)

    def _independent(self, j) -> np.ndarray:
        """Cells someone other than j has viewed: me, another reporter, or an AUDITED event."""
        other = (self.cov.cleared_rep & ~np.uint64(1 << j)) != 0
        out = other | self.cov.own | self.cov.survivor_own | self.cov.survivor_rep.astype(bool)
        aud = self.audited.get(j)
        if aud:
            out[list(aud)] = True
        return out

    def _audit_samples(self) -> Dict[int, set]:
        """Per teammate j (not known faulty, not me): audit ceil(q n_j) of the n_j cells j claimed.

        Cells are sampled uniformly among j's claims nobody else has viewed, so a hidden survivor is caught with
        probability >= the achieved fraction. The order of visits is nearest-first (it does not change the sample).
        """
        q = self.cfg.mission.audit_q
        kf = self.hyp.known_faulty()
        for j in range(self.ctx.static.N):
            if j == self.i or kf[j]:
                self.samples.pop(j, None)
                continue
            claims = self._claims_of(j)
            n = int(claims.sum())
            if n == 0:
                continue
            indep = self._independent(j)
            smp = self.samples.setdefault(j, set())
            smp -= set(np.flatnonzero(indep).tolist())
            need = math.ceil(q * n) - int((claims & indep).sum()) - len(smp)
            if need > 0:
                pool = np.setdiff1d(np.flatnonzero(claims & ~indep & ~self.cov.abandoned), list(smp))
                if len(pool):
                    smp.update(self.ctx.rng.choice(pool, size=min(need, len(pool)), replace=False).tolist())
        return self.samples

    def _todo_audits(self):
        return {j: sorted(c - set(np.flatnonzero(self.cov.abandoned).tolist()))
                for j, c in self._audit_samples().items() if c}

    def _audit_pending(self) -> bool:
        return any(self._todo_audits().values())

    def _pick_audit(self):
        todo = {j: c for j, c in self._todo_audits().items() if c}
        if not todo:
            return None
        N = self.ctx.static.N
        j = min(todo, key=lambda t: (t - self.i) % N)          # divide the work between drones
        cells = np.array(todo[j])
        d = np.linalg.norm(self.cov.centres[cells] - self._pos[:2], axis=1)
        return int(j), int(cells[np.argmin(d)])

    def audit_report(self) -> Dict[int, dict]:
        """Achieved audit fraction per teammate, counting only my own views (the knowledge-level risk, V12)."""
        rep = {}
        mine = self.cov.own | self.cov.survivor_own
        for j in range(self.ctx.static.N):
            claims = self._claims_of(j)
            n = int(claims.sum())
            if j != self.i and n:
                k = int((claims & mine).sum())
                rep[j] = {"n": n, "audited": k, "p_catch_one": k / n}
        return rep

    # ================================================================== control
    def _control(self, obs):
        st, cfg = obs.static, self.cfg
        pos, vel = obs.own.pos, obs.own.vel
        z_goal = self.layers[self.layer_idx]
        look_at = None
        escaping = self.escape is not None and obs.t < self.escape[1] and self.mode not in ("TRANSIT", "LANDED")
        if escaping:
            d = self.escape[0]
            v_des = np.array([d[0] * 1.5, d[1] * 1.5, 0.0])
        elif self.mode == "TRANSIT":
            goal = self._formation_goal(obs)
            v_des = self._seek(pos, goal, cfg.drone.v_ref)
        elif self.mode == "CONFIRM":
            v_des = np.zeros(3)
            look_at = self.cov.centres[self.confirm_cell]
        elif self.mode == "LANDED":
            v_des = np.zeros(3)
            z_goal = st.z_min
        else:
            goal = self._navigate(obs)
            v_des = self._seek(pos, goal, cfg.drone.v_ref) if goal is not None else np.zeros(3)
            if self.mode == "RETURN":
                z_goal = st.slot[2] if np.linalg.norm(pos[:2] - st.slot[:2]) < 3 else z_goal
        v_des[2] = np.clip(1.0 * (z_goal - pos[2]), -cfg.drone.v_z, cfg.drone.v_z)
        # yaw: face the motion; look before you move
        heading = math.atan2(v_des[1], v_des[0]) if np.linalg.norm(v_des[:2]) > 0.2 else obs.own.yaw
        if look_at is not None:
            heading = math.atan2(look_at[1] - pos[1], look_at[0] - pos[0])
        yerr = (heading - obs.own.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(yerr) > math.radians(45) and self.mode != "TRANSIT":
            v_des[:2] *= 0.0
        yaw_rate = 3.0 * yerr
        u_adv = 1.5 * (v_des - vel)
        u = u_adv + self._safety(obs)
        return u, yaw_rate

    @staticmethod
    def _seek(pos, goal, v_ref):
        d = goal - pos
        d[2] = 0.0
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return np.zeros(3)
        return d / dist * min(v_ref, 0.8 * dist)

    def _formation_goal(self, obs):
        """Consensus formation tracking to this drone's entry slot (W-MSR or mean over neighbour errors)."""
        st = obs.static
        own_err = obs.own.pos[:2] - st.base[:2]
        ids, vals = [], []
        for j, b in obs.beacons.items():
            if b.age <= 2:
                ids.append(j)
                vals.append(b.claim.pos[:2] - st.base[:2])
        goal = st.entry.copy()
        if ids:
            # neighbours' formation errors relative to the shared entry line, offsets removed by projection
            res = self.agg.aggregate(own_err[1:2], np.array(vals)[:, 1:2], np.array(ids))
            lag = float(res.value[0] - own_err[1])           # >0: neighbours are ahead in y
            goal = goal.copy()
            goal[1] = min(st.entry[1], obs.own.pos[1] + 4.0 + 0.5 * lag)
            self.formation_err = abs(lag)
        return np.array([goal[0], goal[1], self.layers[self.layer_idx]])

    # ---------------------------------------------------------------- navigation
    def _nav(self):
        free = self.terrain.free(self.std_alloc, self.layer_idx)
        return NavGrid(coarsen(free, self._k), self.nav_cell)

    def _region(self, obs) -> np.ndarray:
        """Coverage cells this drone owns under the partition strategy."""
        st = obs.static
        if self.cfg.mission.partition == "static_strips":
            lab = (np.arange(self.cov.n_cells) // self.cov.n) * st.N // self.cov.n
            return lab == self.i
        if self._labels is None or obs.t - self._labels_t >= 1.0:
            faulty = set(self.faulty_ids().tolist())
            ids, pts = [self.i], [obs.own.pos[:2]]
            for j, (p, t) in sorted(self.poses.items()):
                if j in faulty or obs.t - t > 15.0:
                    continue
                ids.append(j)
                pts.append(p[:2])
            blocked = coarsen(self.terrain.known_blocked()[self.layer_idx], self._k, np.any)
            self._labels = partition_labels(blocked, self.nav_cell, np.array(pts), np.array(ids), self.cov_to_nav)
            self._labels_t = obs.t
        return self._labels == self.i

    def _navigate(self, obs) -> Optional[np.ndarray]:
        pos = obs.own.pos
        st = obs.static
        need = self.target is None or (self.mode == "EXPLORE" and self.cov.done(self.std_alloc)[self.target]) \
            or obs.t - self.last_plan_t > 4.0 + 0.1 * self.i or not self.path
        if self.mode == "AUDIT" and (self.audit_target is None or need):
            self.audit_target = self._pick_audit()
        if need:
            self._plan(obs)
        # follow the path
        while self.path and np.linalg.norm(self.path[0][:2] - pos[:2]) < 1.5:
            self.path.pop(0)
        if not self.path:
            if self.mode == "AUDIT" and self.audit_target is not None:
                c = self.audit_target[1]
                self.cov.attempts[c] += 1
                if self.cov.attempts[c] >= self.cfg.mission.view_attempts or not self._target_reachable:
                    self.cov.abandoned[c] = True
                self.audit_target = None
            if self.mode == "EXPLORE" and self.target is not None:
                self.cov.attempts[self.target] += 1
                near = np.linalg.norm(self.cov.centres[self.target] - pos[:2]) < self.cfg.sensor.r_detect * 0.8
                if near and not self._target_reachable:
                    self.cov.attempts[self.target] = self.cfg.mission.view_attempts
                if self.cov.attempts[self.target] >= self.cfg.mission.view_attempts:
                    self.cov.abandoned[self.target] = True
                self.target = None
            return None
        return self.path[0]

    def _plan(self, obs):
        pos, st = obs.own.pos, obs.static
        self.last_plan_t = obs.t
        free = coarsen(self.terrain.free(self.std_alloc, self.layer_idx), self._k)
        if self.hazard is not None and obs.t - self.hazard[1] < 30.0:    # keep 5 m clear of a liar I have seen
            n = free.shape[0]
            ij = (np.arange(n) + 0.5) * self.nav_cell
            X, Y = np.meshgrid(ij, ij, indexing="ij")
            free &= np.hypot(X - self.hazard[0][0], Y - self.hazard[0][1]) > 5.0
        nav0 = NavGrid.__new__(NavGrid)
        nav0.cell, nav0.n = self.nav_cell, free.shape[0]
        free.ravel()[nav0.cell_of(pos)] = True
        nav = NavGrid(free, self.nav_cell)
        start = nav.cell_of(pos)
        dist, pred = nav.distances(start)
        if self.mode == "RETURN":
            goal_xy = st.slot[:2]
            self.target = None
        elif self.mode == "AUDIT" and self.audit_target is not None:
            goal_xy = self.cov.centres[self.audit_target[1]]
        else:
            todo = self.uncovered()
            region = self._region(obs) & todo
            cand = np.flatnonzero(region) if region.any() else np.flatnonzero(todo)
            if len(cand) == 0:
                self.path, self.target = [], None
                return
            d_nav = dist[self.cov_to_nav[cand]]
            eu = np.linalg.norm(self.cov.centres[cand] - pos[:2], axis=1)
            score = np.where(np.isfinite(d_nav), d_nav, 1.5 * eu + 20.0)
            # hysteresis: keep the current target unless something is clearly better
            best = int(cand[np.argmin(score)])
            if self.target is not None and self.target in set(cand.tolist()):
                cur = score[np.flatnonzero(cand == self.target)[0]]
                if cur <= score.min() + 6.0:
                    best = self.target
            self.target = best
            goal_xy = self.cov.centres[best]
        # goal: the target nav cell if reachable, else the reachable cell nearest to it (frontier approach)
        gcell = nav.cell_of(goal_xy)
        self._target_reachable = bool(np.isfinite(dist[gcell]))
        if not np.isfinite(dist[gcell]):
            reach = np.flatnonzero(np.isfinite(dist))
            cxy = nav.centre(reach)
            sc = np.linalg.norm(cxy - goal_xy, axis=1) + 0.3 * dist[reach]
            gcell = int(reach[np.argmin(sc)])
        cells = nav.path(pred, gcell) or [start]
        cells = nav.shortcut(cells)
        z = self.layers[self.layer_idx]
        self.path = [np.array([*nav.centre(c), z]) for c in cells[1:]] or []
        if not self.path and self.mode == "RETURN":
            self.path = [np.array([*goal_xy, z])]

    # ---------------------------------------------------------------- safety (never gated)
    def _safety(self, obs):
        cfg = self.cfg
        pos, vel = obs.own.pos, obs.own.vel
        u = np.zeros(3)
        r_sep = cfg.gate.r_sep
        others = [b.claim.pos for b in obs.beacons.values() if b.age <= 5]
        if obs.frame is not None and len(obs.frame.sightings):
            others += list(obs.frame.sightings)
        for p in others:
            d = pos - p
            dist = np.linalg.norm(d)
            if 1e-6 < dist < r_sep:
                n = d / dist
                closing = max(0.0, -float(vel @ n))          # own velocity only
                u += n * (3.0 * (r_sep - dist) / r_sep + 1.5 * closing)
        # terrain: push away from known blocked cells at the current layer, within 3 m
        blk = self.terrain.known_blocked()[self.layer_idx]
        ij = np.floor(pos[:2] / self.terrain.cell).astype(int)
        r = 3
        i0, j0 = max(0, ij[0] - r), max(0, ij[1] - r)
        win = blk[i0: ij[0] + r + 1, j0: ij[1] + r + 1]
        if win.any():
            cells = np.argwhere(win) + [i0, j0]
            c = (cells + 0.5) * self.terrain.cell
            d = pos[:2] - c
            dist = np.linalg.norm(d, axis=1)
            k = dist < 3.0
            if k.any():
                u[:2] += (d[k] / dist[k, None] * (2.0 * (3.0 - dist[k]) / 3.0)[:, None]).sum(axis=0) / max(1, k.sum()) * 3
        # geofence
        lo, hi = 1.0, obs.static.arena - 1.0
        u[:2] += np.where(pos[:2] < lo + 2, 2.0, 0.0) - np.where(pos[:2] > hi - 2, 2.0, 0.0)
        return u

    def _track_hazards(self, obs):
        """A sighting at exactly the measured UWB range of a proven liar is the liar's real position."""
        f = obs.frame
        if f is None or not len(f.sightings):
            return
        me = obs.own.pos
        for j in np.flatnonzero(self.hyp.known_faulty()):
            b = obs.beacons.get(int(j))
            if b is None or b.uwb_range is None:
                continue
            d = np.abs(np.linalg.norm(f.sightings - me, axis=1) - b.uwb_range)
            k = int(np.argmin(d))
            if d[k] < 1.0:
                self.hazard = (f.sightings[k].copy(), obs.t)

    def _apply_gate(self, obs, u):
        if not self.gate_on or self.mode == "LANDED":
            self.gate_open = True
            return u, False
        if self.trusting:          # believes everyone: every report is true, nobody might be lying
            status_faulty = np.zeros(obs.static.N, bool)
            status_honest = np.ones(obs.static.N, bool)
        else:
            status_faulty = self.hyp.possibly_faulty()
            status_honest = ~self.hyp.known_faulty()
        safe, blocker = knows_safe(obs, u, status_honest, status_faulty, self.cfg, self.cfg.silent_ticks)
        self.gate_open = safe
        if safe:
            self.hold_time = 0.0
            return u, False
        self.holds += 1
        self.hold_time += obs.static.dt
        if self.hold_time > 3.0:            # detour: give up this target for now and replan elsewhere
            if self.target is not None:
                self.cov.attempts[self.target] += 1
            self.target, self.path, self.hold_time = None, [], 0.0
        brake = -obs.own.vel / obs.static.dt
        n = np.linalg.norm(brake)
        if n > self.cfg.drone.a_max:
            brake *= self.cfg.drone.a_max / n
        u_hold = brake + self._safety(obs)
        # a possibly-lying blocker: its claimed position may be false, so escape using what I can see
        if self.hold_time > self.cfg.gate.patience and blocker is not None and status_faulty[blocker]:
            if self.hazard is not None and obs.t - self.hazard[1] < 30.0:
                away = obs.own.pos[:2] - self.hazard[0][:2]                     # away from where I saw it
            else:
                h = obs.own.yaw + (np.pi / 2 if self.i % 2 else -np.pi / 2)   # unseen: sidestep, looking first
                away = np.array([np.cos(h), np.sin(h)])
            self.escape = (away / (np.linalg.norm(away) + 1e-9), obs.t + 2.5)
            self.path, self.hold_time = [], 0.0
        # right of way (A20): after `patience`, the lower-priority drone (higher id) backs away from an honest blocker
        elif self.hold_time > self.cfg.gate.patience and blocker is not None and self.i > blocker:
            p = obs.beacons[blocker].claim.pos
            away = obs.own.pos - p
            away[2] = 0.0
            nrm = np.linalg.norm(away)
            if nrm > 1e-6:
                u_hold = 1.5 * (away / nrm * 1.0 - obs.own.vel)
        return u_hold, True

    # ================================================================== sharing
    def _share(self, obs, out):
        k = obs.k
        new = np.flatnonzero(self.cov.own & ~self.sent_cleared)
        if len(new):
            self.sent_cleared[new] = True
            out.append(Draft(EventKind.CLEARED, {"cells": new, "pose": obs.own.pos.copy(), "t": obs.t}))
        if k % 10 == 0:
            deltas = self.terrain.new_free_cells()
            walls = self.terrain.new_wall_cells()
            if deltas or len(walls):
                out.append(Draft(EventKind.TERRAIN, {"free": deltas, "walls": walls}))
        if k % self.cfg.comms.pose_every == 0:
            out.append(Draft(EventKind.POSE, obs.own.pos.copy()))
        if self.detect and k % 5 == 0 and self._ranges_out:
            out.append(Draft(EventKind.RANGES, [(j, s, r, obs.own.pos.copy()) for j, s, r in self._ranges_out]))
        self._ranges_out = []
        kf = int(self.hyp.known_faulty().sum())
        if kf != self._kf_sent:
            self._kf_sent = kf
            out.append(Draft(EventKind.KNOWN_FAULTY, tuple(self.faulty_ids().tolist())))
        if k % 20 == 0:
            for cell, tr in self.tracks.items():
                if tr["est"]:
                    out.append(Draft(EventKind.TRACK_EST, (cell, self.track_estimate(cell))))

    def track_estimate(self, cell):
        tr = self.tracks[cell]
        own = np.mean(tr["est"], axis=0) if tr["est"] else None
        others = {j: v for j, v in tr["others"].items() if j not in set(self.faulty_ids().tolist())}
        if own is None:
            return np.mean(list(others.values()), axis=0) if others else None
        if not others:
            return own
        ids = np.array(sorted(others))
        return self.agg.aggregate(own, np.array([others[j] for j in ids]), ids).value

    # ================================================================== diagnostics
    def diagnostics(self):
        return {
            "known_faulty": self.hyp.known_faulty().copy(),
            "n_hyp": self.hyp.n_hypotheses,
            "gate_open": self.gate_open,
            "mode": MODES.index(self.mode),
            "known_cleared_frac": float(self.cov.cleared(self.std_done).mean()),
            "believed_cleared_frac": float(self.cov.cleared(self.std_alloc).mean()),
            "formation_err": self.formation_err,
        }
