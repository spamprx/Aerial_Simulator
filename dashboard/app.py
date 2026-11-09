"""AERIAL v2 dashboard (T4.7). Run with:  make dashboard   (needs  pip install -e .[dashboard])."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aerial.core.config import Config
from aerial.spec.mission_specs import grade
from aerial.viz import dashboard_data as dd
from aerial.viz.scene3d import scene_figure

st.set_page_config(page_title="AERIAL v2", layout="wide")
st.title("AERIAL v2: knowledge-gated search and rescue")

with st.sidebar:
    st.header("Scenario")
    scenario = st.selectbox("scenario", ["demo_district", "district", "dense_rubble", "open_field"])
    N = st.slider("N drones", 4, 20, 10)
    F = st.slider("F (assumed liars)", 0, 2, 1)
    liars = st.slider("actual liars", 0, 2, 1)
    p_drop = st.slider("packet drop p", 0.0, 0.3, 0.1, 0.05)
    R = st.slider("radio range R (m)", 30, 60, 35, 5)
    fov = st.select_slider("camera FOV (deg)", [90, 150], 90)
    lie = st.selectbox("liar's position lie", ["range_aware", "fake_sweep_naive", "radio_shadow", "true_pose"])
    q = st.slider("audit fraction q", 0.0, 0.3, 0.1, 0.05)
    pair = st.selectbox("fleets", ["naive vs gated", "unfiltered vs gated", "gated vs oracle"])
    seed = st.number_input("seed", 0, 10_000, 0)
    go_btn = st.button("run (cached)")

fleets = tuple(pair.split(" vs "))
cfg = Config(N=N, F=F)
cfg = cfg.override([f"world.scenario={scenario}", f"attack.n_compromised={min(liars, F) if F else 0}",
                    f"comms.p_drop={p_drop}", f"comms.R={R}", f"sensor.fov_h={fov}",
                    f"attack.position_lie={lie}", f"mission.audit_q={q}", f"sim.seed={seed}", "sim.T=360"])

if "runs" not in st.session_state or go_btn:
    with st.spinner("simulating both fleets (first run of a config takes a few minutes)..."):
        st.session_state.runs = dd.run_pair_cached(cfg, fleets)
runs, scn = st.session_state.runs

tabs = st.tabs(["Mission", "Knowledge", "Safety", "Coordination", "Batch"])
with tabs[0]:
    cols = st.columns(len(runs))
    for col, (name, res) in zip(cols, runs.items()):
        m = res.metrics
        with col:
            frac = dd.fraction_knowing(res)
            caught = float(res.t[1:][np.argmax(frac > 0)]) if frac.size and frac.max() > 0 else None
            st.plotly_chart(scene_figure(res, scn, title=name, caught_at=caught), use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("survivors found", f"{m['found']}/{m['survivors']}")
            b.metric("completion (s)", f"{m['completion_time']:.0f}")
            c.metric("collisions", m["collisions"])
            st.caption(f"missed in liar's claimed area: {m['missed_in_claim']} · liar known by all at "
                       f"{m.get('t_all_know')} s · false accusations: {m['false_accusations']}")

with tabs[1]:
    for name, res in runs.items():
        st.subheader(name)
        k = st.slider(f"time step ({name})", 0, max(0, len(res.t) - 2), max(0, len(res.t) - 2), key=f"k{name}")
        c1, c2 = st.columns(2)
        c1.plotly_chart(go.Figure(go.Heatmap(z=dd.knowledge_matrix(res, k), colorscale="Reds", zmin=0, zmax=1),
                                  layout=dict(title="K_i(j is a liar): rows i, cols j", height=350)),
                        use_container_width=True)
        fig = go.Figure()
        for key, v in dd.cleared_curves(res).items():
            fig.add_trace(go.Scatter(x=res.t[1:], y=v, name=key))
        fr = dd.fraction_knowing(res)
        if fr.size:
            fig.add_trace(go.Scatter(x=res.t[1:], y=fr, name="fraction knowing the liar"))
            beta, curve = dd.si_fit(res.t[1:], fr)
            fig.add_trace(go.Scatter(x=res.t[1:], y=curve, name=f"SI fit (beta={beta:.2f})", line=dict(dash="dot")))
        fig.update_layout(height=350, title="known vs believed coverage, knowledge spread")
        c2.plotly_chart(fig, use_container_width=True)
        go_ = res.extras.get("gate_open")
        if go_ is not None and len(go_):
            st.plotly_chart(go.Figure(go.Heatmap(z=go_.T.astype(float), colorscale="RdYlGn", zmin=0, zmax=1),
                                      layout=dict(title="K_i(Safe): gate open per drone over time", height=250)),
                            use_container_width=True)

with tabs[2]:
    fig = go.Figure()
    for name, res in runs.items():
        g = grade(res, scn)
        fig.add_trace(go.Scatter(x=res.t, y=g["rho_safe_t"], name=f"{name}: rho_safe(t)"))
        st.write(name, {k: round(v, 2) for k, v in g.items() if k != "rho_safe_t"})
    fig.add_hline(y=0, line=dict(color="red", dash="dash"))
    fig.update_layout(height=400, title="STL robustness of G(Safe), prefix by prefix")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    fig = go.Figure()
    for name, res in runs.items():
        fe = res.extras.get("formation_err")
        if fe is not None and len(fe):
            fig.add_trace(go.Scatter(x=res.t[1:], y=fe.max(axis=1), name=f"{name}: transit formation error"))
    fig.update_layout(height=350, title="consensus / formation error over time (transit)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    rows = dd.tradeoff_rows()
    if not rows:
        st.info("No trade-off sweep yet. Run: python scripts/exp_tradeoff.py --quick")
    else:
        fig = go.Figure()
        for fleet in sorted({r["fleet"] for r in rows}):
            rr = [r for r in rows if r["fleet"] == fleet]
            fig.add_trace(go.Scatter(x=[float(r["completion_time"]) for r in rr],
                                     y=[int(r["survivors"]) - int(r["found"]) for r in rr], mode="markers", name=fleet))
        fig.update_layout(xaxis_title="completion time (s)", yaxis_title="survivors missed", height=450)
        st.plotly_chart(fig, use_container_width=True)
